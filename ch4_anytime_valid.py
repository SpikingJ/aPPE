import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from factor_model import (CandidatePosterior, HybridPolicy, add_common_args, get_device, get_factor_model,
                          load_history, resolve_rounds, select_candidates, summarize)


def run_trials(k, fm_t, T_max, R, policy, reliance, etas, weights, delta, eps, widths, grid_size, gen,
               update_repeats=True, n_traj=0, bisect_iters=24, run_until=1, cost=None):
    V, mu0, Sigma0 = fm_t
    N, dev, tau = k.numel(), k.device, policy.tau
    f64 = dict(dtype=torch.float64, device=dev)
    rho = torch.tensor(reliance, **f64)
    n_eta = len(etas)
    rho_c = rho.repeat_interleave(n_eta)
    eta_c = torch.tensor(etas, **f64).repeat(len(reliance))
    lam_L = eta_c * tau / (tau + rho_c)
    lam_U = eta_c * tau / (1 + rho_c * tau)
    log_w = torch.log(torch.tensor(weights, **f64).repeat_interleave(n_eta) / n_eta)
    log_c_L, log_c_U = -math.log(eps * delta), -math.log((1 - eps) * delta)
    K_lo, K_hi = -rho / tau, rho + 1 / tau

    G, step = grid_size, 1.0 / (grid_size - 1)
    grid = torch.linspace(0, 1, G, **f64)
    ar = torch.arange(G, device=dev)
    logE_L = torch.zeros(R, len(rho_c), G, **f64)
    logE_U = torch.zeros_like(logE_L)
    logEbar_L = torch.zeros(R, G, **f64)
    logEbar_U = torch.zeros_like(logEbar_L)
    K_hist = torch.empty(R, T_max, len(reliance), **f64)
    zeros, ones = torch.zeros(R, **f64), torch.ones(R, **f64)

    def log_Ebar(Kh, K0, lam, sign):
        logE = torch.cumsum(torch.log1p(sign * lam * (Kh - K0[:, None, None])), dim=1) + log_w
        return torch.logsumexp(logE, dim=2).max(dim=1).values.clamp_min(0)

    def endpoints(runs, t, a_L, a_U):
        Kh = K_hist[runs, :t].repeat_interleave(n_eta, dim=2)
        lo_L, hi_L = grid[a_L.clamp(0, G - 2)], grid[(a_L + 1).clamp(1, G - 1)]
        lo_U, hi_U = grid[(a_U - 1).clamp(0, G - 2)], grid[a_U.clamp(1, G - 1)]
        for _ in range(bisect_iters):
            mid = (lo_L + hi_L) / 2
            rej = log_Ebar(Kh, mid, lam_L, 1.0) >= log_c_L
            lo_L, hi_L = torch.where(rej, mid, lo_L), torch.where(rej, hi_L, mid)
            mid = (lo_U + hi_U) / 2
            rej = log_Ebar(Kh, mid, lam_U, -1.0) >= log_c_U
            lo_U, hi_U = torch.where(rej, lo_U, mid), torch.where(rej, mid, hi_U)
        B = len(runs)
        L = torch.where(a_L < 0, zeros[:B], torch.where(a_L >= G - 1, ones[:B], lo_L))
        U = torch.where(a_U >= G, ones[:B], torch.where(a_U <= 0, zeros[:B], hi_U))
        return L, U

    post = CandidatePosterior(V, mu0, Sigma0, R)
    rows = post.rows
    seen = torch.zeros(R, N, dtype=torch.bool, device=dev)
    n_seen = torch.zeros(R, dtype=torch.long, device=dev)
    c_seen = torch.zeros(R, **f64)
    cost = torch.zeros(N, **f64) if cost is None else torch.as_tensor(cost, **f64)
    Wv = torch.tensor(widths, **f64)[:, None]
    done = torch.zeros(len(widths), R, dtype=torch.bool, device=dev)
    rec_T = torch.full((len(widths), R), T_max, device=dev)
    rec_n = torch.zeros(len(widths), R, dtype=torch.long, device=dev)
    rec_c = torch.zeros(len(widths), R, **f64)
    rec_L, rec_U = torch.zeros(len(widths), R, **f64), torch.ones(len(widths), R, **f64)
    traj = []
    for t in range(1, T_max + 1):
        p = post.p
        q = policy(post, t, T_max)
        I = torch.multinomial(q, 1, generator=gen).squeeze(1)
        kI, pI, qI = k[I], p[rows, I], q[rows, I]
        K_hat = rho * p.mean(1, keepdim=True) + (kI[:, None] - rho * pI[:, None]) / (N * qI[:, None])
        assert bool(((K_hat >= K_lo - 1e-9) & (K_hat <= K_hi + 1e-9)).all()), 'bound (15) violated'
        K_hist[:, t - 1] = K_hat

        diff = K_hat.repeat_interleave(n_eta, dim=1)[:, :, None] - grid
        logE_L += torch.log1p(lam_L[:, None] * diff)
        logE_U += torch.log1p(-lam_U[:, None] * diff)
        torch.maximum(logEbar_L, torch.logsumexp(logE_L + log_w[:, None], dim=1), out=logEbar_L)
        torch.maximum(logEbar_U, torch.logsumexp(logE_U + log_w[:, None], dim=1), out=logEbar_U)

        a_L = torch.where(logEbar_L >= log_c_L, ar, -1).max(1).values
        a_U = torch.where(logEbar_U >= log_c_U, ar, G).min(1).values
        L = torch.where(a_L >= 0, grid[a_L.clamp_min(0)], zeros)
        U = torch.where(a_U < G, grid[a_U.clamp_max(G - 1)], ones)
        need = (~done & (U - L - 2 * step < Wv)).any(0)
        if t == T_max:
            need |= (~done).any(0)
        if need.any():
            runs = need.nonzero().squeeze(1)
            L[runs], U[runs] = endpoints(runs, t, a_L[runs], a_U[runs])

        new = ~seen[rows, I]
        n_seen += new
        c_seen += new * cost[I]
        seen[rows, I] = True
        post.update(I, kI, None if update_repeats else new)

        if n_traj:
            traj.append(torch.stack([L[:n_traj], U[:n_traj]]))
        hit = ~done & ((U - L).clamp_min(0) <= Wv)
        done |= hit
        rec_T = torch.where(hit, t, rec_T)
        rec_n = torch.where(hit, n_seen, rec_n)
        rec_c = torch.where(hit, c_seen, rec_c)
        rec_L, rec_U = torch.where(hit, L, rec_L), torch.where(hit, U, rec_U)
        if bool(done.all()) and t >= run_until:
            break

    out = []
    for i, W in enumerate(widths):
        d = done[i]
        res = dict(T=torch.where(d, rec_T[i], t), stopped=d, n_unique=torch.where(d, rec_n[i], n_seen),
                   cost=torch.where(d, rec_c[i], c_seen), L=torch.where(d, rec_L[i], L), U=torch.where(d, rec_U[i], U))
        out.append((W, {key: v.cpu().numpy() for key, v in res.items()}))
    return out, (torch.stack(traj, -1).cpu().numpy() if traj else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument('--T_max', type=float, default=1.0,
                    help='maximum number of rounds in (2), also T_bar in (29)-(30); <= 1: fraction of N (default N)')
    ap.add_argument('--widths', type=float, nargs='+', default=[0.1, 0.05, 0.02], help='target widths W_th')
    ap.add_argument('--tau', type=float, default=0.5, help='exploration probability tau in eq. (12)')
    ap.add_argument('--reliance', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help='reliance levels rho_1, ..., rho_S in [0, 1]')
    ap.add_argument('--weights', type=float, nargs='*', default=None,
                    help='mixture weights w_s of the reliance levels, eq. (18) (default: uniform)')
    ap.add_argument('--eta', type=float, nargs='+', default=[0.5], help='stake fraction eta')
    ap.add_argument('--eps', type=float, default=0.5, help='error split epsilon in eq. (17)')
    ap.add_argument('--grid', type=int, default=4001, help='K0 grid points used to bracket L_t and U_t')
    ap.add_argument('--run_until', type=float, default=0,
                    help='keep simulating for at least this many rounds after all W_th are reached, so that the '
                         'paths saved with --save_traj have a common length (<= 1: fraction of N)')
    ap.add_argument('--out', default='results/ch4.csv')
    args = ap.parse_args()

    weights = args.weights or [1.0 / len(args.reliance)] * len(args.reliance)
    assert len(weights) == len(args.reliance) and abs(sum(weights) - 1) < 1e-9 and min(weights) >= 0
    assert all(0 <= r <= 1 for r in args.reliance) and all(0 < e < 1 for e in args.eta)
    assert 0 < args.tau <= 1 and 0 < args.eps < 1

    device = get_device(args.device)
    history = load_history(args.history)
    fm = get_factor_model(history, args.factor, args.d, args.kappa, device=device)
    fm_t = fm.tensors(device)
    N = history['C'].shape[1]
    T_max = resolve_rounds(args.T_max, N)
    run_until = min(resolve_rounds(args.run_until, N), T_max) if args.run_until > 0 else 1
    widths = sorted(args.widths, reverse=True)
    policy = HybridPolicy(args.tau, args.beta0, args.rho, args.gamma)
    cands = select_candidates(history, args.candidates)
    print(f'N = {N}, d = {fm.d}, T_max = {T_max}, W_th {widths}, reliance {args.reliance}, eta {args.eta}, '
          f'tau {args.tau}; {len(cands)} candidates x {args.trials} runs, device {device}')

    frames, trajs = [], {}
    for c in cands:
        name = str(history['cand_names'][c])
        k = torch.as_tensor(history['C'][c], dtype=torch.float64, device=device)
        K_star = k.mean().item()
        cost = history['cost_C'][c] if 'cost_C' in history else None
        t0 = time.time()
        cur = []
        for start in range(0, args.trials, args.chunk):
            R = min(args.chunk, args.trials - start)
            gen = torch.Generator(device=device).manual_seed(args.seed + 1_000_003 * c + start)
            n_traj = min(args.save_traj - start, R) if start < args.save_traj else 0
            out, traj = run_trials(k, fm_t, T_max, R, policy, args.reliance, args.eta, weights, args.delta,
                                   args.eps, widths, args.grid, gen, not args.no_repeat_update, n_traj,
                                   run_until=run_until, cost=cost)
            cur += [pd.DataFrame(dict(candidate=name, K_star=K_star, W_th=W, trial=np.arange(start, start + R), **res))
                    for W, res in out]
            if traj is not None:
                trajs.setdefault(name, []).append(traj)
        frames += cur
        cur = pd.concat(cur)
        stats = ', '.join(f"W={W:g}: cov {np.mean((g.L <= K_star) & (K_star <= g.U)):.3f} T {g['T'].mean():.0f}"
                          for W, g in cur.groupby('W_th', sort=False))
        print(f'[{name}] K* = {K_star:.4f}  {stats}  ({time.time() - t0:.1f}s)', flush=True)

    df = pd.concat(frames, ignore_index=True)
    df['width'] = df.U - df.L
    df['covered'] = (df.L <= df.K_star) & (df.K_star <= df.U)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = summarize(df, ['W_th'])
    summary['stopped'] = df.groupby('W_th').stopped.mean().values
    base = os.path.splitext(args.out)[0]
    summary.to_csv(base + '_summary.csv', index=False)
    with open(base + '_config.json', 'w') as f:
        json.dump(dict(vars(args), T_max_rounds=T_max, N=N, d=fm.d, weights=weights), f, indent=1)
    if trajs:
        pad = lambda a, n: np.pad(a, ((0, 0), (0, 0), (0, n - a.shape[2])), constant_values=np.nan)
        np.savez_compressed(base + '_traj.npz', **{key: np.concatenate([pad(a, max(b.shape[2] for b in v)) for a in v], 1)
                                                   for key, v in trajs.items()})
    print(f'\nguaranteed coverage >= {1 - args.delta:.3f}')
    print(summary.to_string(index=False, float_format='%.4f'))
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
