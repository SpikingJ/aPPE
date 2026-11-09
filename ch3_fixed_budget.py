import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from factor_model import (CandidatePosterior, HybridPolicy, add_common_args, get_device, get_factor_model,
                          load_history, resolve_rounds, select_candidates, summarize)


def clt_interval(acc, t, N, z):
    K_hat = acc['K_sum'] / t
    S_z = acc['ipw_k'] / t
    centred = S_z ** 2 - 2 * S_z * acc['Sp'] / t + acc['Sp2'] / t
    second = acc['aipw2'] / t
    var = (second - centred) / N ** 2
    var = torch.where(var > 1e-12 * second / N ** 2, var, torch.zeros_like(var))
    half = z * torch.sqrt(var / t)
    return K_hat - half, K_hat + half, K_hat


def run_trials(k, fm_t, T_max, R, policy, z, widths, min_rounds, gen, no_prediction=False,
               update_repeats=True, n_traj=0, cost=None):
    V, mu0, Sigma0 = fm_t
    N, dev = k.numel(), k.device
    post = CandidatePosterior(V, mu0, Sigma0, R)
    rows = post.rows
    acc = {key: torch.zeros(R, dtype=torch.float64, device=dev) for key in ('K_sum', 'aipw2', 'ipw_k', 'Sp', 'Sp2')}
    seen = torch.zeros(R, N, dtype=torch.bool, device=dev)
    n_seen = torch.zeros(R, dtype=torch.long, device=dev)
    c_seen = torch.zeros(R, dtype=torch.float64, device=dev)
    cost = torch.zeros(N, dtype=torch.float64, device=dev) if cost is None else \
        torch.as_tensor(cost, dtype=torch.float64, device=dev)
    peek = [dict(W=W, done=torch.zeros(R, dtype=torch.bool, device=dev), T=torch.full((R,), T_max, device=dev),
                 n=n_seen.clone(), c=c_seen.clone(), L=torch.zeros(R, dtype=torch.float64, device=dev),
                 U=torch.ones(R, dtype=torch.float64, device=dev), K=torch.zeros(R, dtype=torch.float64, device=dev))
            for W in widths]
    traj = []
    for t in range(1, T_max + 1):
        p = torch.zeros_like(post.p) if no_prediction else post.p
        q = policy(post, t, T_max)
        I = torch.multinomial(q, 1, generator=gen).squeeze(1)
        kI, pI, qI = k[I], p[rows, I], q[rows, I]
        Sp = p.sum(1)
        aipw = (kI - pI) / qI
        acc['K_sum'] += (Sp + aipw) / N
        acc['aipw2'] += aipw ** 2
        acc['ipw_k'] += kI / qI
        acc['Sp'] += Sp
        acc['Sp2'] += Sp ** 2
        new = ~seen[rows, I]
        n_seen += new
        c_seen += new * cost[I]
        seen[rows, I] = True
        if not no_prediction:
            post.update(I, kI, None if update_repeats else new)

        if (widths and t >= min_rounds) or n_traj:
            L, U, K = clt_interval(acc, t, N, z)
            if n_traj:
                traj.append(torch.stack([L[:n_traj], U[:n_traj], K[:n_traj]]))
            if widths and t >= min_rounds:
                for rec in peek:
                    hit = ~rec['done'] & (U - L <= rec['W']) & (U - L > 0)
                    rec['done'] |= hit
                    rec['T'] = torch.where(hit, t, rec['T'])
                    for key, val in (('n', n_seen), ('c', c_seen), ('L', L), ('U', U), ('K', K)):
                        rec[key] = torch.where(hit, val, rec[key])

    L, U, K = clt_interval(acc, T_max, N, z)
    out = [('fixed', np.nan, dict(T=torch.full((R,), T_max), n_unique=n_seen, cost=c_seen, K_hat=K, L=L, U=U))]
    for rec in peek:
        d = rec['done']
        out.append(('peek', rec['W'], dict(T=rec['T'], n_unique=torch.where(d, rec['n'], n_seen),
                                           cost=torch.where(d, rec['c'], c_seen),
                                           K_hat=torch.where(d, rec['K'], K), L=torch.where(d, rec['L'], L),
                                           U=torch.where(d, rec['U'], U))))
    out = [(mode, W, {key: v.cpu().numpy() for key, v in res.items()}) for mode, W, res in out]
    return out, (torch.stack(traj, -1).cpu().numpy() if traj else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument('--budgets', type=float, nargs='+', default=[0.025, 0.05, 0.1, 0.2],
                    help='fixed numbers of rounds T_bar <= N (values <= 1 are fractions of N)')
    ap.add_argument('--tau', type=float, default=0.05, help='exploration probability tau in eq. (12)')
    ap.add_argument('--stop_widths', type=float, nargs='*', default=[],
                    help="W_th values for the optional 'peek' rows (misuse discussed in Sec. IV; not part of PPE)")
    ap.add_argument('--min_rounds', type=int, default=30, help="first round at which 'peek' may stop")
    ap.add_argument('--no_prediction', action='store_true', help='uniform sampling, no prediction baseline')
    ap.add_argument('--out', default='results/ch3.csv')
    args = ap.parse_args()

    device = get_device(args.device)
    history = load_history(args.history)
    fm = get_factor_model(history, args.factor, args.d, args.kappa, device=device)
    fm_t = fm.tensors(device)
    N = history['C'].shape[1]
    budgets = sorted({resolve_rounds(b, N) for b in args.budgets})
    policy = HybridPolicy(1.0 if args.no_prediction else args.tau, args.beta0, args.rho, args.gamma)
    z = norm.ppf(1 - args.delta / 2)
    cands = select_candidates(history, args.candidates)
    print(f'N = {N}, d = {fm.d}, budgets {budgets}, {len(cands)} candidates x {args.trials} runs, device {device}')

    frames, trajs = [], {}
    for c in cands:
        name = str(history['cand_names'][c])
        k = torch.as_tensor(history['C'][c], dtype=torch.float64, device=device)
        K_star = k.mean().item()
        cost = history['cost_C'][c] if 'cost_C' in history else None
        for T_max in budgets:
            t0 = time.time()
            cur = []
            for start in range(0, args.trials, args.chunk):
                R = min(args.chunk, args.trials - start)
                gen = torch.Generator(device=device).manual_seed(args.seed + 1_000_003 * c + 7_919 * T_max + start)
                n_traj = min(args.save_traj - start, R) if start < args.save_traj else 0
                out, traj = run_trials(k, fm_t, T_max, R, policy, z, args.stop_widths, args.min_rounds, gen,
                                       args.no_prediction, not args.no_repeat_update, n_traj, cost)
                cur += [pd.DataFrame(dict(candidate=name, K_star=K_star, mode=mode, T_max=T_max, W_th=W,
                                          trial=np.arange(start, start + R), **res)) for mode, W, res in out]
                if traj is not None:
                    trajs.setdefault(f'{name}|T{T_max}', []).append(traj)
            frames += cur
            fixed = pd.concat([f for f in cur if f['mode'].iat[0] == 'fixed'])
            print(f'[{name}] K* = {K_star:.4f}  T_max = {T_max}: fixed-budget coverage '
                  f'{np.mean((fixed.L <= K_star) & (K_star <= fixed.U)):.3f}, width {np.mean(fixed.U - fixed.L):.4f} '
                  f'({time.time() - t0:.1f}s)', flush=True)

    df = pd.concat(frames, ignore_index=True)
    df['width'] = df.U - df.L
    df['covered'] = (df.L <= df.K_star) & (df.K_star <= df.U)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = summarize(df, ['mode', 'T_max', 'W_th'])
    base = os.path.splitext(args.out)[0]
    summary.to_csv(base + '_summary.csv', index=False)
    with open(base + '_config.json', 'w') as f:
        json.dump(dict(vars(args), budgets_rounds=budgets, N=N, d=fm.d), f, indent=1)
    if trajs:
        np.savez_compressed(base + '_traj.npz', **{key: np.concatenate(v, 1) for key, v in trajs.items()})
    print(f'\nnominal coverage {1 - args.delta:.3f}')
    print(summary.to_string(index=False, float_format='%.4f'))
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
