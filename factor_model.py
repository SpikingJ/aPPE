import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def get_device(name=None):
    return torch.device(name or ('cuda' if torch.cuda.is_available() else 'cpu'))


def load_history(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def history_signature(H):
    return hashlib.md5(np.ascontiguousarray(H).tobytes()).hexdigest()


def fit_factors(H, d, kappa, mask=None, iters=3000, lr=0.05, device=None, seed=0, verbose=False):
    device = get_device(device) if device is None or isinstance(device, str) else device
    g = torch.Generator().manual_seed(seed)
    Ht = torch.as_tensor(H, dtype=torch.float32, device=device)
    W = None if mask is None else torch.as_tensor(mask, dtype=torch.float32, device=device)
    U = (0.1 * torch.randn(H.shape[0], d, generator=g)).to(device).requires_grad_()
    V = (0.1 * torch.randn(H.shape[1], d, generator=g)).to(device).requires_grad_()
    opt = torch.optim.Adam([U, V], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        nll = F.binary_cross_entropy_with_logits(U @ V.T, Ht, weight=W, reduction='sum')
        loss = nll + 0.5 * kappa * (U.pow(2).sum() + V.pow(2).sum())
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (it % 500 == 0 or it == iters - 1):
            print(f'  iter {it:5d}  objective {-loss.item():.1f}  (log-likelihood {-nll.item():.1f})', flush=True)
    return U.detach().double().cpu().numpy(), V.detach().double().cpu().numpy(), -loss.item()


def log_loss(logits, h):
    return np.mean(h * np.logaddexp(0, -logits) + (1 - h) * np.logaddexp(0, logits))


def cross_validate(H, d_grid, kappa_grid, holdout=0.1, seed=0, **fit_kw):
    test = np.random.default_rng(seed).random(H.shape) < holdout
    rows = []
    for d in d_grid:
        for kappa in kappa_grid:
            U, V, _ = fit_factors(H, d, kappa, mask=~test, seed=seed, **fit_kw)
            logits, h = (U @ V.T)[test], H[test]
            rows.append(dict(d=d, kappa=kappa, logloss=log_loss(logits, h), acc=np.mean((logits > 0) == h)))
            print(f"  d={d:3d} kappa={kappa:<6g} held-out log-loss {rows[-1]['logloss']:.4f}, "
                  f"accuracy {rows[-1]['acc']:.4f}", flush=True)
    return rows


class FactorModel:
    def __init__(self, U, V, kappa, signature=''):
        self.U, self.V, self.kappa, self.signature = U, V, kappa, signature
        self.d = V.shape[1]
        self.mu0 = U.mean(axis=0)
        self.Sigma0 = np.atleast_2d(np.cov(U, rowvar=False, bias=True))

    def save(self, path):
        np.savez(path, U=self.U, V=self.V, kappa=self.kappa, signature=self.signature)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(f['U'], f['V'], float(f['kappa']), str(f['signature']))

    def tensors(self, device):
        as_t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
        return as_t(self.V), as_t(self.mu0), as_t(self.Sigma0)


def get_factor_model(history, path, d=16, kappa=1.0, device=None, **fit_kw):
    sig = history_signature(history['H'])
    if path and os.path.exists(path):
        fm = FactorModel.load(path)
        if fm.signature != sig:
            raise ValueError(f'{path} was fitted on a different H; delete it or pass another --factor path')
        return fm
    print(f"fitting the factor model (d={d}, kappa={kappa}) on H of shape {history['H'].shape}", flush=True)
    U, V, _ = fit_factors(history['H'], d, kappa, device=device, **fit_kw)
    fm = FactorModel(U, V, kappa, sig)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fm.save(path)
    return fm


class CandidatePosterior:
    def __init__(self, V, mu0, Sigma0, R):
        self.V = V
        self.mu = mu0.expand(R, -1).clone()
        self.Sigma = Sigma0.expand(R, -1, -1).clone()
        self.vSv = ((V @ Sigma0) * V).sum(1).expand(R, -1).clone()
        self.rows = torch.arange(R, device=V.device)
        self.p = torch.sigmoid(self.mu @ V.T)

    def update(self, I, k, active=None):
        v = self.V[I]
        p = self.p[self.rows, I]
        w = p * (1 - p)
        if active is not None:
            w = w * active
        b = torch.bmm(self.Sigma, v.unsqueeze(-1)).squeeze(-1)
        s = (v * b).sum(1)
        c = w / (1 + w * s)
        self.Sigma -= c[:, None, None] * b[:, :, None] * b[:, None, :]
        self.vSv -= c[:, None] * (b @ self.V.T) ** 2
        self.vSv.clamp_(min=0)
        gain = (k - p) / (1 + w * s)
        if active is not None:
            gain = gain * active
        self.mu += gain[:, None] * b
        self.p = torch.sigmoid(self.mu @ self.V.T)


class HybridPolicy:
    def __init__(self, tau=0.05, beta0=1.0, rho=0.25, gamma=0.25):
        self.tau, self.beta0, self.rho, self.gamma = tau, beta0, rho, gamma

    def schedule(self, t, T_max):
        alpha = max(0.0, 1 - t / (self.rho * T_max)) if self.rho > 0 else 0.0
        beta = self.beta0 * min(1.0, t / (self.gamma * T_max)) if self.gamma > 0 else self.beta0
        return alpha, beta

    def __call__(self, post, t, T_max):
        p = post.p
        N = p.shape[1]
        if self.tau >= 1:
            return torch.full_like(p, 1.0 / N)
        alpha, beta = self.schedule(t, T_max)
        w = (p * (1 - p)).clamp_min(1e-300)
        h = torch.sqrt(w)
        h = h / h.sum(1, keepdim=True)
        if alpha > 0:
            r = w @ post.V / N
            proj = torch.bmm(post.Sigma, r.unsqueeze(-1)).squeeze(-1) @ post.V.T
            log_sa = torch.log(w) + 2 * torch.log(proj.abs().clamp_min(1e-300)) - torch.log1p(w * post.vSv)
            h = (1 - alpha) * h + alpha * torch.softmax(log_sa, dim=1)
        h = h ** beta
        return self.tau / N + (1 - self.tau) * h / h.sum(1, keepdim=True)


def add_common_args(ap):
    ap.add_argument('--history', default='outputs/history.npz', help='output of build_history.py')
    ap.add_argument('--factor', default='outputs/factor.npz',
                    help='fitted factor model (factor_model.py); fitted with --d/--kappa if missing')
    ap.add_argument('--d', type=int, default=16, help='latent dimension if the factor model is fitted here')
    ap.add_argument('--kappa', type=float, default=1.0, help='regularization if the factor model is fitted here')
    ap.add_argument('--candidates', nargs='*', default=None, help='candidate indices or names (default: all)')
    ap.add_argument('--trials', type=int, default=200, help='independent evaluation runs per candidate')
    ap.add_argument('--chunk', type=int, default=200, help='runs simulated in parallel')
    ap.add_argument('--delta', type=float, default=0.05, help='target error probability')
    ap.add_argument('--beta0', type=float, default=1.0, help='tempering strength beta_0, eq. (30)')
    ap.add_argument('--rho', type=float, default=0.25, help='active-learning schedule rho, eq. (29)')
    ap.add_argument('--gamma', type=float, default=0.25, help='tempering schedule gamma, eq. (30)')
    ap.add_argument('--no_repeat_update', action='store_true',
                    help='update the posterior only the first time a sample is drawn')
    ap.add_argument('--save_traj', type=int, default=0, help='save the interval path of the first n runs')
    ap.add_argument('--device', default=None, help='default: cuda if available, else cpu')
    ap.add_argument('--seed', type=int, default=0)


def select_candidates(history, spec):
    names = [str(n) for n in history['cand_names']]
    if not spec:
        return list(range(len(names)))
    return [int(s) if s.isdigit() else names.index(s) for s in spec]


def resolve_rounds(value, N):
    return max(1, int(round(value * N))) if value <= 1 else int(value)


def summarize(df, keys):
    per_cand = df.groupby(keys + ['candidate'], dropna=False).agg(
        coverage=('covered', 'mean'), width=('width', 'mean'), T=('T', 'mean'),
        n_unique=('n_unique', 'mean')).reset_index()
    return per_cand.groupby(keys, dropna=False).agg(
        coverage=('coverage', 'mean'), min_coverage=('coverage', 'min'), width=('width', 'mean'),
        T=('T', 'mean'), n_unique=('n_unique', 'mean'), n_candidates=('candidate', 'count')).reset_index()


def diagnose(fm, history, n_obs=500, seed=0):
    V, mu0, Sigma0 = fm.tensors('cpu')
    g = torch.Generator().manual_seed(seed)
    rows = []
    for name, k in zip(history['cand_names'], history['C']):
        k = torch.as_tensor(k, dtype=torch.float64)
        post = CandidatePosterior(V, mu0, Sigma0, 1)
        prior = post.p.mean().item()
        for I in torch.randint(len(k), (n_obs,), generator=g):
            post.update(I.view(1), k[I].view(1))
        rows.append(dict(candidate=str(name), K_star=k.mean().item(), prior_pred=prior,
                         post_pred=post.p.mean().item(), post_mae=(post.p[0] - k).abs().mean().item()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--history', default='outputs/history.npz')
    ap.add_argument('--out', default='outputs/factor.npz')
    ap.add_argument('--d', type=int, default=16, help='latent dimension')
    ap.add_argument('--kappa', type=float, default=1.0, help='regularization kappa in eq. (26)')
    ap.add_argument('--cv', action='store_true', help='choose (d, kappa) on held-out entries of H')
    ap.add_argument('--d_grid', type=int, nargs='+', default=[2, 4, 8, 16, 32])
    ap.add_argument('--kappa_grid', type=float, nargs='+', default=[0.1, 1.0, 10.0])
    ap.add_argument('--holdout', type=float, default=0.1, help='share of held-out entries for --cv')
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    history = load_history(args.history)
    H = history['H']
    device = get_device(args.device)
    print(f'H: {H.shape[0]} models x {H.shape[1]} samples')
    d, kappa = args.d, args.kappa
    if args.cv:
        rows = cross_validate(H, args.d_grid, args.kappa_grid, args.holdout, args.seed,
                              iters=args.iters, lr=args.lr, device=device)
        best = min(rows, key=lambda r: r['logloss'])
        d, kappa = best['d'], best['kappa']
        print(f'selected d={d}, kappa={kappa:g}')
    U, V, _ = fit_factors(H, d, kappa, iters=args.iters, lr=args.lr, device=device, seed=args.seed, verbose=True)
    fm = FactorModel(U, V, kappa, history_signature(H))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fm.save(args.out)

    logits = U @ V.T
    row_err = np.abs((1 / (1 + np.exp(-logits))).mean(1) - H.mean(1))
    print(f'fit on H: log-loss {log_loss(logits, H):.4f}, accuracy {np.mean((logits > 0) == H):.4f}, '
          f'max |predicted - true accuracy| over historical models {row_err.max():.4f}')
    print('transfer to the candidates (predicted performance vs true K*):')
    print(diagnose(fm, history, seed=args.seed).to_string(index=False, float_format='%.4f'))
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
