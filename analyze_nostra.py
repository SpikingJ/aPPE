import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import C_CH4, C_RHO1, INK, MathScalarFormatter, save, style


def cost_table(results, histories, tbar=400):
    rows = []
    for d in sorted(glob.glob(os.path.join(results, '*', 'ch4.csv'))):
        run = os.path.basename(os.path.dirname(d))
        h = np.load(os.path.join(histories, f'history_{run}.npz'))
        full = dict(zip(h['cand_names'].astype(str), h['cost_C'].sum(1) / 3600))
        ch4 = pd.read_csv(d)
        read = lambda name: (pd.read_csv(os.path.join(os.path.dirname(d), name))
                             if os.path.exists(os.path.join(os.path.dirname(d), name)) else None)
        rho1, ch3 = read('ch4_rho1.csv'), read('ch3.csv')
        for (cand, W), g in ch4.groupby(['candidate', 'W_th']):
            row = dict(run=run, candidate=cand, W_th=W, K_star=g.K_star.iat[0], full_h=full[cand],
                       appe_h=g.cost.mean() / 3600, appe_cov=g.covered.mean(), appe_T=g['T'].mean(),
                       appe_n=g.n_unique.mean(), appe_stopped=g.stopped.mean())
            if rho1 is not None:
                r = rho1[(rho1.candidate == cand) & np.isclose(rho1.W_th, W)]
                row.update(s1_h=r.cost.mean() / 3600, s1_cov=r.covered.mean(), s1_T=r['T'].mean(),
                           s1_n=r.n_unique.mean(), s1_stopped=r.stopped.mean())
            if ch3 is not None:
                f = ch3[(ch3['mode'] == 'fixed') & (ch3.candidate == cand) & (ch3.T_max == tbar)]
                if len(f):
                    row.update(ppe_fixed_h=f.cost.mean() / 3600, ppe_fixed_width=f.width.mean(),
                               ppe_fixed_cov=f.covered.mean())
                p = ch3[(ch3['mode'] == 'peek') & (ch3.candidate == cand) & np.isclose(ch3.W_th, W)
                        & (ch3.T_max == ch3.T_max.max())]
                if len(p):
                    row.update(ppe_peek_h=p.cost.mean() / 3600, ppe_peek_cov=p.covered.mean(), ppe_peek_T=p['T'].mean())
            rows.append(row)
    df = pd.DataFrame(rows)
    df['appe_savings'] = 1 - df.appe_h / df.full_h
    if 's1_h' in df:
        df['s1_savings'] = 1 - df.s1_h / df.full_h
    return df


def fig_savings(df, out, candidates=None, widths=None, y_max=6):
    d = df[~df.run.str.endswith('_checkpoints')]
    if candidates:
        d = d[d.candidate.isin(candidates)]
    if widths:
        d = d[np.isclose(d.W_th.values[:, None], widths).any(1)]
    g = d.groupby('W_th', sort=True)
    Ws = np.array(sorted(d.W_th.unique()))
    mean = lambda col: g[col].mean().reindex(Ws).values
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(Ws))
    bars = [(r'aPPE, $S=1$, $\rho_1 = 1$', mean('s1_h'), mean('s1_stopped'), C_RHO1),
            (r'aPPE, $S=5$', mean('appe_h'), mean('appe_stopped'), C_CH4)]
    w = 0.34
    for i, (lab, v, reached, col) in enumerate(bars):
        part = reached < 0.99
        xs = x + (i - 0.5) * w
        ax.bar(xs, v, w * 0.92, color=col, label=lab, zorder=2)
        ax.bar(xs[part], v[part], w * 0.92, color='none', edgecolor='white', hatch='////', lw=0, zorder=3)
        for xi, vi, ri in zip(xs[part], v[part], reached[part]):
            ax.text(xi, min(vi, y_max) * 0.5, f'${100 * ri:.0f}\\%$', color='white', fontsize=6.5, ha='center',
                    va='center', rotation=90, zorder=4, bbox=dict(boxstyle='square,pad=0.15', fc=col, lw=0))
    full = mean('full_h')
    ax.text(0.98, 0.97, f'Full validation: ${full.mean():.1f}$ hours', transform=ax.transAxes, ha='right', va='top',
            fontsize=7.5, color=INK, bbox=dict(boxstyle='square,pad=0.3', fc='#c3c1b9', ec='none', alpha=0.6))
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.5, 1))
    ax.set_xticks(x)
    ax.set_xticklabels([f'${W:g}$' for W in Ws])
    ax.set_xlabel(r'Target width $W_{\mathrm{th}}$')
    ax.set_ylabel('Emulation time per predictor (hour)')
    ax.yaxis.set_major_formatter(MathScalarFormatter())
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=7, handlelength=1.2,
              columnspacing=0.9, handletextpad=0.4)
    save(fig, out, 'fig_savings')


def fig_checkpoints(results, histories, out, run='gb_d3_checkpoints', W=0.2, delta=0.05):
    ch4 = pd.read_csv(os.path.join(results, run, 'ch4.csv'))
    h = np.load(os.path.join(histories, f'history_{run}.npz'))
    full = dict(zip(h['cand_names'].astype(str), h['cost_C'].sum(1) / 3600))
    g = ch4[np.isclose(ch4.W_th, W)].groupby('candidate', sort=False)
    names = sorted(g.groups, key=lambda c: int(c.rsplit('@', 1)[1]))
    s = g.agg(K=('K_star', 'first'), L=('L', 'mean'), U=('U', 'mean'), cost=('cost', 'mean')).loc[names]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw=dict(wspace=0.32))
    x = np.arange(len(names))
    ax.vlines(x, s.L, s.U, color=C_CH4, lw=5, alpha=0.5, label=fr'aPPE interval at $W_{{\mathrm{{th}}}} = {W:g}$ (mean)')
    ax.plot(x, s.K, 'o', color=INK, ms=4, label=r'$K^*$ (full validation)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'${c.rsplit("@", 1)[1]}$' for c in names])
    ax.set_xlabel('Boosting iterations of the checkpoint')
    ax.set_ylabel(r'Success rate $K^*$')
    ax.yaxis.set_major_formatter(MathScalarFormatter())
    ax.legend(loc='best', frameon=False, fontsize=7)
    ax.set_title('(a)', loc='left', color=INK)
    bx.bar(x - 0.18, [full[c] for c in names], 0.34, color='#c3c1b9', label='Full validation', zorder=2)
    bx.bar(x + 0.18, s.cost / 3600, 0.34, color=C_CH4, label=r'aPPE, $S=5$', zorder=2)
    bx.set_xticks(x)
    bx.set_xticklabels([f'${c.rsplit("@", 1)[1]}$' for c in names])
    bx.set_xlabel('Boosting iterations of the checkpoint')
    bx.set_ylabel('Emulation time (hour)')
    bx.yaxis.set_major_formatter(MathScalarFormatter())
    bx.legend(loc='best', frameon=False, fontsize=7)
    bx.set_title('(b)', loc='left', color=INK)
    save(fig, out, 'fig_checkpoints')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results_nostra')
    ap.add_argument('--histories', default='outputs_nostra')
    ap.add_argument('--out', default='figures_nostra')
    ap.add_argument('--candidates', nargs='*', default=None, help='candidates averaged in fig_savings (default: all)')
    ap.add_argument('--widths', type=float, nargs='*', default=None, help='target widths shown in fig_savings')
    ap.add_argument('--tbar', type=int, default=400, help='fixed budget T_bar of PPE (rounds)')
    ap.add_argument('--checkpoints', action='store_true', help='also draw fig_checkpoints')
    ap.add_argument('--checkpoint_width', type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()
    df = cost_table(args.results, args.histories, args.tbar)
    df.to_csv(os.path.join(args.out, 'savings.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(df.round(3).to_string(index=False))
    fig_savings(df, args.out, args.candidates, args.widths)
    if args.checkpoints and os.path.exists(os.path.join(args.results, 'gb_d3_checkpoints', 'ch4.csv')):
        fig_checkpoints(args.results, args.histories, args.out, W=args.checkpoint_width)
    print(f'saved savings.csv and figures to {args.out}/')


if __name__ == '__main__':
    main()
