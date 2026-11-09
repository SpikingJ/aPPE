import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from analyze_results import C_CH4, C_RHO1, INK, math_int, save, style


def panel(ax, results, cand, title, letter, percent=False):
    n_full = json.load(open(os.path.join(results, 'ch4_config.json')))['N']
    runs = {lab: pd.read_csv(os.path.join(results, f)) for lab, f in
            ((r'aPPE, $S=1$, $\rho_1 = 1$', 'ch4_rho1.csv'), (r'aPPE, $S=5$', 'ch4.csv'))}
    Ws = np.array(sorted(runs[r'aPPE, $S=5$'].W_th.unique()))
    x = np.arange(len(Ws))
    w = 0.34
    top = 0
    for i, ((lab, d), col) in enumerate(zip(runs.items(), (C_RHO1, C_CH4))):
        g = (d if cand == 'all' else d[d.candidate == cand]).groupby('W_th')
        n = g.n_unique.mean().reindex(Ws).values
        if percent:
            n = 100 * n / n_full
            print(f'{title} | {lab} | ' + ', '.join(f'W_th={W:g}: {v:.2f}%' for W, v in zip(Ws, n)))
        ax.bar(x + (i - 0.5) * w, n, w * 0.92, color=col, label=lab, zorder=2)
        top = max(top, n.max())
    ax.text(0.97, 0.96, f'Full validation: $N = {n_full:,}$'.replace(',', '{,}'), transform=ax.transAxes,
            ha='right', va='top', fontsize=7, color=INK,
            bbox=dict(boxstyle='square,pad=0.3', fc='#c3c1b9', ec='none', alpha=0.6))
    ax.set_ylim(0, top * 1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'${W:g}$' for W in Ws])
    ax.set_xlabel(r'Target width $W_{\mathrm{th}}$')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:.0f}$') if percent else math_int)
    ax.set_title(f'({letter}) {title}', loc='left', color=INK)
    return top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--panels', nargs='+', required=True, help='results directory:candidate:title, one per panel')
    ap.add_argument('--out', default='figures_sec5')
    ap.add_argument('--legend_panel', type=int, default=1, help='panel (from 0) whose blank area holds the legend')
    ap.add_argument('--percent', action='store_true',
                    help='evaluated samples as a percentage of N, on a common y-axis (saved as fig_samples_pct)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    style()
    fig, axes = plt.subplots(1, len(args.panels), figsize=(7.16, 2.3), gridspec_kw=dict(wspace=0.32))
    tops = []
    for ax, spec, letter in zip(np.atleast_1d(axes), args.panels, 'abcdef'):
        results, cand, title = spec.split(':', 2)
        tops.append(panel(ax, results, cand, title, letter, args.percent))
    if args.percent:
        for ax in np.atleast_1d(axes):
            ax.set_ylim(0, max(tops) * 1.3)
            ax.yaxis.set_major_locator(MultipleLocator(10))
    axes[0].set_ylabel('Percentage of evaluated samples' if args.percent else 'Evaluated samples')
    axes[args.legend_panel].legend(loc='upper right', bbox_to_anchor=(1.0, 0.84), ncol=1, frameon=False, fontsize=7,
                                   handlelength=1.2, handletextpad=0.5)
    name = 'fig_samples_pct' if args.percent else 'fig_samples'
    save(fig, args.out, name)
    print(f'saved {name} to {args.out}/')


if __name__ == '__main__':
    main()
