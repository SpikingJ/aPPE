import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator, ScalarFormatter

C_CH4, C_CH3, C_RHO1 = '#2a78d6', '#eb6834', '#1baf7a'
C_UNIF, CRITICAL = '#898781', '#d03b3b'
INK, GRID = 'black', '#e1e0d9'

CH4_RUNS = {
    'ch4': 'aPPE, S = 5',
    'ch4_S3': 'aPPE, S = 3 (rho in {0, 0.5, 1})',
    'ch4_rho1': 'aPPE, S = 1 (rho = 1)',
    'ch4_rho0': 'aPPE, rho = 0 only',
    'ch4_etamix': 'aPPE, eta in {0.2, 0.5, 0.8}',
    'ch4_etamix9': 'aPPE, eta in {0.25, 0.5, 0.9}',
    'ch4_eta0.3': 'aPPE, eta = 0.3',
    'ch4_eta0.2': 'aPPE, eta = 0.2',
    'ch4_eta0.5': 'aPPE, eta = 0.5',
    'ch4_uniform_eta0.9': 'Uniform anytime-valid, eta = 0.9',
    'ch4_uniform_eta0.7': 'Uniform anytime-valid, eta = 0.7',
    'ch4_uniform_eta0.5': 'Uniform anytime-valid, eta = 0.5',
    'ch4_uniform_eta0.3': 'Uniform anytime-valid, eta = 0.3',
    'ch4_uniform_eta0.2': 'Uniform anytime-valid, eta = 0.2',
    'ch4_uniform_eta0.1': 'Uniform anytime-valid, eta = 0.1',
    'ch4_uniform_eta0.05': 'Uniform anytime-valid, eta = 0.05',
}


def style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'], 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.edgecolor': INK, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-', 'axes.axisbelow': True,
        'lines.linewidth': 1.5, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
        'lines.markersize': 5, 'lines.markeredgewidth': 1.0, 'lines.markeredgecolor': 'white',
        'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })


class MathScalarFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)
        return f'${s}$' if s else s


math_int = FuncFormatter(lambda v, _: f'${v:,.0f}$'.replace(',', '{,}'))
math_g = FuncFormatter(lambda v, _: f'${v:g}$')


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'))
    plt.close(fig)


def load(results):
    read = lambda stem: pd.read_csv(os.path.join(results, f'{stem}_summary.csv'))
    exists = lambda stem: os.path.exists(os.path.join(results, f'{stem}_summary.csv'))
    with open(os.path.join(results, 'ch4_config.json')) as f:
        cfg = json.load(f)
    return dict(ch3=read('ch3'), ch3_uniform=read('ch3_uniform') if exists('ch3_uniform') else None,
                ch4={stem: read(stem) for stem in CH4_RUNS if exists(stem)}, N=cfg['N'], delta=cfg['delta'])


def comparison_table(r):
    rows = []
    for key, name in (('ch3', 'PPE'), ('ch3_uniform', 'PPE, uniform, no prediction')):
        if r[key] is not None:
            fixed = r[key][r[key]['mode'] == 'fixed']
            rows += [dict(row, method=f'{name}, fixed T_bar = {row["T_max"]}', W_th=np.nan)
                     for row in fixed.to_dict('records')]
    for stem, s in r['ch4'].items():
        rows += [dict(row, method=CH4_RUNS[stem]) for row in s.to_dict('records')]
    df = pd.DataFrame(rows)
    df['share_of_N'] = df.n_unique / r['N']
    cols = ['method', 'W_th', 'coverage', 'min_coverage', 'width', 'T', 'n_unique', 'share_of_N', 'stopped']
    return df.reindex(columns=cols).sort_values(['W_th', 'n_unique'], ascending=[False, True], na_position='first')


def log_width_axis(ax, ticks):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{w:g}' for w in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def label(ax, x, y, text, **kw):
    ax.text(x, y, text, color=INK, fontsize=6.5, bbox=dict(facecolor='white', edgecolor='none', pad=0.6, alpha=0.85),
            **kw)


def fig_coverage(r, out):
    ch3, ch4, delta = r['ch3'], r['ch4']['ch4'].sort_values('width'), r['delta']
    fixed = ch3[ch3['mode'] == 'fixed'].sort_values('width')
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
    titles = ['(a) Average over candidate models', '(b) Worst candidate model']
    for ax, col, title in zip(axes, ['coverage', 'min_coverage'], titles):
        ax.plot(fixed.width, fixed[col], color=C_CH3, marker='s', label=r'PPE, fixed $\bar T$')
        for _, row in fixed.iterrows():
            ax.text(row.width, row[col] - 0.006, f'{row.T_max:.0f}', color=INK, fontsize=6, ha='center', va='top')
        ax.plot(ch4.width, ch4[col], color=C_CH4, marker='o', label=r'aPPE, $S=5$, stopped at $W_{\mathrm{th}}$')
        ax.axhline(1 - delta, color=INK, lw=0.8, ls=(0, (4, 3)))
        ax.text(0.115, 1 - delta + 0.002, r'$1-\delta$', color=INK, fontsize=7, ha='right', va='bottom')
        log_width_axis(ax, [0.02, 0.05, 0.1])
        ax.set_xlim(0.014, 0.12)
        ax.set_xlabel('Interval width')
        ax.set_title(title, loc='left', color=INK)
    lo = np.floor((min(fixed.min_coverage.min(), ch4.min_coverage.min()) - 0.01) * 20) / 20
    axes[0].set_ylim(lo, 1.005)
    axes[0].set_yticks(np.arange(lo, 1.0001, 0.05))
    axes[0].set_ylabel('Coverage')
    axes[0].legend(loc='lower right')
    save(fig, out, 'fig_coverage')


def open_if_not_stopped(ax, s, color, marker, **kw):
    m = s[s.stopped < 1]
    if len(m):
        ax.plot(m.width, m.n_unique, ls='none', marker=marker, mfc='white', mec=color, zorder=6, **kw)


def fig_cost(r, out):
    ch3, ch4, N = r['ch3'], r['ch4'], r['N']
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    s = ch4['ch4'].sort_values('width')
    ax.plot(s.width, s.n_unique, color=C_CH4, marker='o', label=r'aPPE, $S=5$', zorder=5)
    open_if_not_stopped(ax, s, C_CH4, 'o')
    if 'ch4_rho1' in ch4:
        s = ch4['ch4_rho1'].sort_values('width')
        ax.plot(s.width, s.n_unique, color=C_RHO1, marker='D', markersize=4.5, label=r'aPPE, $S=1$ ($\rho_1=1$)')
        open_if_not_stopped(ax, s, C_RHO1, 'D', markersize=4.5)
    uniform = [s for stem, s in ch4.items() if stem.startswith('ch4_uniform')]
    if uniform:
        u = pd.concat(uniform)
        u = pd.concat([g[g.stopped == 1].nsmallest(1, 'n_unique') if (g.stopped == 1).any()
                       else g.sort_values(['stopped', 'n_unique'], ascending=[False, True]).head(1)
                       for _, g in u.groupby('W_th')]).sort_values('width')
        ax.plot(u.width, u.n_unique, color=C_UNIF, marker='v', label='Uniform, anytime-valid')
        open_if_not_stopped(ax, u, C_UNIF, 'v')
    fixed = ch3[ch3['mode'] == 'fixed'].sort_values('width')
    ax.plot(fixed.width, fixed.n_unique, color=C_CH3, marker='s', label=r'PPE, fixed $\bar T$')
    ax.axhline(N, color=INK, lw=0.8)
    ax.text(0.115, N * 1.12, f'Full benchmark (N = {N:,})', color=INK, fontsize=7, ha='right', va='bottom')
    log_width_axis(ax, [0.02, 0.05, 0.1])
    ax.set_xlim(0.014, 0.12)
    ax.set_yscale('log')
    ax.set_ylim(80, N * 2.2)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xlabel('Interval width')
    ax.set_ylabel('Evaluated test samples')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.2)
    save(fig, out, 'fig_cost')


ARCH_NAMES = {'mlp': 'MLP', 'cnn': 'CNN', 'vgg': 'VGG', 'resnet': 'ResNet', 'tcn': 'TCN', 'inception': 'Inception',
              'cldnn': 'CLDNN', 'lstm': 'LSTM', 'gru': 'GRU', 'transformer': 'Transformer'}


def load_paths(results):
    stem = lambda ch: next(s for s in (f'{ch}_paths', f'{ch}_traj')
                           if os.path.exists(os.path.join(results, f'{s}_traj.npz')))
    s3, s4 = stem('ch3'), stem('ch4')
    t3 = np.load(os.path.join(results, f'{s3}_traj.npz'))
    return ({k.split('|')[0]: t3[k] for k in t3.files}, np.load(os.path.join(results, f'{s4}_traj.npz')),
            pd.read_csv(os.path.join(results, f'{s4}.csv')))


APPE_PATHS = [
    ('ch4_paths', r'aPPE, $S=5$', C_CH4, '-'),
    ('ch4_paths_S1', r'aPPE, $S=1$, $\rho_1 = 1$', C_RHO1, '-'),
]
PPE_DASH = (0, (4, 2))
S1_STOP_DASH = (0, (2, 1.2))


def load_run(results, stem):
    z = np.load(os.path.join(results, f'{stem}_traj.npz'))
    return {k.split('|')[0]: z[k] for k in z.files}, pd.read_csv(os.path.join(results, f'{stem}.csv'))


def unclip(L, U, K):
    inside = lambda v: (v > 0) & (v < 1)
    h = np.where(inside(L), K - L, np.where(inside(U), U - K, (U - L) / 2))
    return K - h, K + h


def ffill(a):
    a = a.copy()
    for r in range(len(a)):
        rec = np.flatnonzero(~np.isnan(a[r]))
        if len(rec) and rec[-1] < a.shape[1] - 1:
            a[r, rec[-1] + 1:] = a[r, rec[-1]]
    return a


def wilson(share, n, z=1.96):
    centre = (share + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(share * (1 - share) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def fig_paths(results, out, cand, run=0, widths=(0.1, 0.05, 0.02), delta=0.05, t_min=10, t_max=4000, c_min=0.01,
              y_max=None, b_min=0.2, uniform_a=False, b_legend=None):
    p3, _ = load_run(results, 'ch3_paths')
    pu = load_run(results, 'ch3_uniform_paths')[0] if os.path.exists(
        os.path.join(results, 'ch3_uniform_paths_traj.npz')) else None
    appe = [(lab, col, ls) + load_run(results, stem) for stem, lab, col, ls in APPE_PATHS
            if os.path.exists(os.path.join(results, f'{stem}_traj.npz'))]
    K = appe[0][4].groupby('candidate').K_star.first()[cand]
    L3, U3 = unclip(*p3[cand][:3])
    n_runs, T_bar = L3.shape
    stops = [s[(s.candidate == cand) & (s.trial == run)].set_index('W_th') for *_, s in appe]

    fig = plt.figure(figsize=(7.16, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.42, wspace=0.3)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    t3 = np.arange(t_min, T_bar + 1)
    key = []

    ax = ax_a
    l3, u3 = L3[run, t_min - 1:], U3[run, t_min - 1:]
    ax.fill_between(t3, 0, 0.035, where=(l3 > K) | (u3 < K), transform=ax.get_xaxis_transform(), color=CRITICAL,
                    alpha=0.85, lw=0)
    ax.axhspan(l3[-1], u3[-1], color=C_CH3, alpha=0.35, lw=0)
    ax.plot(t3, l3, color=C_CH3, lw=1.0, ls=PPE_DASH)
    ax.plot(t3, u3, color=C_CH3, lw=1.0, ls=PPE_DASH)
    if pu is not None and uniform_a:
        for v in unclip(*pu[cand][:3]):
            ax.plot(np.arange(t_min, v.shape[1] + 1), v[run, t_min - 1:], color=C_UNIF, lw=1.0, ls=PPE_DASH,
                    zorder=1.9)
    for (lab, col, ls, paths, _), st, main in zip(appe, stops, [True] + [False] * len(appe)):
        if not (main or lab.startswith(r'aPPE, $S=1$')):
            continue
        T_end = int(st['T'].max())
        t4 = np.arange(t_min, T_end + 1)
        l4, u4 = paths[cand][0, run, t_min - 1:T_end], paths[cand][1, run, t_min - 1:T_end]
        if main:
            ax.fill_between(t4, l4, u4, color=col, alpha=0.12, lw=0)
        for y in (l4, u4):
            ax.plot(t4, y, color=col, lw=1.3 if main else 1.1, ls=ls)
    for W in widths:
        row = stops[0].loc[W]
        ax.plot([row['T'], row['T']], [row.L, row.U], color=INK, lw=2.0, solid_capstyle='butt')
    for (lab, col, *_), st in zip(appe[1:], stops[1:]):
        if lab.startswith(r'aPPE, $S=1$'):
            for W in widths:
                row = st.loc[W]
                ax.plot([row['T'], row['T']], [row.L, row.U], color=col, lw=1.6, ls=S1_STOP_DASH)
    ax.axhline(K, color=INK, lw=0.9, ls=(0, (6, 3)))
    ax.text(1.006, K, r'$K^*$', color=INK, fontsize=9, ha='left', va='center', transform=ax.get_yaxis_transform(),
            clip_on=False)
    lo, hi = max(0, K - 0.25), min(1, K + 0.25)
    top = hi + 0.05 * (hi == 1) if y_max is None else y_max
    ax.set_ylim(lo - 0.05 * (lo == 0), top)
    if lo == 0 or hi == 1:
        ax.set_yticks([v for v in ax.get_yticks() if lo - 1e-9 <= v <= hi + 1e-9])
    ax.yaxis.set_major_formatter(MathScalarFormatter())
    ax.set_ylabel(r'Confidence interval $\mathcal{C}_t$')
    ax.set_title('(a)', loc='left', color=INK)

    ok3 = (L3 <= K) & (K <= U3)
    series = []
    for lab, col, ls, paths, _ in appe:
        L4, U4 = (ffill(paths[cand][i][:, :max(T_bar, t_max)]) for i in (0, 1))
        ok4 = (L4 <= K) & (K <= U4)
        series.append((lab, col, ls, ok4[:, t_min - 1:].mean(0), U4 - L4, 2))
    series += [('PPE: fixed-time coverage (b), intermediate intervals (a), (c)', C_CH3, PPE_DASH,
                ok3[:, t_min - 1:].mean(0), U3 - L3, 2),
               ('PPE: time-uniform coverage (b)', C_CH3, '-', np.cumprod(ok3[:, t_min - 1:], axis=1).mean(0), None,
                2)]
    if pu is not None:
        Lu, Uu = unclip(*pu[cand][:3])
        oku = (Lu <= K) & (K <= Uu)
        series += [(f'Uniform sampling: fixed-time coverage (b), {"intervals (a), " if uniform_a else ""}(c)', C_UNIF,
                    PPE_DASH, oku[:, t_min - 1:].mean(0), Uu - Lu, 1.9),
                   ('Uniform sampling: time-uniform coverage (b)', C_UNIF, '-',
                    np.cumprod(oku[:, t_min - 1:], axis=1).mean(0), None, 1.9)]
    for lab, col, ls, cover, width, z in series:
        lo, hi = wilson(cover, n_runs)
        t = np.arange(t_min, t_min + len(cover))
        ax_b.fill_between(t, lo, hi, color=col, alpha=0.13, lw=0)
        ax_b.plot(t, cover, color=col, lw=1.3, ls=ls, zorder=z)
        if width is not None:
            w = width[:, t_min - 1:]
            w = np.where(w > 0, w, np.nan)
            ax_c.fill_between(t[:w.shape[1]], np.nanpercentile(w, 10, 0), np.nanpercentile(w, 90, 0), color=col,
                              alpha=0.13, lw=0)
            ax_c.plot(t[:w.shape[1]], np.nanmean(w, 0), color=col, lw=1.3, ls=ls, zorder=z)
        key.append((dict(color=col, lw=1.3, ls=ls), lab))
    ax_b.axhline(1 - delta, color=INK, lw=0.8, ls=(0, (1, 2)), zorder=1.8)
    ax_b.set_ylim(b_min - 0.03 * (b_min == 0), 1.03)
    ax_b.set_yticks(np.round(np.arange(b_min, 1.001, 0.2), 1))
    ax_b.yaxis.set_major_formatter(MathScalarFormatter())
    ax_b.set_ylabel('Empirical coverage rate')
    ax_b.set_title('(b)', loc='left', color=INK)
    if b_legend:
        names = [(C_CH3, 'PPE')] + [(C_UNIF, 'Uniform sampling')] * (pu is not None)
        ax_b.legend([Line2D([], [], color=c, lw=1.3) for c, _ in names], [n for _, n in names], loc=b_legend)
    for W in widths:
        ax_c.axhline(W, color=INK, lw=0.7, ls=(0, (1, 2)), zorder=1.8)
    ax_c.set_yscale('log')
    ax_c.set_ylim(bottom=c_min)
    ax_c.yaxis.set_major_formatter(math_g)
    ax_c.set_ylabel(r'Interval length $U_t - L_t$')
    ax_c.set_title('(c)', loc='left', color=INK)

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xscale('log')
        ax.set_xlim(t_min, t_max)
        ax.set_xticks([v for v in (10, 100, 1000, 10000) if v < t_max / 1.6] + [t_max])
        ax.get_xgridlines()[-1].set_visible(False)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(math_int)
        ax.set_xlabel('Evaluation round $t$')
    save(fig, out, 'fig_paths')

    lines =[(Line2D([], [], **kw), lab) for kw, lab in key]
    marks = [(Line2D([], [], color=INK, lw=2.0, solid_capstyle='butt'),
              r'(a) aPPE ($S=5$) stops, $U_T - L_T \leq W_{\mathrm{th}}$'),
             (Line2D([], [], color=C_RHO1, lw=1.6, ls=S1_STOP_DASH),
              r'(a) aPPE ($S=1$) stops, $U_T - L_T \leq W_{\mathrm{th}}$'),
             (Patch(color=C_CH3, alpha=0.35, lw=0), r'(a) PPE confidence interval $\mathcal{C}_{\bar T}$'),
             (Patch(color=CRITICAL, alpha=0.85, lw=0), r'(a) $K^*$ outside the PPE intermediate interval')]
    rows = max(len(lines), len(marks))
    blank = (Line2D([], [], alpha=0), '')
    entries = lines + [blank] * (rows - len(lines)) + marks + [blank] * (rows - len(marks))
    fig = plt.figure(figsize=(7.16, 0.2 + 0.18 * rows))
    fig.legend(*zip(*entries), loc='center', ncol=2, columnspacing=1.5, handlelength=2.6)
    save(fig, out, 'fig_paths_legend')


def fig_traj(results, out, run=0, min_rounds=10):
    _, t4, s4 = load_paths(results)
    K_star = s4.groupby('candidate').K_star.first()
    cands = sorted(t4.files, key=lambda c: K_star[c])
    fig, axes = plt.subplots(1, len(cands), figsize=(7.16, 2.5), gridspec_kw=dict(wspace=0.3))
    axes = np.atleast_1d(axes)
    for i, (ax, cand) in enumerate(zip(axes, cands)):
        K = K_star[cand]
        stops = s4[(s4.candidate == cand) & (s4.trial == run)]
        T_end = int(stops['T'].max())
        t = np.arange(min_rounds, T_end + 1)
        L, U = t4[cand][0, run, t - 1], t4[cand][1, run, t - 1]
        ax.fill_between(t, L, U, color=C_CH4, alpha=0.10, lw=0)
        ax.plot(t, L, color=C_CH4, lw=1.2, label=r'Confidence sequence $\mathcal{C}_t$')
        ax.plot(t, U, color=C_CH4, lw=1.2)
        for j, (_, row) in enumerate(stops.sort_values('T').iterrows()):
            T = int(row['T'])
            ax.plot([T, T], [row.L, row.U], color=INK, lw=1.8, solid_capstyle='butt',
                    label=r'$\mathcal{C}_T$ when the width reaches $W_{\mathrm{th}}$' if j == 0 else None)
            ax.text(T, row.L - 0.012, f'{row.W_th:g}', color=INK, fontsize=6.5, ha='center', va='top')
        ax.axhline(K, color=INK, lw=0.9)
        ax.text(min_rounds * 1.1, K + 0.006, r'$K^*$', color=INK, fontsize=7, ha='left', va='bottom')
        ax.set_xscale('log')
        ax.set_xlim(min_rounds, T_end * 1.5)
        ax.set_ylim(max(0, K - 0.2), min(1, K + 0.2))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.set_xlabel('Evaluation round $t$')
        arch = cand.split('_')[1]
        ax.set_title(f'({"abcdef"[i]}) {ARCH_NAMES.get(arch, arch)}, $K^*$ = {K:.3f}', loc='left', color=INK)
    axes[0].set_ylabel('Accuracy')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2)
    save(fig, out, 'fig_traj')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--extra_figures', action='store_true', help='also draw fig_coverage, fig_cost and fig_traj')
    ap.add_argument('--traj_run', type=int, default=0, help='run shown in fig_traj')
    ap.add_argument('--paths_candidate', default=None, help='candidate of fig_paths (default: highest K* saved)')
    ap.add_argument('--paths_miss_after', type=int, default=10,
                    help='default run of fig_paths (a): the first one in which the PPE interval misses K* at some '
                         'round t >= this value while aPPE covers K* at every t >= 10')
    ap.add_argument('--paths_final_miss', action='store_true',
                    help='default run of fig_paths (a): among the runs in which the PPE interval at T_bar misses K* '
                         'while aPPE covers K* at every t >= 10, the one with the median number of rounds in which '
                         'the PPE interval misses K*')
    ap.add_argument('--paths_run', type=int, default=None,
                    help='run shown in fig_paths (a) (default: first run where PPE misses K* and aPPE does not)')
    ap.add_argument('--paths_tmax', type=int, default=4000, help='last round t shown in fig_paths')
    ap.add_argument('--paths_cmin', type=float, default=0.01, help='lower limit of the set size axis of fig_paths (c)')
    ap.add_argument('--paths_ymax', type=float, default=None, help='upper limit of the interval axis of fig_paths (a)')
    ap.add_argument('--paths_bmin', type=float, default=0.2,
                    help='lower limit of the coverage axis of fig_paths (b) (0 shows the whole time-uniform coverage)')
    ap.add_argument('--paths_widths', type=float, nargs='+', default=[0.1, 0.05, 0.02],
                    help='target widths W_th whose aPPE stopping times are marked in fig_paths (a)')
    ap.add_argument('--paths_uniform_a', action='store_true',
                    help='also draw the intervals of uniform sampling (same run index) in fig_paths (a)')
    ap.add_argument('--paths_b_legend', nargs='?', const='lower right', default=None, metavar='LOC',
                    help="key of the colors of PPE and uniform sampling in fig_paths (b), at the legend location LOC "
                         "(default 'lower right')")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    style()
    r = load(args.results)
    table = comparison_table(r)
    table.to_csv(os.path.join(args.out, 'comparison.csv'), index=False)
    print(f'N = {r["N"]:,}, nominal coverage {1 - r["delta"]:.2f}')
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        fixed = table.W_th.isna()
        print('\n=== PPE, Sec. III (number of rounds T_bar fixed in advance)')
        print(table[fixed].drop(columns=['W_th', 'stopped']).to_string(index=False, float_format='%.4f'))
        for W, g in table[~fixed].groupby('W_th', sort=False):
            print(f'\n=== aPPE (Sec. IV) and baselines, W_th = {W:g}')
            print(g.drop(columns='W_th').to_string(index=False, float_format='%.4f'))
    has = lambda ch: any(os.path.exists(os.path.join(args.results, f'{ch}_{s}_traj.npz')) for s in ('paths', 'traj'))
    if args.extra_figures:
        fig_coverage(r, args.out)
        fig_cost(r, args.out)
        if has('ch4'):
            fig_traj(args.results, args.out, args.traj_run)
    if has('ch3') and has('ch4'):
        p3, p4, s4 = load_paths(args.results)
        K_star = s4.groupby('candidate').K_star.first()
        cand = args.paths_candidate or max((c for c in p3 if c in K_star), key=lambda c: K_star[c])
        run = args.paths_run
        if run is None:
            K, T_bar, t0, t3 = K_star[cand], p3[cand].shape[2], 9, args.paths_miss_after - 1
            if args.paths_final_miss:
                t3 = T_bar - 1
            miss3 = ((p3[cand][0, :, t3:] > K) | (p3[cand][1, :, t3:] < K)).any(1)
            miss4 = ((p4[cand][0, :, t0:T_bar] > K) | (p4[cand][1, :, t0:T_bar] < K)).any(1)
            runs = np.flatnonzero(miss3 & ~miss4[:len(miss3)])
            run = int(runs[0])
            if args.paths_final_miss:
                L3, U3 = unclip(*p3[cand][:3])
                missed = ((L3[runs, t0:] > K) | (U3[runs, t0:] < K)).sum(1)
                run = int(runs[np.argsort(missed, kind='stable')[len(runs) // 2]])
        fig_paths(args.results, args.out, cand, run, tuple(args.paths_widths), r['delta'], t_max=args.paths_tmax,
                  c_min=args.paths_cmin, y_max=args.paths_ymax, b_min=args.paths_bmin, uniform_a=args.paths_uniform_a,
                  b_legend=args.paths_b_legend)
        print(f'fig_paths: candidate {cand}, run {run}')
    print(f'\nsaved figures and comparison.csv to {args.out}/')


if __name__ == '__main__':
    main()
