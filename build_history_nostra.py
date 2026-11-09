import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def load_campaign(camp, models_csv, min_ue=None):
    runs = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(os.path.join(camp, 'runs*.jsonl')))
                      if os.path.getsize(f)], ignore_index=True)
    runs = runs[runs.valid.astype(bool)].drop_duplicates(['config', 'arm', 'rep'], keep='last')
    cfg = pd.read_csv(os.path.join(camp, 'configs.csv')).set_index('config')
    models = pd.read_csv(models_csv)
    olla = runs[runs.arm == 'olla'].groupby('config').dl_mbps.agg(['mean', 'std', 'count'])
    pred = runs[runs.arm != 'olla']
    G = pred.pivot(index='arm', columns='config', values='dl_mbps')
    cost = pred.pivot(index='arm', columns='config', values='wall_s')
    names = [m for m in models.name if m in G.index]
    configs = [c for c in G.columns if c in olla.index and olla.loc[c, 'count'] >= 2
               and G.loc[names, c].notna().all() and (min_ue is None or cfg.loc[c, 'n_ue'] >= min_ue)]
    G, cost = G.loc[names, configs], cost.loc[names, configs]
    gain = G / olla.loc[configs, 'mean'].values - 1
    return dict(gain=gain, cost=cost, olla=olla.loc[configs], cfg=cfg.loc[configs], models=models.set_index('name'),
                olla_cost=runs[runs.arm == 'olla'].groupby('config').wall_s.mean().loc[configs])


def write_history(path, d, K, cand_runs, snapshots=False):
    models = d['models']
    names = np.array(K.index, dtype=str)
    run = models.loc[names, 'run'].values
    final = ~models.loc[names, 'is_snapshot'].values.astype(bool)
    held = np.isin(run, cand_runs)
    hist, cand = ~held, held & (final | snapshots)
    cost, gain = d['cost'].values.astype(float), d['gain'].values.astype(float)
    np.savez_compressed(path, H=K.values[hist].astype(np.uint8), hist_names=names[hist],
                        C=K.values[cand].astype(np.uint8), cand_names=names[cand],
                        cost_H=cost[hist], cost_C=cost[cand], gain_H=gain[hist], gain_C=gain[cand],
                        configs=np.array(K.columns, dtype=str), olla_cost=d['olla_cost'].values.astype(float))
    return names[hist], names[cand]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--campaign', default='campaign')
    ap.add_argument('--models', default='models/models.csv')
    ap.add_argument('--out', default='outputs_nostra')
    ap.add_argument('--threshold', type=float, default=0.01, help='success: gain over OLLA >= threshold')
    ap.add_argument('--min_ue', type=int, default=4,
                    help='configurations with fewer UEs are left out (run-to-run variability of OLLA above the margin)')
    ap.add_argument('--candidates', nargs='+', default=['collab'], help='training runs held out as candidates')
    ap.add_argument('--leave_one_out', action='store_true', help='one history file per candidate run')
    ap.add_argument('--snapshot_candidates', action='store_true', help='snapshots of held-out runs are candidates')
    ap.add_argument('--suffix', default='', help='appended to the history file names')
    args = ap.parse_args()

    d = load_campaign(args.campaign, args.models, args.min_ue)
    gain, cost = d['gain'], d['cost']
    K = (gain >= args.threshold).astype(np.uint8)
    os.makedirs(args.out, exist_ok=True)
    N = K.shape[1]
    print(f'N = {N} complete configurations, {K.shape[0]} predictors, success = gain >= {args.threshold:.1%}')
    print(f"OLLA: {d['olla']['count'].mean():.1f} runs per configuration, run-to-run std of the cell goodput "
          f"{np.nanmean(d['olla']['std'] / d['olla']['mean']):.2%} (mean over configurations)")
    summary = pd.DataFrame(dict(K_star=K.mean(1), mean_gain=gain.mean(1), median_gain=gain.median(1),
                                full_validation_h=cost.sum(1) / 3600))
    print(summary.round(4).to_string())
    summary.to_csv(os.path.join(args.out, 'predictors.csv'))
    pd.concat([d['cfg'], d['olla'].add_prefix('olla_'), gain.T.add_prefix('gain_')], axis=1).to_csv(
        os.path.join(args.out, 'configs_gain.csv'))
    if args.leave_one_out:
        for r in args.candidates:
            name = f'history_{r}{args.suffix}.npz'
            h, c = write_history(os.path.join(args.out, name), d, K, [r], args.snapshot_candidates)
            print(f'{name}: H {len(h)} models, candidates {list(c)}')
    else:
        name = f'history{args.suffix}.npz'
        h, c = write_history(os.path.join(args.out, name), d, K, args.candidates, args.snapshot_candidates)
        print(f'{name}: H {len(h)} models, candidates {list(c)}')
    with open(os.path.join(args.out, 'history_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)


if __name__ == '__main__':
    main()
