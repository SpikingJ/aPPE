import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def load_zoo(zoo_dir, key='correct'):
    metas = []
    for path in sorted(glob.glob(os.path.join(zoo_dir, '*.json'))):
        with open(path) as f:
            metas.append(json.load(f))
    finished = {m['run'] for m in metas if not m['is_snapshot']}
    metas = [m for m in metas if m['run'] in finished]
    signatures = {m['test_signature'] for m in metas}
    if len(signatures) != 1:
        raise ValueError(f'models were evaluated on different test sets: {signatures}')
    K = np.stack([np.load(os.path.join(zoo_dir, f"{m['name']}.npz"))[key] for m in metas]).astype(np.uint8)
    return pd.DataFrame(metas), K


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zoo', default='outputs/zoo')
    ap.add_argument('--out', default='outputs/history.npz')
    ap.add_argument('--key', default='correct', help='success vector in the model files (success criterion)')
    ap.add_argument('--holdout', default='random', choices=['random', 'arch', 'names'],
                    help='random runs / all runs of some architectures / explicit run names')
    ap.add_argument('--n_candidates', type=int, default=12, help='for --holdout random')
    ap.add_argument('--archs', nargs='*', default=[], help='for --holdout arch')
    ap.add_argument('--names', nargs='*', default=[], help='for --holdout names (run names)')
    ap.add_argument('--no_snapshots', action='store_true', help='keep only final models in H')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    df, K = load_zoo(args.zoo, args.key)
    runs = df.loc[~df.is_snapshot, ['run', 'arch']].drop_duplicates()
    if args.holdout == 'random':
        rng = np.random.default_rng(args.seed)
        held = set(rng.choice(runs.run.values, size=args.n_candidates, replace=False))
    elif args.holdout == 'arch':
        held = set(runs.run[runs.arch.isin(args.archs)])
    else:
        held = set(args.names)
    unknown = held - set(runs.run)
    if unknown or not held:
        raise ValueError(f'no candidates selected / unknown runs: {unknown}')

    df['role'] = 'history'
    df.loc[df.run.isin(held), 'role'] = 'excluded'
    df.loc[df.run.isin(held) & ~df.is_snapshot, 'role'] = 'candidate'
    if args.no_snapshots:
        df.loc[(df.role == 'history') & df.is_snapshot, 'role'] = 'excluded'
    hist, cand = np.flatnonzero(df.role == 'history'), np.flatnonzero(df.role == 'candidate')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    names = df['name'].to_numpy(dtype=str)
    np.savez_compressed(args.out, H=K[hist], hist_names=names[hist], C=K[cand], cand_names=names[cand])
    table = os.path.splitext(args.out)[0] + '_models.csv'
    df.drop(columns=[c for c in df.columns if c.startswith('test_acc_by_')]).to_csv(table, index=False)

    acc = K.mean(1)
    print(f'N = {K.shape[1]} test samples, {len(df)} models from {len(runs)} runs, success vector {args.key!r}')
    print(f'H: M = {len(hist)} historical models, accuracy {acc[hist].min():.3f} .. {acc[hist].max():.3f}')
    print(f'{len(cand)} candidates:')
    for i in cand:
        print(f'  {names[i]:40s} K* = {acc[i]:.4f}')
    print(f'saved {args.out} and {table}')


if __name__ == '__main__':
    main()
