import argparse
import hashlib
import json
import os
import time

import h5py
import numpy as np

CLASSES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
           '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
           'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
SPLITS = ('test', 'val', 'train')


def load_labels(f, chunk):
    n = f['Y'].shape[0]
    y = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        y[s:s + chunk] = np.argmax(f['Y'][s:s + chunk], axis=1)
    snr = f['Z'][:, 0].astype(np.int64)
    return y, snr


def stratified_split(y, snr, sizes, seed):
    snr_values = np.unique(snr)
    cell = y * len(snr_values) + np.searchsorted(snr_values, snr)
    order = np.argsort(cell, kind='stable')
    bounds = np.searchsorted(cell[order], np.arange(cell.max() + 2))
    need = sum(sizes.values())
    rng = np.random.default_rng(seed)
    out = {s: [] for s in SPLITS}
    for c in range(len(bounds) - 1):
        members = order[bounds[c]:bounds[c + 1]]
        if len(members) == 0:
            continue
        if len(members) < need:
            raise ValueError(f'cell {c} has {len(members)} frames, {need} requested')
        members = rng.permutation(members)
        start = 0
        for s in SPLITS:
            out[s].append(members[start:start + sizes[s]])
            start += sizes[s]
    return {s: np.sort(np.concatenate(v)) for s, v in out.items()}, snr_values


def copy_frames(f, idx, out_dir, dtype, chunk):
    X = f['X']
    n, L, _ = X.shape
    arrays = {s: np.lib.format.open_memmap(os.path.join(out_dir, f'{s}_X.npy'), mode='w+',
                                           dtype=dtype, shape=(len(i), 2, L)) for s, i in idx.items()}
    pos = {s: 0 for s in idx}
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = {s: i[np.searchsorted(i, start):np.searchsorted(i, end)] - start for s, i in idx.items()}
        if not any(len(r) for r in rows.values()):
            continue
        block = X[start:end]
        for s, r in rows.items():
            arrays[s][pos[s]:pos[s] + len(r)] = block[r].transpose(0, 2, 1)
            pos[s] += len(r)
        print(f'  read {end}/{n} frames ({time.time() - t0:.0f}s)', flush=True)
    for a in arrays.values():
        a.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5', required=True, help='path to GOLD_XYZ_OSC.0001_1024.hdf5')
    ap.add_argument('--out', default='data/radioml2018')
    ap.add_argument('--n_test', type=int, default=32, help='frames per (class, SNR) cell in the benchmark D')
    ap.add_argument('--n_val', type=int, default=64, help='frames per cell for validation')
    ap.add_argument('--n_train', type=int, default=1024, help='frames per cell in the training pool')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'float16'])
    ap.add_argument('--chunk', type=int, default=32768, help='frames read from the HDF5 file at a time')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.h5.endswith('.zip'):
        raise SystemExit(f'{args.h5} is the zip archive; extract it first (needs ~21.4 GB), e.g.\n'
                         f'  unzip {args.h5} -d <dir>   or   python -m zipfile -e {args.h5} <dir>\n'
                         f'and pass <dir>/GOLD_XYZ_OSC.0001_1024.hdf5 to --h5')
    os.makedirs(args.out, exist_ok=True)
    sizes = {'test': args.n_test, 'val': args.n_val, 'train': args.n_train}
    with h5py.File(args.h5, 'r') as f:
        print(f"X {f['X'].shape}, Y {f['Y'].shape}, Z {f['Z'].shape}")
        y, snr = load_labels(f, args.chunk)
        idx, snr_values = stratified_split(y, snr, sizes, args.seed)
        for s in SPLITS:
            print(f'{s}: {len(idx[s])} frames')
        copy_frames(f, idx, args.out, np.dtype(args.dtype), args.chunk)
        frame_len = f['X'].shape[1]

    for s in SPLITS:
        np.save(os.path.join(args.out, f'{s}_y.npy'), y[idx[s]])
        np.save(os.path.join(args.out, f'{s}_snr.npy'), snr[idx[s]])
        np.save(os.path.join(args.out, f'{s}_idx.npy'), idx[s])
    meta = dict(source=os.path.abspath(args.h5), classes=CLASSES[:int(y.max()) + 1],
                snr_values=[int(v) for v in snr_values], frame_len=int(frame_len), seed=args.seed,
                per_cell=sizes, counts={s: int(len(idx[s])) for s in SPLITS}, dtype=args.dtype,
                test_signature=hashlib.md5(idx['test'].tobytes()).hexdigest())
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'done -> {args.out}  (benchmark D: N = {len(idx["test"])})')


if __name__ == '__main__':
    main()
