import argparse
import json
from pathlib import Path

import numpy as np

PER_SAMPLE = ('slot_ok', 'cw_err', 'nmse', 'ebno_db', 'batch')
SLOTS = 4


def load(path):
    z = dict(np.load(path))
    ebnos = np.unique(z['ebno_db'])
    n_batch = z['slot_ok'].size // (len(ebnos) * SLOTS)
    if 'batch' not in z:
        z['batch'] = np.tile(np.repeat(np.arange(n_batch), SLOTS), len(ebnos)).astype(np.int16)
    return z, ebnos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='zoo_sb')
    ap.add_argument('--ext', default='zoo_sb_ext')
    ap.add_argument('--ebno_min', type=float, default=2.5)
    ap.add_argument('--out', default='zoo_sb460')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    names = sorted(p.stem for p in Path(args.base).glob('*.json'))
    missing = [n for n in names if not Path(args.ext, f'{n}.json').exists()]
    if missing:
        raise SystemExit(f'no extension for {missing}')
    signatures = set()
    for n in names:
        meta = json.loads(Path(args.base, f'{n}.json').read_text())
        ext_meta = json.loads(Path(args.ext, f'{n}.json').read_text())
        (b, b_ebnos), (e, e_ebnos) = load(Path(args.base, f'{n}.npz')), load(Path(args.ext, f'{n}.npz'))
        ebnos = b_ebnos[b_ebnos >= args.ebno_min - 1e-6]
        if not np.array_equal(ebnos, e_ebnos) or b['batch'].max() >= e['batch'].min():
            raise SystemExit(f'{n}: extension does not continue the base (Eb/N0 {e_ebnos}, batches)')
        parts = [{k: z[k][z['ebno_db'] == v] for k in PER_SAMPLE} for v in ebnos for z in (b, e)]
        merged = {k: np.concatenate([p[k] for p in parts]) for k in PER_SAMPLE}
        rows = np.isin(b_ebnos, ebnos)
        seconds = np.concatenate([b['batch_seconds'][rows], e['batch_seconds']], axis=1)
        np.savez_compressed(out / f'{n}.npz', **merged, batch_seconds=seconds)
        ok = merged['slot_ok']
        signature = f"{meta['test_signature']}+{ext_meta['test_signature']}+ebno>={args.ebno_min:g}"
        signatures.add(signature)
        meta.update(test_signature=signature, n_samples=int(ok.size), K_slot=float(ok.mean()),
                    slot_success_by_ebno={float(v): float(ok[merged['ebno_db'] == v].mean()) for v in ebnos},
                    n_batches=int(seconds.shape[1]), ebno_min=args.ebno_min,
                    seconds_per_slot_ext=ext_meta['seconds_per_slot'],
                    eval_seconds_ext=ext_meta['eval_seconds'])
        (out / f'{n}.json').write_text(json.dumps(meta, indent=1))
        print(f'{n:22s} N = {ok.size}, K* = {ok.mean():.4f}')
    if len(signatures) != 1:
        raise SystemExit(f'models of different benchmarks: {signatures}')
    print(f'{len(names)} models merged into {out}/')


if __name__ == '__main__':
    main()
