import argparse
import json
from pathlib import Path

import numpy as np


def find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='sionna-bench')
    ap.add_argument('--work', default='work_sb')
    args = ap.parse_args()
    for res in sorted(Path(args.work).glob('*/*/*/results.json')):
        model, paper, scenario = res.parts[-4:-1]
        canon_dir = Path(args.repo, 'outputs/mimo_ofdm_chest', paper, scenario)
        if not (canon_dir / 'results.json').exists():
            print(f'{model:22s} no canonical results of {paper}')
            continue
        canon = json.loads((canon_dir / 'results.json').read_text())['results']
        n = find(json.loads((canon_dir / 'config.json').read_text()), 'num_mc_batches')
        for name, entry in json.loads(res.read_text())['results'].items():
            ref = canon.get(name)
            slots = res.parent / Path(entry['file']).with_suffix('.slots.npz').name
            if ref is None or ref['constructor_arguments'] != entry['constructor_arguments'] or not slots.exists():
                print(f'{model:22s} {name}: no canonical counterpart')
                continue
            z, c = np.load(slots), np.load(canon_dir / ref['file'])
            bler = z['cw_err'][:, :n].sum((1, 2)) / z['num_cw'][:, :n].sum((1, 2))
            ber = z['bit_err'][:, :n].sum((1, 2)) / z['num_bits'][:, :n].sum((1, 2))
            nmse = z['nmse_num'][:, :n].sum((1, 2)) / z['nmse_den'][:, :n].sum((1, 2))
            num_cw = z['num_cw'][:, :n].sum((1, 2))
            d_cw = np.abs(bler - c['bler']) * num_cw
            print(f'{model:22s} {name}: max |dBLER| = {np.abs(bler - c["bler"]).max():.2e} '
                  f'({d_cw.max():.0f} of {num_cw[0]} codewords), max |dBER|/BER = '
                  f'{np.max(np.abs(ber - c["ber"]) / np.maximum(c["ber"], 1e-12)):.2e}, max |dNMSE|/NMSE = '
                  f'{np.max(np.abs(nmse - c["nmse"]) / c["nmse"]):.2e}')


if __name__ == '__main__':
    main()
