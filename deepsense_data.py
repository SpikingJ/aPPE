import argparse
import glob
import hashlib
import io
import json
import multiprocessing as mp
import os
import posixpath
import re
import zipfile

import numpy as np
import pandas as pd

N_BEAMS = 64
EARTH_RADIUS = 6371000.0
FLOAT = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def find_sources(root, numbers):
    entries = [e for e in sorted(glob.glob(os.path.join(glob.escape(root), '*')))
               if os.path.isdir(e) or e.lower().endswith('.zip')]
    out = {}
    for n in numbers:
        pat = re.compile(rf'scenario[\s_-]*0*{n}(?!\d)', re.IGNORECASE)
        hits = [e for e in entries if pat.search(os.path.basename(e))]
        hits = [e for e in hits if e.lower().endswith('.zip')] or hits
        if len(hits) != 1:
            raise SystemExit(f'scenario {n}: expected one zip file or folder matching "scenario{n}" in {root}, '
                             f'found {hits}')
        out[n] = hits[0]
    return out


def find_csv(folder):
    for d in [folder] + sorted(glob.glob(os.path.join(glob.escape(folder), '*', ''))):
        csvs = sorted(glob.glob(os.path.join(glob.escape(d), '*.csv')))
        csvs = [c for c in csvs if 'scenario' in os.path.basename(c).lower()] or csvs
        if csvs:
            return csvs[0], os.path.dirname(csvs[0])
    raise SystemExit(f'no csv file in {folder}')


def detect_columns(df, path):
    low = {c.lower().strip(): c for c in df.columns}

    def first(pred):
        return next((low[c] for c in low if pred(c)), None)

    cols = dict(rgb=first(lambda c: 'rgb' in c), pwr=first(lambda c: 'pwr' in c),
                bs=first(lambda c: c == 'unit1_loc') or first(lambda c: c.startswith('unit1_loc')),
                ue=first(lambda c: c == 'unit2_loc_cal') or first(lambda c: c == 'unit2_loc')
                or first(lambda c: c.startswith('unit2_loc')),
                beam=first(lambda c: c in ('unit1_beam_index', 'unit1_beam', 'beam_index')),
                seq=first(lambda c: c == 'seq_index'))
    missing = [k for k in ('rgb', 'pwr', 'bs', 'ue') if cols[k] is None]
    if missing:
        raise SystemExit(f'{path}: no column for {missing}; columns are {list(df.columns)}')
    return cols


class Source:
    def __init__(self, path):
        self.path, self.is_zip = path, path.lower().endswith('.zip')
        if self.is_zip:
            with zipfile.ZipFile(path) as z:
                self.members = {m for m in z.namelist() if not m.endswith('/') and not m.startswith('__MACOSX')}
            csvs = [m for m in self.members if m.lower().endswith('.csv')]
            if not csvs:
                raise SystemExit(f'no csv file in {path}')
            self.csv = min(csvs, key=lambda m: (m.count('/'), 'scenario' not in posixpath.basename(m).lower(), m))
            self.base = posixpath.dirname(self.csv)
        else:
            self.csv, self.base = find_csv(path)

    def read_csv(self):
        if not self.is_zip:
            return pd.read_csv(self.csv)
        with zipfile.ZipFile(self.path) as z:
            return pd.read_csv(io.BytesIO(z.read(self.csv)))

    def resolve(self, rel):
        rel = str(rel).strip().replace('\\', '/')
        if self.is_zip:
            member = posixpath.normpath(posixpath.join(self.base, rel))
            if member in self.members:
                return member
            raise FileNotFoundError(f'{rel} (in {self.path})')
        for p in (os.path.join(self.base, rel), os.path.join(self.base, rel.lstrip('./'))):
            if os.path.exists(p):
                return os.path.normpath(p)
        raise FileNotFoundError(f'{rel} (relative to {self.base})')


_ZIPS = {}


def read_bytes(src, ref):
    if src.lower().endswith('.zip'):
        if src not in _ZIPS:
            _ZIPS[src] = zipfile.ZipFile(src)
        return _ZIPS[src].read(ref)
    with open(ref, 'rb') as f:
        return f.read()


def parse_floats(data):
    return np.array([float(v) for v in FLOAT.findall(data.decode())])


def load_sample(job):
    src, img_ref, pwr_ref, ue_ref, size = job
    from PIL import Image
    bilinear = getattr(Image, 'Resampling', Image).BILINEAR
    with Image.open(io.BytesIO(read_bytes(src, img_ref))) as im:
        img = np.asarray(im.convert('RGB').resize((size[1], size[0]), bilinear), dtype=np.uint8)
    pwr, ue = parse_floats(read_bytes(src, pwr_ref)), parse_floats(read_bytes(src, ue_ref))
    if pwr.size != N_BEAMS or ue.size < 2:
        raise ValueError(f'{pwr_ref}: {pwr.size} values (expected {N_BEAMS}); {ue_ref}: {ue.size} values')
    return img, pwr, ue[:2]


def local_en(lat, lon, lat0, lon0):
    east = np.radians(lon - lon0) * EARTH_RADIUS * np.cos(np.radians(lat0))
    north = np.radians(lat - lat0) * EARTH_RADIUS
    return np.stack([east, north], 1).astype(np.float32)


def relative_db(pwr, scale):
    if scale == 'db':
        return (pwr - pwr.max(1, keepdims=True)).astype(np.float32)
    pwr = np.maximum(pwr, 1e-30)
    return (10 * np.log10(pwr / pwr.max(1, keepdims=True))).astype(np.float32)


def load_scenario(n, path, args, pool):
    src = Source(path)
    df = src.read_csv()
    cols = detect_columns(df, src.csv)
    ok = df[[cols['rgb'], cols['pwr'], cols['ue']]].notna().all(1).values
    jobs, missing = [], []
    for i in np.flatnonzero(ok):
        try:
            jobs.append((path, src.resolve(df.at[i, cols['rgb']]), src.resolve(df.at[i, cols['pwr']]),
                         src.resolve(df.at[i, cols['ue']]), args.size))
        except FileNotFoundError as e:
            ok[i] = False
            missing.append(str(e))
    if len(missing) > 0.01 * len(df):
        raise SystemExit(f'scenario {n}: {len(missing)} of {len(df)} samples have missing files, e.g. {missing[:3]}')
    df = df[ok].reset_index(drop=True)
    rows = np.flatnonzero(ok).astype(np.int32)
    print(f'scenario {n}: {path} ({src.csv}), {len(jobs)} samples ({len(ok) - len(jobs)} rows skipped, {len(missing)} of them '
          f'with missing files{", e.g. " + missing[0] if missing else ""}); columns {cols}', flush=True)
    out = pool.map(load_sample, jobs, chunksize=64)
    img = np.stack([o[0] for o in out])
    pwr = np.stack([o[1] for o in out])
    ue = np.stack([o[2] for o in out])
    bs = parse_floats(read_bytes(path, src.resolve(df[cols['bs']].iloc[0])))[:2]
    scale = args.power_scale if args.power_scale != 'auto' else ('db' if (pwr < 0).any() else 'linear')
    pwr_db = relative_db(pwr, scale)
    beam = pwr.argmax(1).astype(np.int16)
    info = dict(source=path, csv=src.csv, columns=cols, n=len(beam), power_scale=scale, bs_lat_lon=bs.tolist(),
                power_raw=dict(min=float(pwr.min()), median=float(np.median(pwr)), max=float(pwr.max())),
                second_beam_db_median=float(np.median(np.sort(pwr_db, 1)[:, -2])))
    if cols['beam'] is not None:
        csv_beam = df[cols['beam']].values.astype(np.int64)
        info['csv_beam_agreement'] = dict(zero_based=float(np.mean(csv_beam == beam)),
                                          one_based=float(np.mean(csv_beam - 1 == beam)))
    if cols['seq'] is not None:
        seq = df[cols['seq']].values.astype(np.int64)
        info['grouping'] = 'seq_index'
    else:
        seq = np.arange(len(beam)) // args.block
        info['grouping'] = f'blocks of {args.block} rows'
    pos = local_en(ue[:, 0], ue[:, 1], bs[0], bs[1])
    info['distance_m'] = dict(min=float(np.hypot(*pos.T).min()), max=float(np.hypot(*pos.T).max()))
    info['n_passes'] = int(len(np.unique(seq)))
    seq = (n * 100000 + seq).astype(np.int32)
    arrays = dict(img=img, pos=pos, pwr=pwr_db, beam=beam, scen=np.full(len(beam), n, np.int16), seq=seq, row=rows)
    return arrays, info


def split_passes(scen, seq, test_frac, val_frac, seed):
    role = np.full(len(scen), 'train', dtype='<U5')
    for s in np.unique(scen):
        m = scen == s
        groups, counts = np.unique(seq[m], return_counts=True)
        order = np.random.default_rng([seed, int(s)]).permutation(len(groups))
        before = (np.cumsum(counts[order]) - counts[order]) / counts.sum()
        test = groups[order[before < test_frac]]
        val = groups[order[(before >= test_frac) & (before < test_frac + val_frac)]]
        role[m & np.isin(seq, test)] = 'test'
        role[m & np.isin(seq, val)] = 'val'
    return role


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/raw_deepsense',
                    help='folder with the scenario zip files (or the extracted scenario folders)')
    ap.add_argument('--scenarios', type=int, nargs='+', default=list(range(1, 10)))
    ap.add_argument('--dirs', nargs='*', default=None,
                    help='explicit scenario zip files / folders instead of --root/--scenarios')
    ap.add_argument('--out', default='data/deepsense')
    ap.add_argument('--size', type=int, nargs=2, default=[144, 256], metavar=('H', 'W'), help='stored image size')
    ap.add_argument('--test_frac', type=float, default=0.5, help='share of every scenario (whole passes) in D')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--block', type=int, default=100, help='rows per group if a scenario has no seq_index')
    ap.add_argument('--power_scale', default='auto', choices=['auto', 'linear', 'db'],
                    help='scale of the power files (auto: dB if any value is negative, else linear)')
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.size = tuple(args.size)

    if args.dirs:
        folders = {}
        for d in args.dirs:
            m = re.search(r'scenario[\s_-]*0*(\d+)', os.path.basename(os.path.normpath(d)), re.IGNORECASE)
            if not m:
                raise SystemExit(f'cannot read the scenario number from the name {d}')
            folders[int(m.group(1))] = d
    else:
        folders = find_sources(args.root, args.scenarios)

    parts, infos = [], {}
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for n, folder in sorted(folders.items()):
            arrays, infos[n] = load_scenario(n, folder, args, pool)
            parts.append(arrays)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    role = split_passes(data['scen'], data['seq'], args.test_frac, args.val_frac, args.seed)

    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for split in ('train', 'val', 'test'):
        m = role == split
        for k, v in data.items():
            np.save(os.path.join(args.out, f'{split}_{k}.npy'), v[m])
        counts[split] = {int(s): int(np.sum(m & (data['scen'] == s))) for s in folders}
    test = role == 'test'
    signature = hashlib.sha1(np.stack([data['scen'][test].astype(np.int64),
                                       data['row'][test].astype(np.int64)]).tobytes()).hexdigest()[:16]
    meta = dict(dataset='DeepSense 6G', n_beams=N_BEAMS, image_size=list(args.size), scenarios=sorted(folders),
                test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed, counts=counts,
                test_signature=signature, scenario_info=infos)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    print(f"\n{'scen':>4} {'samples':>7} {'passes':>6} {'train':>6} {'val':>5} {'test':>5}  power scale, "
          f"median 2nd-best beam [dB], csv beam agreement")
    for n in sorted(folders):
        i = infos[n]
        agree = i.get('csv_beam_agreement', {})
        print(f"{n:4d} {i['n']:7d} {i['n_passes']:6d} {counts['train'][n]:6d} {counts['val'][n]:5d} "
              f"{counts['test'][n]:5d}  {i['power_scale']}, {i['second_beam_db_median']:.2f}, {agree}")
    print(f"total: train {int((role == 'train').sum())}, val {int((role == 'val').sum())}, "
          f"test (N) {int(test.sum())}; saved to {args.out}")


if __name__ == '__main__':
    main()
