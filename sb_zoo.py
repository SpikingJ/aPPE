import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def M(name, paper, argv=(), *, arch, run=None, assets=False, own_assets=False):
    return dict(name=name, paper=paper, argv=list(argv), arch=arch, run=run or name, assets=assets,
                own_assets=own_assets)


REESNET_BANK = ['--training-data-manifest', 'assets_sb/3gpp-umi/li2020/reesnet-training-data/train6400-val256-snrm10-'
                'm7p5-m5-m2p5-0-2p5-5-7p5-10-12p5-15-paper-speed-seed42/manifest.json']

ZOO = [
    M('ls_nn', 'builtin', ['--estimator', 'ls-nn'], arch='ls'),
    M('ls_lin', 'builtin', ['--estimator', 'ls-lin'], arch='ls'),
    M('coleri_second', 'coleri2002', ['--method', 'second-order'], arch='ls-interp'),
    M('coleri_lowpass', 'coleri2002', ['--method', 'low-pass'], arch='ls-interp'),
    M('coleri_spline', 'coleri2002', ['--method', 'cubic-spline'], arch='ls-interp'),
    M('coleri_time', 'coleri2002', ['--method', 'time-domain'], arch='ls-interp'),
    M('coleri_lowpass_c0.05', 'coleri2002', ['--method', 'low-pass', '--lowpass-cutoff', '0.05'], arch='ls-interp'),
    M('coleri_lowpass_c0.2', 'coleri2002', ['--method', 'low-pass', '--lowpass-cutoff', '0.2'], arch='ls-interp'),
    M('coleri_second_ev0.4', 'coleri2002', ['--method', 'second-order', '--err-var-scale', '0.4'], arch='ls-interp'),
    M('coleri_spline_ev1.5', 'coleri2002', ['--method', 'cubic-spline', '--err-var-scale', '1.5'], arch='ls-interp'),
    M('sbl', 'wei2017', [], arch='sparsity'),
    M('sbl_it3', 'wei2017', ['--num-iter', '3'], arch='sparsity'),
    M('sbl_it30', 'wei2017', ['--num-iter', '30'], arch='sparsity'),
    M('sbl_nn', 'wei2017', ['--interpolation-type', 'nn'], arch='sparsity'),
    M('sbl_lin', 'wei2017', ['--interpolation-type', 'lin'], arch='sparsity'),
    M('lmmse', 'builtin', ['--estimator', 'lmmse'], arch='lmmse', assets=True),
    M('lmmse_cov10', 'builtin', ['--estimator', 'lmmse', '--cov-num-iters', '10'], arch='lmmse', assets=True,
      own_assets=True),
    M('edfors_lmmse', 'edfors1996', ['--method', 'LMMSE'], arch='delay-trunc', assets=True),
    M('edfors_diag', 'edfors1996', ['--method', 'diagonal LMMSE'], arch='delay-trunc', assets=True),
    M('edfors_ls', 'edfors1996', ['--method', 'LS'], arch='delay-trunc', assets=True),
    M('edfors_lmmse_t16', 'edfors1996', ['--method', 'LMMSE', '--num-taps', '16'], arch='delay-trunc', assets=True),
    M('edfors_ls_t16', 'edfors1996', ['--method', 'LS', '--num-taps', '16'], arch='delay-trunc', assets=True),
    M('edfors_lmmse_t128', 'edfors1996', ['--method', 'LMMSE', '--num-taps', '128'], arch='delay-trunc', assets=True),
    M('edfors_lmmse_cov10', 'edfors1996', ['--method', 'LMMSE', '--cov-num-iters', '10'], arch='delay-trunc',
      assets=True),
    M('edfors_ls_nn', 'edfors1996', ['--method', 'LS', '--interpolation-type', 'nn'], arch='delay-trunc',
      assets=True),
    M('gmm_spectral', 'koller2022', ['--method', 'spectral'], arch='gmm', assets=True),
    M('gmm_spatial', 'koller2022', ['--method', 'spatial'], arch='gmm', assets=True),
    M('gmm_spectral_k4', 'koller2022', ['--method', 'spectral', '--gmm-components', '4'], arch='gmm',
      assets=True),
    M('gmm_spatial_k4', 'koller2022', ['--method', 'spatial', '--gmm-components', '4'], arch='gmm',
      assets=True),
    M('gmm_spatial_k8_tb20', 'koller2022', ['--method', 'spatial', '--gmm-components', '8', '--train-batches', '20'],
      arch='gmm', assets=True),
    M('gmm_spectral_s7', 'koller2022', ['--method', 'spectral', '--model-seed', '7'], arch='gmm', assets=True),
    M('reesnet', 'li2020', [], arch='resnet', assets=True),
    M('reesnet_s43', 'li2020', ['--model-seed', '43'], arch='resnet', assets=True),
    M('reesnet_ep50', 'li2020', ['--epochs', '50', *REESNET_BANK], arch='resnet', assets=True),
    M('reesnet_tb100', 'li2020', ['--train-batches', '100'], arch='resnet', assets=True),
    M('dnn_m2', 'melgar2022', ['--method', 'model2'], arch='dnn', assets=True),
    M('dnn_m2_s43', 'melgar2022', ['--method', 'model2', '--model-seed', '43'], arch='dnn', assets=True),
    M('dnn_m3', 'melgar2022', ['--method', 'model3'], arch='dnn', assets=True),
    M('dnn_m3_s43', 'melgar2022', ['--method', 'model3', '--model-seed', '43'], arch='dnn', assets=True),
    M('dnn_m2_mixed', 'melgar2022', ['--method', 'model2', '--mixed-snr-weights'], arch='dnn', assets=True),
    M('sparse_diffuse', 'lyu2026', [], arch='sparsity'),
    M('deep_prior', 'balevi2020', [], arch='untrained-nn'),
    M('deep_prior_s400', 'balevi2020', ['--num-steps', '400'], arch='untrained-nn'),
]
SOLO = {'deep_prior', 'deep_prior_s400', 'sparse_diffuse'}
PRIORITY = ['builtin', 'coleri2002', 'wei2017', 'koller2022', 'edfors1996',
            'balevi2020:deep_prior', 'lyu2026:sparse_diffuse', 'li2020',
            'melgar2022', 'balevi2020:deep_prior_s400']


def group(spec, solo=False):
    return f"{spec['paper']}:{spec['name']}" if solo or spec['name'] in SOLO else spec['paper']


def claim(zoo, g):
    try:
        fd = os.open(Path(zoo, f"{g.replace(':', '__')}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {os.environ.get('CUDA_VISIBLE_DEVICES', '')}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def run(cmd, log):
    with open(log, 'a') as f:
        f.write(f'$ {" ".join(cmd)}\n')
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE).returncode


def build_model(spec, args):
    work = Path(args.work) / spec['name']
    log = Path(args.logs) / f"{spec['name']}.log"
    common = ['--output-dir', str(work), '--num-mc-batches', str(args.batches), '--device', 'cuda:0']
    root = Path(args.assets) / spec['name'] if spec['own_assets'] else Path(args.assets)
    asset = ['--asset-root', str(root)] if spec['assets'] else []
    subset = ['--first-batch', str(args.first_batch)] + ([] if args.ebno_min is None else ['--ebno-min', str(args.ebno_min)])
    t0 = time.time()
    if spec['assets'] and not args.no_prepare:
        run([sys.executable, 'sb_run.py', spec['paper'], *common, *asset, '--prepare-assets', *spec['argv']], log)
    t1 = time.time()
    if run([sys.executable, 'sb_run.py', *subset, spec['paper'], *common, *asset, *spec['argv']], log) != 0:
        raise RuntimeError(f"evaluation failed, see {log}")
    slots = sorted(work.rglob('*.slots.npz'), key=os.path.getmtime)
    if not slots:
        raise RuntimeError(f'no per-slot results in {work}')
    z = np.load(slots[-1])
    ok = (z['cw_err'] == 0).reshape(-1).astype(np.uint8)
    n_ebno, n_batch, n_slot = z['cw_err'].shape
    ebno = np.repeat(z['ebno_dbs'], n_batch * n_slot).astype(np.float32)
    batch_index = z['batch_index'] if 'batch_index' in z else np.arange(n_batch)
    batch = np.tile(np.repeat(batch_index, n_slot), n_ebno).astype(np.int16)
    nmse = (z['nmse_num'] / np.maximum(z['nmse_den'], 1e-30)).reshape(-1).astype(np.float32)
    np.savez_compressed(Path(args.zoo) / f"{spec['name']}.npz", slot_ok=ok, cw_err=z['cw_err'].reshape(-1),
                        nmse=nmse, ebno_db=ebno, batch=batch, batch_seconds=z['batch_seconds'].astype(np.float32))
    by_snr = (z['cw_err'] == 0).reshape(len(z['ebno_dbs']), -1).mean(1)
    meta = dict(name=spec['name'], run=spec['run'], arch=spec['arch'], paper=spec['paper'], argv=spec['argv'],
                is_snapshot=False, test_signature=str(z['config_fingerprint']), n_samples=int(ok.size),
                K_slot=float(ok.mean()), slot_success_by_ebno=dict(zip(map(float, z['ebno_dbs']), map(float, by_snr))),
                seconds_per_slot=float(z['batch_seconds'].mean() / z['cw_err'].shape[2]),
                asset_seconds=t1 - t0, eval_seconds=time.time() - t1, first_batch=args.first_batch,
                ebno_min=args.ebno_min)
    Path(args.zoo, f"{spec['name']}.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zoo', default='zoo_sb')
    ap.add_argument('--work', default='work_sb')
    ap.add_argument('--assets', default='assets_sb')
    ap.add_argument('--logs', default='logs/zoo')
    ap.add_argument('--batches', type=int, default=250, help='Monte Carlo batches of 4 slots per Eb/N0')
    ap.add_argument('--first_batch', type=int, default=0, help='evaluate only batches first_batch, ..., batches - 1')
    ap.add_argument('--ebno_min', type=float, default=None, help='evaluate only the Eb/N0 values >= ebno_min')
    ap.add_argument('--no_prepare', action='store_true', help='use the existing assets, do not prepare them')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--solo', action='store_true',
                    help='every model is a group of its own (to build models in parallel whose shared assets exist)')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    for d in (args.zoo, args.work, args.assets, args.logs):
        os.makedirs(d, exist_ok=True)
    if args.list:
        for s in ZOO:
            done = Path(args.zoo, f"{s['name']}.json")
            info = json.loads(done.read_text()) if done.exists() else None
            status = f"K={info['K_slot']:.3f} {info['seconds_per_slot']:.3f} s/slot" if info else '-'
            print(f"{group(s):28s} {s['name']:22s} {s['paper']:11s} {' '.join(s['argv']):45s} {status}")
        return
    todo = [s for s in ZOO if (args.only is None or s['name'] in args.only)
            and not Path(args.zoo, f"{s['name']}.json").exists()]
    groups = sorted({group(s, args.solo) for s in todo},
                    key=lambda g: PRIORITY.index(g) if g in PRIORITY else len(PRIORITY))
    for g in groups:
        if not claim(args.zoo, g):
            continue
        print(f"{time.strftime('%m-%d %H:%M')} group {g}", flush=True)
        for s in [s for s in todo if group(s, args.solo) == g]:
            t0 = time.time()
            try:
                meta = build_model(s, args)
                print(f"{time.strftime('%m-%d %H:%M')} {s['name']}: K_slot = {meta['K_slot']:.4f}, "
                      f"{meta['seconds_per_slot']:.3f} s/slot, {time.time() - t0:.0f} s", flush=True)
            except Exception as e:
                print(f"{time.strftime('%m-%d %H:%M')} {s['name']}: FAILED ({e})", flush=True)


if __name__ == '__main__':
    main()
