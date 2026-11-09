import argparse
import copy
import json
import math
import os
import signal
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from ds_models import PerScenario, build_neural, position_features
from train_zoo import autocast, claim, parse_runs, resolve_amp

DAY, NIGHT = [1, 3, 6, 7, 8, 9], [2, 4, 5]

VISION = [
    ('resnet18', (144, 256), 1.0, 12),
    ('resnet18', (72, 128), 0.3, 12),
    ('resnet34', (144, 256), 0.5, 12),
    ('resnet50', (144, 256), 1.0, 12),
    ('mobilenet_v3_small', (144, 256), 1.0, 15),
    ('mobilenet_v3_large', (72, 128), 1.0, 15),
    ('efficientnet_b0', (144, 256), 0.3, 15),
    ('regnet_y_400mf', (144, 256), 1.0, 12),
    ('shufflenet_v2_x1_0', (144, 256), 0.5, 15),
    ('convnext_tiny', (144, 256), 1.0, 10),
    ('densenet121', (72, 128), 0.5, 12),
]
MLP = [
    (64, 2, 0.3, 60),
    (256, 3, 1.0, 60),
    (512, 4, 1.0, 60),
]
FUSION = [
    ('resnet18', (144, 256), 1.0, 12),
    ('mobilenet_v3_small', (72, 128), 0.5, 15),
]
CLASSIC = [
    ('knn', dict(k=1), 1.0, [0]),
    ('knn', dict(k=5), 1.0, [0]),
    ('knn', dict(k=25), 1.0, [0]),
    ('knn', dict(k=5), 0.3, [0, 1, 2]),
    ('lut', dict(bins=16), 1.0, [0]),
    ('lut', dict(bins=64), 1.0, [0]),
    ('rf', dict(depth=None), 1.0, [0, 1, 2]),
    ('rf', dict(depth=6), 1.0, [0]),
    ('et', {}, 1.0, [0, 1, 2]),
    ('hgb', {}, 1.0, [0]),
    ('logreg', {}, 1.0, [0]),
    ('gnb', {}, 1.0, [0]),
]
SPECIALISTS = [
    ('vision', 'resnet18', (144, 256), 12, DAY, 'day', [0, 1]),
    ('vision', 'mobilenet_v3_large', (72, 128), 15, NIGHT, 'night', [0]),
    ('mlp', dict(hidden=256, depth=3), None, 60, [1, 2, 3, 4], 'sc1-4', [0, 1]),
    ('fusion', 'resnet18', (144, 256), 12, [1, 2, 3, 4], 'sc1-4', [0]),
]
HYPER = {
    'vision': dict(lr=5e-4, wd=0.05, bs=128, snapshots=[1, 3]),
    'fusion': dict(lr=5e-4, wd=0.05, bs=128, snapshots=[1, 3]),
    'mlp': dict(lr=2e-3, wd=1e-4, bs=256, snapshots=[4, 12]),
    'classic': dict(),
}


def neural_run(family, arch, size, frac, epochs, seed, scen_train=None, tag=''):
    if family == 'mlp':
        kw, arch, label = dict(arch), 'mlp', f"h{arch['hidden']}x{arch['depth']}"
    else:
        kw, label = {}, f'{arch}_r{size[0]}'
    return dict(family=family, arch=arch, size=size, train_frac=frac, scen_train=scen_train, epochs=epochs,
                seed=seed, kw=kw, label=f"{family}_{label}_f{frac:g}{'_' + tag if tag else ''}", **HYPER[family])


def build_zoo():
    runs = []
    for s in range(3):
        runs += [neural_run('vision', a, size, f, e, s) for a, size, f, e in VISION]
        runs += [neural_run('mlp', dict(hidden=h, depth=d), None, f, e, s) for h, d, f, e in MLP]
        runs += [neural_run('fusion', a, size, f, e, s) for a, size, f, e in FUSION]
    for method, kw, f, seeds in CLASSIC:
        tag = ''.join(f'_{k[0]}{v}' for k, v in kw.items() if v is not None)
        runs += [dict(family='classic', arch=method, size=None, train_frac=f, scen_train=None, epochs=None, seed=s,
                      kw=kw, label=f'classic_{method}{tag}_f{f:g}') for s in seeds]
    for family, arch, size, e, scen, tag, seeds in SPECIALISTS:
        runs += [neural_run(family, arch, size, 1.0, e, s, scen, tag) for s in seeds]
    for i, r in enumerate(runs):
        r['name'] = f"{i:03d}_{r.pop('label')}_s{r['seed']}"
    return runs


class Split:
    def __init__(self, data_dir, name, device, scenarios, img_device):
        load = lambda k, **kw: np.load(os.path.join(data_dir, f'{name}_{k}.npy'), **kw)
        self.scen, self.seq, self.pos = load('scen'), load('seq'), load('pos')
        self.beam_np = load('beam').astype(np.int64)
        self.beam = torch.from_numpy(self.beam_np).to(device)
        self.pwr = torch.from_numpy(load('pwr')).to(device)
        self.feats = torch.from_numpy(position_features(self.pos, self.scen, scenarios)).to(device)
        self.img = None if img_device is None else torch.from_numpy(np.ascontiguousarray(load('img'))).to(img_device)

    def __len__(self):
        return len(self.beam_np)


def training_subset(split, frac, scen_train, seed):
    keep = np.ones(len(split), bool) if scen_train is None else np.isin(split.scen, scen_train)
    if frac < 1:
        rng, chosen = np.random.default_rng(seed), np.zeros(len(split), bool)
        for s in np.unique(split.scen[keep]):
            m = keep & (split.scen == s)
            passes = np.unique(split.seq[m])
            pick = rng.choice(passes, size=max(1, round(frac * len(passes))), replace=False)
            chosen |= m & np.isin(split.seq, pick)
        keep = chosen
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        raise ValueError(f'no training samples for scenarios {scen_train} in this dataset')
    return idx


def fetch(split, j, device, with_img):
    img = split.img[j.to(split.img.device)].to(device, non_blocking=True) if with_img else None
    return img, split.feats[j], split.beam[j]


@torch.no_grad()
def predict(model, split, device, amp_dtype, with_img, bs=512):
    model.eval()
    out = []
    for s in range(0, len(split), bs):
        img, feats, _ = fetch(split, torch.arange(s, min(s + bs, len(split)), device=device), device, with_img)
        with autocast(device, amp_dtype):
            out.append(torch.softmax(model(img, feats).float(), 1))
    return torch.cat(out)


def save_outcomes(out_dir, name, probs, split, info):
    probs = probs.float().cpu()
    top5 = probs.topk(5, dim=1).indices
    pred, beam = top5[:, 0], torch.from_numpy(split.beam_np)
    loss_db = -split.pwr.cpu().gather(1, pred[:, None]).squeeze(1)
    succ = dict(top1=pred == beam, top3=(top5[:, :3] == beam[:, None]).any(1), db1=loss_db <= 1.0, db3=loss_db <= 3.0)
    succ = {k: v.numpy().astype(np.uint8) for k, v in succ.items()}
    np.savez_compressed(os.path.join(out_dir, f'{name}.npz'), top5=top5.numpy().astype(np.uint8),
                        prob=probs.numpy().astype(np.float16), **succ)
    info = dict(info, name=name, **{f'test_{k}': float(v.mean()) for k, v in succ.items()},
                test_acc_by_scenario={int(s): {k: float(v[split.scen == s].mean()) for k, v in succ.items()}
                                      for s in np.unique(split.scen)})
    tmp = os.path.join(out_dir, f'{name}.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(info, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, f'{name}.json'))
    return info


def run_info(run, meta, **extra):
    info = {k: run[k] for k in ('family', 'arch', 'size', 'train_frac', 'scen_train', 'seed', 'kw', 'epochs')}
    info.update(run=run['name'], test_signature=meta['test_signature'], **extra)
    return info


def train_neural(run, data, args, device, amp_dtype, meta):
    name, with_img = run['name'], run['family'] != 'mlp'
    torch.manual_seed(run['seed'])
    pretrained = args.pretrained == 'imagenet'
    model = build_neural(run, meta['n_beams'], data['train'].feats.shape[1], pretrained).to(device)
    idx = torch.from_numpy(training_subset(data['train'], run['train_frac'], run['scen_train'], run['seed'])).to(device)
    epochs = run['epochs'] if args.max_epochs is None else min(run['epochs'], args.max_epochs)
    bs = run['bs']
    steps = max(1, len(idx) // bs)
    total = epochs * steps
    warmup = max(1, min(steps, total // 10))
    lr = run['lr'] if pretrained or not with_img else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=run['wd'])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = (torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16) if hasattr(torch.amp, 'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16))
    info = run_info(run, meta, epochs=epochs, lr=lr, batch_size=bs, pretrained=args.pretrained if with_img else None,
                    n_train=len(idx), n_params=sum(p.numel() for p in model.parameters()))
    print(f"[{name}] {info['n_params']} params, {len(idx)} training samples, {epochs} epochs", flush=True)

    best_acc, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = idx[torch.randperm(len(idx), device=device)]
        loss_sum = torch.zeros((), device=device)
        for b in range(steps):
            img, feats, y = fetch(data['train'], perm[b * bs:(b + 1) * bs], device, with_img)
            with autocast(device, amp_dtype):
                logits = model(img, feats)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach() * len(y)
        val = data['val']
        val_acc = (predict(model, val, device, amp_dtype, with_img).argmax(1) == val.beam).float().mean().item()
        if val_acc > best_acc:
            best_acc, best_epoch, best_state = val_acc, ep, copy.deepcopy(model.state_dict())
        print(f'[{name}] epoch {ep}/{epochs}  loss {loss_sum.item() / (steps * bs):.4f}  val top-1 {val_acc:.4f}  '
              f'({time.time() - t0:.0f}s)', flush=True)
        if ep in run['snapshots'] and ep < epochs:
            save_outcomes(args.out, f'{name}_ep{ep}', predict(model, data['test'], device, amp_dtype, with_img),
                          data['test'], dict(info, is_snapshot=True, epoch=ep, val_acc=val_acc))

    model.load_state_dict(best_state)
    if not args.no_ckpt:
        torch.save(best_state, os.path.join(args.out, 'ckpt', f'{name}.pt'))
    return save_outcomes(args.out, name, predict(model, data['test'], device, amp_dtype, with_img), data['test'],
                         dict(info, is_snapshot=False, epoch=best_epoch, val_acc=best_acc,
                              train_time_s=time.time() - t0))


def train_classic(run, data, args, meta):
    tr, val, test = data['train'], data['val'], data['test']
    idx = training_subset(tr, run['train_frac'], run['scen_train'], run['seed'])
    print(f"[{run['name']}] {len(idx)} training samples", flush=True)
    t0 = time.time()
    model = PerScenario(run['arch'], run['kw'], run['seed'], meta['n_beams'], n_jobs=args.cpu_jobs)
    import sklearn.ensemble
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.cpu_jobs):
        model.fit(tr.pos[idx], tr.scen[idx], tr.beam_np[idx])
        p_val, p_test = (torch.from_numpy(model.predict_proba(s.pos, s.scen)) for s in (val, test))
    val_acc = float((p_val.argmax(1).numpy() == val.beam_np).mean())
    info = run_info(run, meta, n_train=len(idx), is_snapshot=False, epoch=None, val_acc=val_acc)
    return save_outcomes(args.out, run['name'], p_test, test, dict(info, train_time_s=time.time() - t0))


def download_weights(runs):
    import torchvision
    for arch in sorted({r['arch'] for r in runs if r['family'] in ('vision', 'fusion')}):
        weights = torchvision.models.get_model_weights(arch).DEFAULT
        torchvision.models.get_model(arch, weights=weights)
        print(f'{arch:20s} {weights.url}')
    print(f'weights are cached in {os.path.join(torch.hub.get_dir(), "checkpoints")}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/deepsense', help='output directory of deepsense_data.py')
    ap.add_argument('--out', default='outputs_ds/zoo')
    ap.add_argument('--runs', default=None, help="subset of run indices, e.g. '0-9,15'")
    ap.add_argument('--list', action='store_true', help='print the zoo and exit')
    ap.add_argument('--download_weights', action='store_true', help='fetch the ImageNet weights of the zoo and exit')
    ap.add_argument('--pretrained', default='imagenet', choices=['imagenet', 'none'],
                    help='initialization of the CNNs (none: random, e.g. for tests without internet access)')
    ap.add_argument('--device', default=None, help='default: cuda if available')
    ap.add_argument('--data_device', default='auto', choices=['auto', 'cpu', 'gpu'],
                    help='where to keep the images (auto: GPU if they take < 50%% of the free memory)')
    ap.add_argument('--max_epochs', type=int, default=None, help='cap on the epochs of every run (smoke tests)')
    ap.add_argument('--amp', default='auto', choices=['auto', 'bf16', 'fp16', 'none'])
    ap.add_argument('--cpu_jobs', type=int, default=8, help='threads of the classical models')
    ap.add_argument('--no_ckpt', action='store_true', help='do not save model weights')
    args = ap.parse_args()

    runs = build_zoo()
    done = lambda r: os.path.exists(os.path.join(args.out, f"{r['name']}.json"))
    if args.list:
        for i, r in enumerate(runs):
            print(f"{i:3d}  {r['name']:45s} epochs={r['epochs'] or '-':>2}  {'done' if done(r) else ''}")
        print(f"{sum(map(done, runs))}/{len(runs)} runs done")
        return
    if args.download_weights:
        download_weights(runs)
        return
    selected = parse_runs(args.runs, len(runs))
    todo = [r for i, r in enumerate(runs) if i in selected and not done(r)]
    if not todo:
        print('nothing to do')
        return

    os.makedirs(os.path.join(args.out, 'ckpt'), exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    with open(os.path.join(args.data, 'meta.json')) as f:
        meta = json.load(f)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    amp_dtype = resolve_amp(device, args.amp)
    img_device = None
    if any(r['family'] in ('vision', 'fusion') for r in todo):
        img_device = device if args.data_device == 'gpu' else torch.device('cpu')
        if args.data_device == 'auto' and device.type == 'cuda':
            size = sum(os.path.getsize(os.path.join(args.data, f'{s}_img.npy')) for s in ('train', 'val', 'test'))
            img_device = device if size < 0.5 * torch.cuda.mem_get_info(device)[0] else torch.device('cpu')
    data = {s: Split(args.data, s, device, meta['scenarios'], img_device) for s in ('train', 'val', 'test')}
    print(f"device {device}, amp {amp_dtype}, images on {img_device}; train/val/test = "
          f"{len(data['train'])}/{len(data['val'])}/{len(data['test'])}; {len(todo)} runs to do", flush=True)

    for run in todo:
        lock = os.path.join(args.out, f"{run['name']}.lock")
        if done(run) or not claim(lock):
            continue
        try:
            if run['family'] == 'classic':
                info = train_classic(run, data, args, meta)
            else:
                info = train_neural(run, data, args, device, amp_dtype, meta)
            print(f"[{run['name']}] finished: test top-1 {info['test_top1']:.4f}, 1 dB {info['test_db1']:.4f}, "
                  f"top-3 {info['test_top3']:.4f}", flush=True)
        except Exception:
            print(f"[{run['name']}] FAILED:\n{traceback.format_exc()}", flush=True)
        finally:
            os.remove(lock)


if __name__ == '__main__':
    main()
