import argparse
import contextlib
import copy
import json
import math
import os
import signal
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model

LR = {'transformer': 5e-4}


def build_zoo():
    base = [
        ('mlp', 128, 0.1, 10, dict(hidden=256)),
        ('mlp', 512, 0.5, 10, dict(hidden=512)),
        ('cnn', 256, 0.2, 10, dict(width=32)),
        ('cnn', 1024, 1.0, 12, dict(width=64, n_conv=3)),
        ('vgg', 512, 0.3, 15, {}),
        ('vgg', 1024, 1.0, 15, {}),
        ('resnet', 256, 0.3, 15, {}),
        ('resnet', 1024, 0.1, 15, dict(width=16)),
        ('resnet', 1024, 1.0, 15, {}),
        ('tcn', 512, 0.3, 12, {}),
        ('tcn', 1024, 1.0, 12, {}),
        ('inception', 512, 0.3, 12, {}),
        ('inception', 1024, 1.0, 12, {}),
        ('cldnn', 512, 0.5, 12, {}),
        ('cldnn', 1024, 1.0, 12, {}),
        ('lstm', 128, 0.5, 12, {}),
        ('lstm', 256, 1.0, 12, {}),
        ('gru', 512, 0.5, 12, {}),
        ('gru', 1024, 1.0, 12, {}),
        ('transformer', 256, 0.3, 15, dict(patch=8, d_model=64, n_layers=2)),
        ('transformer', 1024, 1.0, 15, dict(patch=16)),
    ]
    runs = [dict(arch=a, L=L, train_frac=f, epochs=e, seed=s, snr_min=None, model_kw=kw)
            for s in range(3) for a, L, f, e, kw in base]
    for s in range(2):
        runs += [dict(arch='cnn', L=1024, train_frac=1.0, epochs=12, seed=s, snr_min=0, model_kw=dict(width=64, n_conv=3)),
                 dict(arch='resnet', L=1024, train_frac=1.0, epochs=15, seed=s, snr_min=0, model_kw={}),
                 dict(arch='transformer', L=512, train_frac=1.0, epochs=15, seed=s, snr_min=0, model_kw=dict(patch=16))]
    for i, r in enumerate(runs):
        snr = '' if r['snr_min'] is None else f"_snr{r['snr_min']}"
        r['name'] = f"{i:03d}_{r['arch']}_L{r['L']}_f{r['train_frac']:g}{snr}_s{r['seed']}"
    return runs


def parse_runs(spec, n):
    if not spec:
        return set(range(n))
    out = set()
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


class Split:
    def __init__(self, data_dir, name, device):
        self.X = torch.from_numpy(np.load(os.path.join(data_dir, f'{name}_X.npy'))).to(device)
        self.y = torch.from_numpy(np.load(os.path.join(data_dir, f'{name}_y.npy'))).long().to(device)
        self.snr = np.load(os.path.join(data_dir, f'{name}_snr.npy'))

    def __len__(self):
        return len(self.y)


def training_subset(split, frac, snr_min, seed):
    y, snr = split.y.cpu().numpy(), split.snr
    cell = y * 1000 + (snr - snr.min())
    idx = np.flatnonzero(snr >= snr_min) if snr_min is not None else np.arange(len(y))
    if frac < 1:
        c = cell[idx]
        order = np.lexsort((np.random.default_rng(seed).random(len(idx)), c))
        c = c[order]
        rank = np.arange(len(c)) - np.searchsorted(c, c)
        _, inverse, counts = np.unique(c, return_inverse=True, return_counts=True)
        idx = np.sort(idx[order[rank < np.ceil(frac * counts[inverse])]])
    return torch.from_numpy(idx).to(split.X.device)


def resolve_amp(device, choice):
    if device.type != 'cuda' or choice == 'none':
        return None
    if choice == 'bf16' or (choice == 'auto' and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def autocast(device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def predict(model, split, L, device, amp_dtype, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(split), bs):
        x = split.X[s:s + bs, :, :L].to(device, non_blocking=True).float()
        with autocast(device, amp_dtype):
            out.append(torch.softmax(model(x).float(), dim=1))
    return torch.cat(out)


def save_outcomes(out_dir, name, probs, split, info):
    pred = probs.argmax(1)
    correct = (pred == split.y.to(pred.device)).cpu().numpy().astype(np.uint8)
    np.savez_compressed(os.path.join(out_dir, f'{name}.npz'), correct=correct,
                        pred=pred.cpu().numpy().astype(np.int16), prob=probs.cpu().numpy().astype(np.float16))
    info = dict(info, name=name, test_acc=float(correct.mean()),
                test_acc_by_snr={int(s): float(correct[split.snr == s].mean()) for s in np.unique(split.snr)})
    tmp = os.path.join(out_dir, f'{name}.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(info, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, f'{name}.json'))
    return info


def train_run(run, data, args, device, amp_dtype, meta):
    name, L = run['name'], run['L']
    torch.manual_seed(run['seed'])
    model = build_model(run['arch'], L, n_classes=len(meta['classes']), **run['model_kw']).to(device)
    idx = training_subset(data['train'], run['train_frac'], run['snr_min'], run['seed'])
    epochs = run['epochs'] if args.max_epochs is None else min(run['epochs'], args.max_epochs)
    bs = args.batch_size
    steps = math.ceil(len(idx) / bs)
    total = epochs * steps
    warmup = max(1, min(steps, total // 10))
    lr = LR.get(run['arch'], 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    info = dict(run=name, arch=run['arch'], L=L, train_frac=run['train_frac'], snr_min=run['snr_min'],
                seed=run['seed'], model_kw=run['model_kw'], lr=lr, batch_size=bs, epochs=epochs,
                n_train=len(idx), n_params=sum(p.numel() for p in model.parameters()),
                test_signature=meta['test_signature'])
    print(f"[{name}] {info['n_params']} params, {len(idx)} training frames, {epochs} epochs", flush=True)

    best_acc, best_epoch, best_state = -1.0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = idx[torch.randperm(len(idx), device=idx.device)]
        loss_sum = torch.zeros((), device=device)
        n_correct = torch.zeros((), device=device)
        for b in range(steps):
            j = perm[b * bs:(b + 1) * bs]
            x = data['train'].X[j, :, :L].to(device, non_blocking=True).float()
            y = data['train'].y[j].to(device, non_blocking=True)
            with autocast(device, amp_dtype):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach() * len(j)
            n_correct += (logits.argmax(1) == y).sum()
        val_acc = (predict(model, data['val'], L, device, amp_dtype).argmax(1) == data['val'].y).float().mean().item()
        if val_acc > best_acc:
            best_acc, best_epoch, best_state = val_acc, ep, copy.deepcopy(model.state_dict())
        print(f'[{name}] epoch {ep}/{epochs}  loss {loss_sum.item() / len(idx):.4f}  '
              f'train {n_correct.item() / len(idx):.4f}  val {val_acc:.4f}  ({time.time() - t0:.0f}s)', flush=True)
        if ep in args.snapshot_epochs and ep < epochs:
            save_outcomes(args.out, f'{name}_ep{ep}', predict(model, data['test'], L, device, amp_dtype),
                          data['test'], dict(info, is_snapshot=True, epoch=ep, val_acc=val_acc))

    model.load_state_dict(best_state)
    if not args.no_ckpt:
        torch.save(best_state, os.path.join(args.out, 'ckpt', f'{name}.pt'))
    return save_outcomes(args.out, name, predict(model, data['test'], L, device, amp_dtype), data['test'],
                         dict(info, is_snapshot=False, epoch=best_epoch, val_acc=best_acc,
                              train_time_s=time.time() - t0))


def claim(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w') as f:
        f.write(f'{socket.gethostname()} pid={os.getpid()} {time.ctime()}\n')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/radioml2018', help='output directory of radioml_data.py')
    ap.add_argument('--out', default='outputs/zoo')
    ap.add_argument('--runs', default=None, help="subset of run indices, e.g. '0-9,15'")
    ap.add_argument('--list', action='store_true', help='print the zoo and exit')
    ap.add_argument('--device', default=None, help='default: cuda if available')
    ap.add_argument('--data_device', default='auto', choices=['auto', 'cpu', 'gpu'],
                    help='where to keep the training pool (auto: GPU if it takes < 50%% of free memory)')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--snapshot_epochs', type=int, nargs='*', default=[1, 3],
                    help='also record the outcomes of the model after these epochs')
    ap.add_argument('--max_epochs', type=int, default=None, help='cap on the epochs of every run (smoke tests)')
    ap.add_argument('--amp', default='auto', choices=['auto', 'bf16', 'fp16', 'none'])
    ap.add_argument('--no_ckpt', action='store_true', help='do not save model weights')
    args = ap.parse_args()

    runs = build_zoo()
    done = lambda r: os.path.exists(os.path.join(args.out, f"{r['name']}.json"))
    selected = parse_runs(args.runs, len(runs))
    if args.list:
        for i, r in enumerate(runs):
            print(f"{i:3d}  {r['name']:40s} epochs={r['epochs']:2d}  {'done' if done(r) else ''}")
        print(f"{sum(map(done, runs))}/{len(runs)} runs done")
        return
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
    amp_dtype = resolve_amp(device, args.amp)
    on_gpu = args.data_device == 'gpu'
    if args.data_device == 'auto' and device.type == 'cuda':
        on_gpu = os.path.getsize(os.path.join(args.data, 'train_X.npy')) < 0.5 * torch.cuda.mem_get_info(device)[0]
    data = {'train': Split(args.data, 'train', device if on_gpu else 'cpu'),
            'val': Split(args.data, 'val', device), 'test': Split(args.data, 'test', device)}
    print(f"device {device}, amp {amp_dtype}, training pool on {'GPU' if on_gpu else 'CPU'}; "
          f"train/val/test = {len(data['train'])}/{len(data['val'])}/{len(data['test'])}; {len(todo)} runs to do")

    for run in todo:
        lock = os.path.join(args.out, f"{run['name']}.lock")
        if done(run) or not claim(lock):
            continue
        try:
            info = train_run(run, data, args, device, amp_dtype, meta)
            print(f"[{run['name']}] finished: test acc {info['test_acc']:.4f} (best epoch {info['epoch']})", flush=True)
        finally:
            os.remove(lock)


if __name__ == '__main__':
    main()
