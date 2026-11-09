from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from sionna.bench.mimo_ofdm_chest.benchmarking import channel_estimation as ce

_original = ce.evaluate_channel_estimation_system
_settings = dict(first_batch=0, ebno_min=None)


def evaluate_per_slot(system, output_path, *, ebno_dbs=None, verbose=True):
    cfg = system.cfg
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ebno_dbs is None:
        ebno_dbs = torch.tensor(cfg.snr_db_range(), dtype=torch.float32)
    else:
        ebno_dbs = torch.as_tensor(ebno_dbs, dtype=torch.float32)
    if _settings['ebno_min'] is not None:
        ebno_dbs = ebno_dbs[ebno_dbs >= _settings['ebno_min'] - 1e-6]
    batches = np.arange(_settings['first_batch'], cfg.num_mc_batches)
    shape = (len(ebno_dbs), len(batches), cfg.batch_size)
    cw_err = np.zeros(shape, dtype=np.int16)
    num_cw = np.zeros(shape, dtype=np.int16)
    bit_err = np.zeros(shape, dtype=np.int64)
    num_bits = np.zeros(shape, dtype=np.int64)
    nmse_num = np.zeros(shape, dtype=np.float64)
    nmse_den = np.zeros(shape, dtype=np.float64)
    seconds = np.zeros(shape[:2], dtype=np.float64)
    for s, ebno_db in enumerate(ebno_dbs):
        for i, j in enumerate(batches):
            ce.set_random_seed(int(cfg.evaluation_seed) + int(j))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                b, b_hat, h_true, h_hat = system(cfg.batch_size, ebno_db)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds[s, i] = time.perf_counter() - t0
            be, cwe, nb, ncw = ce._bit_and_block_error_terms_per_example(b, b_hat)
            nn_, nd = ce._channel_nmse_terms_per_example(h_hat, h_true)
            bit_err[s, i], cw_err[s, i] = be.cpu().numpy(), cwe.cpu().numpy()
            num_bits[s, i], num_cw[s, i] = nb.cpu().numpy(), ncw.cpu().numpy()
            nmse_num[s, i], nmse_den[s, i] = nn_.detach().cpu().numpy(), nd.detach().cpu().numpy()
        if verbose:
            print(f'Eb/N0 {float(ebno_db):5.1f} dB: BLER={cw_err[s].sum() / num_cw[s].sum():.4e}, '
                  f'slot success={np.mean(cw_err[s] == 0):.4f}, NMSE={nmse_num[s].sum() / nmse_den[s].sum():.4e}, '
                  f'{seconds[s].mean():.3f} s per batch', flush=True)
    ebno = ebno_dbs.cpu().numpy()
    bler = cw_err.reshape(len(ebno), -1).sum(1) / num_cw.reshape(len(ebno), -1).sum(1)
    ber = bit_err.reshape(len(ebno), -1).sum(1) / num_bits.reshape(len(ebno), -1).sum(1)
    nmse = nmse_num.reshape(len(ebno), -1).sum(1) / nmse_den.reshape(len(ebno), -1).sum(1)
    with output_path.open('wb') as f:
        np.savez(f, ebno_dbs=ebno, nmse=nmse, ber=ber, bler=bler)
    np.savez_compressed(output_path.with_suffix('.slots.npz'), ebno_dbs=ebno, cw_err=cw_err, num_cw=num_cw,
                        bit_err=bit_err, num_bits=num_bits, nmse_num=nmse_num, nmse_den=nmse_den,
                        batch_seconds=seconds, batch_index=batches, config_fingerprint=cfg.config_fingerprint())
    return dict(ebno_dbs=ebno, nmse=nmse, ber=ber, bler=bler)


def install(first_batch=0, ebno_min=None):
    import sionna.bench.mimo_ofdm_chest.benchmarking as bm
    _settings.update(first_batch=int(first_batch), ebno_min=ebno_min)
    ce.evaluate_channel_estimation_system = evaluate_per_slot
    bm.evaluate_channel_estimation_system = evaluate_per_slot
