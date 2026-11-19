"""
Pack raw BVP + ACC per 30-s epoch for the raw-signal network (wrist/rawnet.py). Runs on the cluster (conda `sleep`).

Per night (wrist/epochs.parquet, first 16 h = 1920 epochs as wrist/minute_inputs.py), epoch k covers [t_k, t_k + 30 s) on a
32 Hz grid, from the cut nights (cache/bvp_nights: 64 Hz, averaged in pairs; cache/acc_nights: 32 Hz):
  ch 0    BVP, robust-scaled per night (median / IQR of its finite samples), clipped to +-10
  ch 1-3  ACC x, y, z in g, clipped to +-4
  ch 4    missing (BVP or ACC NaN, or outside the cut window)
NaN -> 0.

Output: cache/raw_pack/raw.npy (float16 [epochs, 5, 960], memmap) + cache/raw_pack/index.parquet (name, row, n)
    nice -n 19 python pack_raw.py [--workers 4]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

P = os.environ.get('SOP_PROJ', 'data')  # project data folder (see training/README.md)
OUT = f'{P}/cache/raw_pack'
FS, EP_S, MAX_EP = 32, 30, 1920
NS = FS * EP_S


def epoch_utc(t, tz):
    u = t.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    s = (u - pd.Timestamp(0, tz='UTC')).dt.total_seconds()  # unit-safe (datetime64[us] vs [ns])
    return s.interpolate(limit_direction='both').to_numpy()


def windows(x, t0, ep_t):
    """x: [c, n] on the FS grid from t0 -> [n_ep, c, NS], NaN outside."""
    start = np.round((ep_t - t0) * FS).astype(np.int64)
    idx = start[:, None] + np.arange(NS)
    ok = (idx >= 0) & (idx < x.shape[1])
    w = x[:, np.clip(idx, 0, x.shape[1] - 1)].transpose(1, 0, 2)
    w[~np.broadcast_to(ok[:, None], w.shape)] = np.nan
    return w


def night(job):
    name, spid, ep_t, row = job
    n = len(ep_t)
    stem = name.replace('.edf', '.npz')
    out = np.full((n, 5, NS), np.nan, np.float32)
    fb, fa = f'{P}/cache/bvp_nights/{spid}/{stem}', f'{P}/cache/acc_nights/{spid}/{stem}'
    if os.path.exists(fb):
        z = np.load(fb)
        b = z['bvp'].astype(np.float32)
        b = b[:len(b) // 2 * 2].reshape(-1, 2).mean(1)
        fin = b[np.isfinite(b)]
        if len(fin) > 1000:
            q1, q2, q3 = np.percentile(fin, [25, 50, 75])
            b = np.clip((b - q2) / max(q3 - q1, 1e-6), -10, 10)
        out[:, 0] = windows(b[None], float(z['t0']), ep_t)[:, 0]
    if os.path.exists(fa):
        z = np.load(fa)
        out[:, 1:4] = np.clip(windows(z['acc'].astype(np.float32), float(z['t0']), ep_t), -4, 4)
    out[:, 4] = np.isnan(out[:, :4]).any(axis=1)
    out = np.nan_to_num(out, nan=0.0).astype(np.float16)
    mm = np.load(f'{OUT}/raw.npy', mmap_mode='r+')
    mm[row:row + n] = out
    mm.flush()
    return name, float(out[:, 4].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    E = pd.read_parquet(f'{P}/wrist/epochs.parquet', columns=['SPID', 'name', 'epoch', 't'])
    E = E.sort_values(['name', 'epoch'])
    rec = pd.read_csv(f'{P}/qc/dreem_recordings.csv').dropna(subset=['dest']).set_index('name')
    jobs, rows, row = [], [], 0
    for name, g in E.groupby('name', sort=True):
        ep_t = epoch_utc(g.t, rec.loc[name, 'tz_str'])[:MAX_EP]
        jobs.append((name, g.SPID.iloc[0], ep_t, row))
        rows.append(dict(name=name, row=row, n=len(ep_t)))
        row += len(ep_t)
    os.makedirs(OUT, exist_ok=True)
    np.lib.format.open_memmap(f'{OUT}/raw.npy', mode='w+', dtype=np.float16, shape=(row, 5, NS)).flush()
    pd.DataFrame(rows).to_parquet(f'{OUT}/index.parquet', index=False, compression=None)
    print(f'{len(jobs)} nights, {row:,} epochs, {row * 5 * NS * 2 / 1e9:.1f} GB', flush=True)
    miss = []
    with Pool(a.workers) as pool:
        for i, (name, m) in enumerate(pool.imap_unordered(night, jobs, chunksize=4)):
            miss.append(m)
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(jobs)} nights', flush=True)
    print(f'done; missing raw samples: median night {np.median(miss):.1%}, mean {np.mean(miss):.1%}')


if __name__ == '__main__':
    main()
