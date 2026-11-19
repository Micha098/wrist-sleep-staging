"""
Pack a slow raw Empatica signal per 30-s epoch, in the row order of cache/raw_pack/index.parquet (pack_raw.py), as
extra raw channels for the deep network (rawnet.py RAWNET_EXTRA). Runs on the cluster (conda `sleep`).

  temp : skin temperature, 1 Hz -> 30 bins per epoch, (deg C - 33) / 2, clipped to +-5
  eda  : electrodermal activity, 4 Hz -> 120 bins per epoch, log(1 + uS) robust-scaled per night (median / IQR,
         IQR at least 0.05), clipped to +-10
Each bin is the mean of the samples in it; NaN where none.

Output: cache/raw_pack/<signal>.npy (float16 [epochs, bins])
    python pack_slow.py --signal temp|eda [--workers 16]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pack_raw import EP_S, MAX_EP, OUT, P, epoch_utc

SIGNALS = {'temp': dict(folder='temperature', col='temp', hz=1), 'eda': dict(folder='eda', col='eda', hz=4)}


def read_day(sig, spid, day):
    s = SIGNALS[sig]
    f = f'{P}/empatica/{s["folder"]}/{spid}/empatica_{sig}_{spid}_{day}.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, usecols=['time', s['col']])
    return d.time.to_numpy(np.float64), d[s['col']].to_numpy(np.float32)


def transform(sig, x):
    if sig == 'temp':
        return np.clip((x - 33) / 2, -5, 5)
    x = np.log1p(np.clip(x, 0, None))
    fin = x[np.isfinite(x)]
    if len(fin) < 100:
        return np.full_like(x, np.nan)
    q1, q2, q3 = np.percentile(fin, [25, 50, 75])
    return np.clip((x - q2) / max(q3 - q1, 0.05), -10, 10)


def night(job):
    sig, name, spid, date, ep_t, row = job
    k_ep = EP_S * SIGNALS[sig]['hz']  # bins per epoch
    n = len(ep_t)
    out = np.full((n, k_ep), np.nan, np.float32)
    ts, xs = [], []
    for k in (-1, 0, 1):  # day files are by date; a night can span three of them in UTC vs local time
        r = read_day(sig, spid, (pd.Timestamp(date) + pd.Timedelta(days=k)).strftime('%Y-%m-%d'))
        if r is not None:
            keep = (r[0] >= ep_t[0]) & (r[0] < ep_t[-1] + EP_S)
            ts.append(r[0][keep])
            xs.append(r[1][keep])
    if ts:
        t, x = np.concatenate(ts), np.concatenate(xs)
        e = np.searchsorted(ep_t, t, 'right') - 1
        b = np.floor((t - ep_t[np.clip(e, 0, None)]) * SIGNALS[sig]['hz']).astype(np.int64)
        ok = (e >= 0) & (b >= 0) & (b < k_ep) & np.isfinite(x)
        cell = e[ok] * k_ep + b[ok]
        tot, cnt = np.zeros(n * k_ep), np.zeros(n * k_ep)
        np.add.at(tot, cell, x[ok])
        np.add.at(cnt, cell, 1)
        flat = out.reshape(-1)
        flat[cnt > 0] = tot[cnt > 0] / cnt[cnt > 0]
        out = transform(sig, out)
    mm = np.load(f'{OUT}/{sig}.npy', mmap_mode='r+')
    mm[row:row + n] = out.astype(np.float16)
    mm.flush()
    return name, float(np.isnan(out).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signal', choices=list(SIGNALS), required=True)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    E = pd.read_parquet(f'{P}/wrist/epochs.parquet', columns=['SPID', 'name', 'date', 'epoch', 't'])
    E = E.sort_values(['name', 'epoch'])
    rec = pd.read_csv(f'{P}/qc/dreem_recordings.csv').dropna(subset=['dest']).set_index('name')
    idx = pd.read_parquet(f'{OUT}/index.parquet').set_index('name')
    jobs = []
    for name, g in E.groupby('name', sort=True):
        ep_t = epoch_utc(g.t, rec.loc[name, 'tz_str'])[:MAX_EP]
        assert len(ep_t) == idx.loc[name, 'n'], name
        jobs.append((a.signal, name, g.SPID.iloc[0], g.date.iloc[0], ep_t, int(idx.loc[name, 'row'])))
    rows = int((idx.row + idx.n).max())
    shape = (rows, EP_S * SIGNALS[a.signal]['hz'])
    np.lib.format.open_memmap(f'{OUT}/{a.signal}.npy', mode='w+', dtype=np.float16, shape=shape).flush()
    print(f'{a.signal}: {len(jobs)} nights, {rows:,} epochs', flush=True)
    miss = []
    with Pool(a.workers) as pool:
        for i, (name, m) in enumerate(pool.imap_unordered(night, jobs, chunksize=4)):
            miss.append(m)
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(jobs)} nights', flush=True)
    print(f'done; missing {a.signal} bins: median night {np.median(miss):.1%}, mean {np.mean(miss):.1%}')


if __name__ == '__main__':
    main()
