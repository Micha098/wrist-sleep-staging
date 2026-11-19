"""
Pack raw Empatica skin temperature (1 Hz day CSVs) per 30-s epoch, in the row order of cache/raw_pack/index.parquet
(pack_raw.py), for the raw-signal network with RAWNET_INPUT=raw_temp. Runs on the cluster (conda `sleep`).

Per epoch: 30 one-second bins, each the mean of the temperature samples in that second (deg C); NaN where none.

Output: cache/raw_pack/temp.npy (float16 [epochs, 30])
    python pack_temp.py [--workers 16]
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pack_raw import EP_S, MAX_EP, OUT, P, epoch_utc


def read_day(spid, day):
    f = f'{P}/empatica/temperature/{spid}/empatica_temp_{spid}_{day}.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, usecols=['time', 'temp'])
    return d.time.to_numpy(np.float64), d.temp.to_numpy(np.float32)


def night(job):
    name, spid, date, ep_t, row = job
    n = len(ep_t)
    out = np.full((n, EP_S), np.nan, np.float32)
    ts, xs = [], []
    for k in (-1, 0, 1):  # day files are by date; a night can span three of them in UTC vs local time
        r = read_day(spid, (pd.Timestamp(date) + pd.Timedelta(days=k)).strftime('%Y-%m-%d'))
        if r is not None:
            keep = (r[0] >= ep_t[0]) & (r[0] < ep_t[-1] + EP_S)
            ts.append(r[0][keep])
            xs.append(r[1][keep])
    if ts:
        t, x = np.concatenate(ts), np.concatenate(xs)
        e = np.searchsorted(ep_t, t, 'right') - 1
        sec = np.floor(t - ep_t[np.clip(e, 0, None)]).astype(np.int64)
        ok = (e >= 0) & (sec >= 0) & (sec < EP_S) & np.isfinite(x)
        cell = e[ok] * EP_S + sec[ok]
        tot, cnt = np.zeros(n * EP_S), np.zeros(n * EP_S)
        np.add.at(tot, cell, x[ok])
        np.add.at(cnt, cell, 1)
        flat = out.reshape(-1)
        flat[cnt > 0] = tot[cnt > 0] / cnt[cnt > 0]
    mm = np.load(f'{OUT}/temp.npy', mmap_mode='r+')
    mm[row:row + n] = out.astype(np.float16)
    mm.flush()
    return name, float(np.isnan(out).mean())


def main():
    ap = argparse.ArgumentParser()
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
        jobs.append((name, g.SPID.iloc[0], g.date.iloc[0], ep_t, int(idx.loc[name, 'row'])))
    rows = int((idx.row + idx.n).max())
    np.lib.format.open_memmap(f'{OUT}/temp.npy', mode='w+', dtype=np.float16, shape=(rows, EP_S)).flush()
    print(f'{len(jobs)} nights, {rows:,} epochs', flush=True)
    miss = []
    with Pool(a.workers) as pool:
        for i, (name, m) in enumerate(pool.imap_unordered(night, jobs, chunksize=4)):
            miss.append(m)
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(jobs)} nights', flush=True)
    print(f'done; missing temperature seconds: median night {np.median(miss):.1%}, mean {np.mean(miss):.1%}')


if __name__ == '__main__':
    main()
