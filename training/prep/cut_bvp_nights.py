"""
Cut raw Empatica BVP (64 Hz day CSVs) to Dreem-night windows, on a uniform 64 Hz grid.

Window per Dreem recording: start_utc - 1 h  ->  start_utc + min(duration_h, 16) h.
Samples are placed on the grid by linear interpolation; grid points more than 1 s from a real sample are NaN.
Nights: the worn Dreem nights listed in cache/wrist_nights.csv (SPID, name, date).

Output: cache/bvp_nights/<SPID>/<name>.npz  (bvp float32, t0 unix s, fs, n_raw)
Run on the login node, single process, low priority:
    nice -n 19 python cut_bvp_nights.py [SPID ...]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.csv as pv

P = os.environ.get('SOP_PROJ', 'data')  # project data folder (see training/README.md)
FS = 64
PRE_H, MAX_H = 1, 16


def read_day(spid, day):
    f = f'{P}/empatica/bvp/{spid}/empatica_bvp_{spid}_{day}.csv'
    if not os.path.exists(f):
        return None
    t = pv.read_csv(f, convert_options=pv.ConvertOptions(include_columns=['time', 'bvp']))
    return t.column('time').to_numpy(), t.column('bvp').to_numpy().astype(np.float32)


def main():
    rec = pd.read_csv(f'{P}/qc/dreem_recordings.csv').dropna(subset=['dest']).set_index('name')
    nights = pd.read_csv(f'{P}/cache/wrist_nights.csv')
    if len(sys.argv) > 1:
        nights = nights[nights.SPID.isin(sys.argv[1:])]
    t_start = time.time()
    done = 0
    for spid, g in nights.groupby('SPID'):
        cache = {}
        for r in g.itertuples():
            out = f'{P}/cache/bvp_nights/{spid}/{r.name.replace(".edf", ".npz")}'
            if os.path.exists(out):
                continue
            s = pd.Timestamp(rec.loc[r.name, 'start_utc']).timestamp()
            t0, t1 = s - PRE_H * 3600, s + min(float(rec.loc[r.name, 'duration_h']), MAX_H) * 3600
            d = pd.Timestamp(r.date)
            ts, xs = [], []
            for day in [(d - pd.Timedelta(days=1)).strftime('%Y-%m-%d'), d.strftime('%Y-%m-%d')]:
                if day not in cache:
                    cache[day] = read_day(spid, day)
                if cache[day] is not None:
                    t, x = cache[day]
                    k = (t >= t0) & (t < t1)
                    ts.append(t[k]); xs.append(x[k])
            t = np.concatenate(ts) if ts else np.array([])
            x = np.concatenate(xs) if xs else np.array([], np.float32)
            grid = t0 + np.arange(int((t1 - t0) * FS)) / FS
            bvp = np.full(len(grid), np.nan, np.float32)
            if len(t) > FS:
                o = np.argsort(t)
                t, x = t[o], x[o]
                bvp = np.interp(grid, t, x).astype(np.float32)
                j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
                near = np.minimum(np.abs(grid - t[j - 1]), np.abs(t[j] - grid))
                bvp[near > 1.0] = np.nan
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez_compressed(out, bvp=bvp, t0=t0, fs=FS, n_raw=len(t))
            done += 1
            for k_ in [k_ for k_ in cache if k_ < (d - pd.Timedelta(days=1)).strftime('%Y-%m-%d')]:
                del cache[k_]
        print(f'{spid}: {len(g)} nights, total done {done}, {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
