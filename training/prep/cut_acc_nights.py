"""
Cut raw Empatica ACC (64 Hz day CSVs, x/y/z in g) to Dreem-night windows, on a uniform 32 Hz grid
(as cut_bvp_nights.py; 64 Hz samples interpolated to a 64 Hz grid, then averaged in pairs).

Window per Dreem recording: start_utc - 1 h  ->  start_utc + min(duration_h, 16) h.
Samples are placed on the grid by linear interpolation; grid points more than 1 s from a real sample are NaN.
Nights: the worn Dreem nights listed in cache/wrist_nights.csv (SPID, name, date).

Output: cache/acc_nights/<SPID>/<name>.npz  (acc float16 [3, n], t0 unix s, fs, n_raw)
Run on the login node, single process, low priority:
    nice -n 19 python cut_acc_nights.py [--part i/k] [SPID ...]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.csv as pv

P = os.environ.get('SOP_PROJ', 'data')  # project data folder (see training/README.md)
FS_IN, FS = 64, 32
PRE_H, MAX_H = 1, 16


def read_day(spid, day):
    f = f'{P}/empatica/acc/{spid}/empatica_acc_{spid}_{day}.csv'
    if not os.path.exists(f):
        return None
    t = pv.read_csv(f, convert_options=pv.ConvertOptions(include_columns=['time', 'x', 'y', 'z']))
    return t.column('time').to_numpy(), np.stack([t.column(c).to_numpy() for c in 'xyz']).astype(np.float32)


def main():
    rec = pd.read_csv(f'{P}/qc/dreem_recordings.csv').dropna(subset=['dest']).set_index('name')
    nights = pd.read_csv(f'{P}/cache/wrist_nights.csv')
    args = sys.argv[1:]
    if args[:1] == ['--part']:
        i, k = map(int, args[1].split('/'))
        spids = sorted(nights.SPID.unique())[i::k]
        nights, args = nights[nights.SPID.isin(spids)], args[2:]
    if args:
        nights = nights[nights.SPID.isin(args)]
    t_start = time.time()
    done = 0
    for spid, g in nights.groupby('SPID'):
        cache = {}
        for r in g.itertuples():
            out = f'{P}/cache/acc_nights/{spid}/{r.name.replace(".edf", ".npz")}'
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
                    ts.append(t[k]); xs.append(x[:, k])
            t = np.concatenate(ts) if ts else np.array([])
            x = np.concatenate(xs, axis=1) if xs else np.zeros((3, 0), np.float32)
            n = int((t1 - t0) * FS)
            grid = t0 + np.arange(2 * n) / FS_IN
            acc = np.full((3, 2 * n), np.nan, np.float32)
            if len(t) > FS_IN:
                o = np.argsort(t)
                t, x = t[o], x[:, o]
                j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
                near = np.minimum(np.abs(grid - t[j - 1]), np.abs(t[j] - grid))
                for c in range(3):
                    acc[c] = np.interp(grid, t, x[c])
                acc[:, near > 1.0] = np.nan
            acc = acc.reshape(3, n, 2).mean(axis=2).astype(np.float16)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez_compressed(out, acc=acc, t0=t0, fs=FS, n_raw=len(t))
            done += 1
            for k_ in [k_ for k_ in cache if k_ < (d - pd.Timedelta(days=1)).strftime('%Y-%m-%d')]:
                del cache[k_]
        print(f'{spid}: {len(g)} nights, total done {done}, {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
