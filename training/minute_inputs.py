"""
Wrist-only sleep model: the per-minute Empatica inputs of the deep network (rawnet.py), one array row per night.

Input  : per 30-s epoch, the minute wristband signals (log activity, log steps, HR, PRV, temperature, respiration,
         HR and temperature relative to the night median, worn flag, missing flags, hours since recording start),
         scaled by median / IQR of the training nights, clipped to +-5, missing -> 0. First 16 h of the night.
Labels : Dreem, 5 classes (baseline_lgbm.CLASS); epochs outside the reliable stretches (mask_quality) are -1 (ignored by the loss).
"""
import numpy as np
import pandas as pd

from baseline_lgbm import CLASS, LABELS, P

MAX_T = 16 * 120  # epochs per night (16 h), as cache/raw_pack (prep/pack_raw.py)


def night_arrays():
    E = pd.read_parquet(f'{P}/wrist/epochs.parquet').sort_values(['name', 'epoch']).reset_index(drop=True)
    E = E.merge(pd.read_parquet(f'{P}/wrist/label_masks.parquet', columns=['name', 'epoch', f'mask_{LABELS}']),
                on=['name', 'epoch'], how='left', validate='1:1').rename(columns={f'mask_{LABELS}': 'use'})
    E['use'] = E.use.fillna(False).astype(bool)
    worn = E.wear >= 50
    sig = ['act', 'steps', 'hr', 'prv', 'temp', 'resp']
    E.loc[~worn, sig] = np.nan
    E['act'], E['steps'] = np.log1p(E.act), np.log1p(E.steps)
    g = E.groupby('name')
    E['hr_rel'] = E.hr - g.hr.transform('median')
    E['temp_rel'] = E.temp - g.temp.transform('median')
    E['pos_h'] = E.epoch / 120
    cont = sig + ['hr_rel', 'temp_rel', 'pos_h']
    names = E.name.unique()
    N, T = len(names), MAX_T
    X = np.full((N, T, len(cont)), np.nan, np.float32)
    W = np.zeros((N, T), np.float32)
    Y = np.full((N, T), -1, np.int64)
    L = np.zeros(N, np.int64)
    meta = []
    for i, (name, d) in enumerate(E.groupby('name', sort=False)):
        n = min(len(d), T)
        X[i, :n] = d[cont].to_numpy(np.float32)[:n]
        W[i, :n] = (d.wear >= 50).to_numpy(np.float32)[:n]
        y = d.stage.map(CLASS).to_numpy()[:n]
        lab = ~np.isnan(y) & d.use.to_numpy()[:n]
        Y[i, :n][lab] = y[lab].astype(np.int64)
        L[i] = n
        meta.append(dict(name=name, SPID=d.SPID.iloc[0], family=d.family_SPID.fillna(d.SPID).iloc[0], n=n))
    return X, W, Y, L, pd.DataFrame(meta), cont


def scale_stats(X, tr):
    """Per-input median and IQR (SD, then 1, when the IQR is ~0) over the nights `tr`."""
    flat = X[tr].reshape(-1, X.shape[2])
    med = np.nanmedian(flat, 0)
    iqr = np.nanpercentile(flat, 75, 0) - np.nanpercentile(flat, 25, 0)
    sd = np.nanstd(flat, 0)
    return med, np.where(iqr > 1e-6, iqr, np.where(sd > 1e-6, sd, 1.0))


def scale(X, W, tr):
    med, iqr = scale_stats(X, tr)
    Z = np.clip((X - med) / iqr, -5, 5)
    miss = np.isnan(Z).astype(np.float32)
    Z = np.nan_to_num(Z, nan=0.0)
    return np.concatenate([Z, miss, W[..., None]], axis=2).astype(np.float32)
