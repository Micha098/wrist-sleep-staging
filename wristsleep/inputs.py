"""
Inputs for the models, built exactly as in training (see training/README.md for the original scripts).

All times are Unix seconds (UTC). A night is scored on 30-s epochs from `start` (bedtime / lights off) to `end`
(out of bed), at most 16 h.

Minute-level signals (Empatica "aggregated per minute" digital biomarkers): each epoch takes the values of the clock
minute it starts in. Minutes with < 50 % wear are treated as missing.
Raw signals (BVP and accelerometer, 64 Hz): placed on a uniform 64 Hz grid by linear interpolation (grid points
> 1 s from a real sample are missing), averaged in pairs to 32 Hz and cut into 960-sample epochs.
"""
import numpy as np
import pandas as pd

EPOCH_S = 30
MAX_EPOCHS = 1920  # 16 h
FS_RAW, FS = 64, 32
NS = FS * EPOCH_S  # samples per epoch
WORN_PCT = 50
MINUTE_COLS = {'pulse_rate_bpm': 'hr', 'prv_rmssd_ms': 'prv', 'temperature_celsius': 'temp',
               'respiratory_rate_brpm': 'resp', 'activity_counts': 'act', 'step_counts': 'steps',
               'wearing_detection_percentage': 'wear'}
SIG = ['act', 'steps', 'hr', 'prv', 'temp', 'resp']
NET_INPUTS = SIG + ['hr_rel', 'temp_rel', 'pos_h']  # order used in training


def epoch_table(minutes, start, end):
    """One row per 30-s epoch with that epoch's minute values. `minutes`: DataFrame with `timestamp_unix`
    (seconds or milliseconds) and Empatica's per-minute columns, or None (then only epoch and time; the signals are
    missing)."""
    n = int(min((end - start) // EPOCH_S, MAX_EPOCHS))
    if n < 1:
        raise ValueError('night shorter than one epoch')
    t = start + EPOCH_S * np.arange(n)
    E = pd.DataFrame({'epoch': np.arange(n), 't': t})
    if minutes is None:
        E[SIG + ['wear']] = np.nan
        return E
    m = minutes.rename(columns=MINUTE_COLS).copy()
    ts = m.timestamp_unix.astype(float)
    m['minute'] = np.floor(np.where(ts > 1e11, ts / 1000, ts) / 60).astype(np.int64)
    m = m.drop_duplicates('minute').set_index('minute')
    for c in SIG + ['wear']:
        E[c] = m[c].reindex(np.floor(t / 60).astype(np.int64)).to_numpy(dtype=float) if c in m else np.nan
    return E


def network_minute_inputs(E, med, iqr):
    """[n_epochs, 19] minute inputs of the deep network: 9 scaled signals, 9 missing flags, worn flag."""
    E = E.copy()
    worn = (E.wear >= WORN_PCT).to_numpy()
    E.loc[~worn, SIG] = np.nan
    E['act'], E['steps'] = np.log1p(E.act), np.log1p(E.steps)
    E['hr_rel'] = E.hr - E.hr.median()
    E['temp_rel'] = E.temp - E.temp.median()
    E['pos_h'] = E.epoch / 120
    X = E[NET_INPUTS].to_numpy(np.float32)
    Z = np.clip((X - np.asarray(med)) / np.asarray(iqr), -5, 5)
    miss = np.isnan(Z).astype(np.float32)
    return np.concatenate([np.nan_to_num(Z, nan=0.0), miss, worn[:, None].astype(np.float32)], axis=1).astype(np.float32)


def to_grid(t, x, t0, t1):
    """Samples (t in s, x [n] or [n, c]) -> uniform 64 Hz grid on [t0, t1), averaged in pairs to 32 Hz. NaN = no data."""
    grid = t0 + np.arange(int((t1 - t0) * FS_RAW)) / FS_RAW
    x = np.asarray(x, np.float32).reshape(len(t), -1)
    out = np.full((x.shape[1], len(grid)), np.nan, np.float32)
    k = (t >= t0) & (t < t1)
    t, x = np.asarray(t, float)[k], x[k]
    if len(t) > FS_RAW:
        o = np.argsort(t)
        t, x = t[o], x[o]
        j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        far = np.minimum(np.abs(grid - t[j - 1]), np.abs(t[j] - grid)) > 1.0
        for c in range(x.shape[1]):
            out[c] = np.interp(grid, t, x[:, c])
            out[c, far] = np.nan
    out = out[:, :out.shape[1] // 2 * 2]
    return out.reshape(out.shape[0], -1, 2).mean(axis=2)


def raw_epochs(bvp, acc, start, n_epochs):
    """[n_epochs, 5, 960] raw inputs: BVP (robust-scaled per night), accelerometer x/y/z (g) and a missing flag.
    bvp: DataFrame (time, bvp); acc: DataFrame (time, x, y, z); either may be None."""
    t0, t1 = start - 3600, start + EPOCH_S * n_epochs + 60  # as training: from 1 h before the start
    out = np.full((n_epochs, 5, NS), np.nan, np.float32)
    starts = np.round((start + EPOCH_S * np.arange(n_epochs) - t0) * FS).astype(np.int64)
    idx = starts[:, None] + np.arange(NS)
    if bvp is not None and len(bvp):
        b = to_grid(bvp.time.to_numpy(float), bvp.bvp.to_numpy(), t0, t1)[0]
        fin = b[np.isfinite(b)]
        if len(fin) > 1000:
            q1, q2, q3 = np.percentile(fin, [25, 50, 75])
            b = np.clip((b - q2) / max(q3 - q1, 1e-6), -10, 10)
        out[:, 0] = b[np.clip(idx, 0, len(b) - 1)]
    if acc is not None and len(acc):
        a = np.clip(to_grid(acc.time.to_numpy(float), acc[['x', 'y', 'z']].to_numpy(), t0, t1), -4, 4)
        out[:, 1:4] = a[:, np.clip(idx, 0, a.shape[1] - 1)].transpose(1, 0, 2)
    out[:, 4] = np.isnan(out[:, :4]).any(axis=1)
    return np.nan_to_num(out, nan=0.0).astype(np.float16)


def lgbm_features(E):
    """The 39 LightGBM features per epoch (as training/baseline_lgbm.py `features`), for one night."""
    E = E.sort_values('epoch').reset_index(drop=True).copy()
    E.loc[E.wear < WORN_PCT, ['act', 'hr', 'prv', 'temp', 'resp', 'steps']] = np.nan
    F = E[['act', 'hr', 'prv', 'temp', 'resp', 'steps']].copy()
    F['temp_rel'] = E.temp - E.temp.median()
    F['hr_rel'] = E.hr - E.hr.median()
    F['pos_h'] = E.epoch / 120
    F['pos_frac'] = E.epoch / E.epoch.max()
    for s in ['act', 'hr', 'prv', 'temp']:
        for w in [10, 30, 62]:
            r = E[s].rolling(w, center=True, min_periods=1)
            F[f'{s}_mean{w}'], F[f'{s}_sd{w}'] = r.mean(), r.std()
        F[f'{s}_max30'] = E[s].rolling(30, center=True, min_periods=1).max()
    F['act_zero_run'] = (E.act == 0).groupby((E.act != 0).cumsum()).cumcount()
    return F
