"""Night-level sleep measures from a hypnogram of stage codes (0 = Wake, > 0 = a sleep stage), one code per 30-s epoch."""
import numpy as np


def night_measures(stages, names):
    """TST, SOL, WASO (min), number of awakenings and minutes / % of sleep per stage. Sleep = any non-wake stage;
    WASO and awakenings are counted between the first and the last sleep epoch; an awakening = >= 2 consecutive
    wake epochs (>= 1 min); SOL = from the start of the night to the first sleep epoch."""
    s = np.asarray(stages, float)
    sleep = np.where(np.isnan(s), np.nan, (s > 0).astype(float))
    idx = np.flatnonzero(sleep == 1)
    out = dict(TST_min=0.0, SOL_min=np.nan, WASO_min=np.nan, awakenings=np.nan)
    if len(idx):
        so, last = idx[0], idx[-1]
        wake = (sleep[so:last + 1] == 0).astype(int)
        starts = np.flatnonzero(np.diff(np.r_[0, wake]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[wake, 0]) == -1)
        out.update(TST_min=np.nansum(sleep) / 2, SOL_min=so / 2, WASO_min=wake.sum() / 2,
                   awakenings=int(((ends - starts + 1) >= 2).sum()))
    n_sleep = np.nansum(sleep)
    for k, nm in enumerate(names):
        if k == 0:
            continue
        m = (s == k).sum() / 2
        out[f'{nm}_min'] = m
        out[f'{nm}_pct_of_sleep'] = 100 * m * 2 / n_sleep if n_sleep else np.nan
    return out
