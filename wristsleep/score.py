"""
Score Empatica nights with the released models.

  deep network (raw pulse + motion + time of night) -> sleep stages per 30-s epoch (the recommended output)
  LightGBM (Empatica per-minute signals)            -> the baseline model of the paper (only when the per-minute
                                                       file is given)

Both models are trained on five stages (Wake, N1, N2, N3, REM). The four-stage output (Wake, Light, N3, REM) is the
same hypnogram with N1 and N2 merged into Light.

Python:
    from wristsleep.score import Scorer
    sc = Scorer()
    epochs, summary = sc.score_night(start, end, bvp=bvp, acc=acc, minutes=None)

Command line (one row per night in nights.csv: night_id, bvp_csv, acc_csv, start, end, and optionally minutes_csv):
    python -m wristsleep.score nights.csv --out results/

Motion-only network (accelerometer + time of night; for devices without a pulse sensor, e.g. ActiGraph on the wrist):
    Scorer(model='acc').score_night(start, end, acc=acc)
    python -m wristsleep.score nights.csv --model acc --out results/   (bvp_csv may be left out)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .inputs import acc_epochs, epoch_table, lgbm_features, network_minute_inputs, raw_epochs
from .measures import night_measures
from .model import RawNet

WEIGHTS = Path(__file__).resolve().parent.parent / 'weights'
NAMES4 = ['Wake', 'Light', 'N3', 'REM']
MERGE = np.array([0, 1, 1, 2, 3])          # Wake, N1, N2, N3, REM -> Wake, Light, N3, REM


def to_unix(x):
    """Unix seconds from a number or a time string with a UTC offset (e.g. 2024-05-03T21:30:00-04:00)."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        raise ValueError(f'time {x!r} has no time zone / UTC offset')
    return ts.timestamp()


class Scorer:
    def __init__(self, weights_dir=WEIGHTS, device='cpu', model='raw'):
        """model: 'raw' (pulse + motion, weights/rawnet.*) or 'acc' (motion only, weights/rawnet_acc.*: for
        accelerometer-only devices)."""
        stem = 'rawnet' if model == 'raw' else f'rawnet_{model}'
        meta = json.load(open(Path(weights_dir) / f'{stem}.json'))  # classes + input scaling of the fit set
        self.names = meta['names']
        self.med, self.iqr = np.asarray(meta['med']), np.asarray(meta['iqr'])
        self.input = meta.get('input', 'both')
        self.net = RawNet(c_min=19 if self.input == 'both' else 1, n_cls=len(self.names),
                          c_raw=4 if self.input == 'acc' else 5)
        self.net.load_state_dict(torch.load(Path(weights_dir) / f'{stem}.pt', map_location='cpu', weights_only=True))
        self.net.eval().to(device)
        self.device = device
        self.lgbm = None
        f = Path(weights_dir) / 'lgbm.txt'
        if f.exists():
            import lightgbm as lgb
            self.lgbm = lgb.Booster(model_file=str(f))
            self.lgbm_features = json.load(open(Path(weights_dir) / 'lgbm.json'))['features']

    def score_night(self, start, end, bvp=None, acc=None, minutes=None):
        """start/end: bedtime and rise time (Unix seconds or time strings with an offset); bvp: DataFrame (time, bvp);
        acc: DataFrame (time, x, y, z) in g; minutes: Empatica per-minute DataFrame (optional; needed for LightGBM and
        for Empatica's wear detection). Returns (epochs DataFrame, night summary dict)."""
        if self.input == 'raw' and bvp is None and acc is None:
            raise ValueError('the deep network needs the raw pulse and/or accelerometer data')
        if self.input == 'acc' and acc is None:
            raise ValueError('the motion-only network needs the raw accelerometer data')
        start, end = to_unix(start), to_unix(end)
        E = epoch_table(minutes, start, end)
        z = network_minute_inputs(E, self.med, self.iqr)
        if self.input != 'both':
            z = z[:, [len(self.med) - 1]]  # scaled hours since the start of the night (last input)
        z = torch.from_numpy(z)[None]
        raw = acc_epochs(acc, start, len(E)) if self.input == 'acc' else raw_epochs(bvp, acc, start, len(E))
        r = torch.from_numpy(raw).float()[None]
        pad = torch.zeros(1, len(E), dtype=torch.bool)
        with torch.no_grad():
            p = torch.softmax(self.net(r.to(self.device), z.to(self.device), pad.to(self.device)), -1)[0].cpu().numpy()
        s5 = p.argmax(1)
        out = pd.DataFrame({'epoch': E.epoch, 'time': pd.to_datetime(E.t, unit='s', utc=True),
                            'stage': np.array(NAMES4)[MERGE[s5]], 'stage5': np.array(self.names)[s5]})
        for k, nm in enumerate(self.names):
            out[f'p_{nm}'] = p[:, k].round(4)
        out['wear_pct'] = E.wear
        summary = {f'net_{k}': v for k, v in night_measures(s5, self.names).items()}   # TST/SOL/WASO + N1..REM
        summary['net_Light_min'] = summary['net_N1_min'] + summary['net_N2_min']
        summary['net_Light_pct_of_sleep'] = summary['net_N1_pct_of_sleep'] + summary['net_N2_pct_of_sleep']
        if self.lgbm is not None and minutes is not None:
            F = lgbm_features(E)[self.lgbm_features]
            l5 = self.lgbm.predict(F).argmax(1)
            out['stage_lgbm'] = np.array(NAMES4)[MERGE[l5]]
            m = night_measures(l5, self.names)
            summary.update({f'lgbm_{k}': m[k] for k in ['TST_min', 'SOL_min', 'WASO_min', 'awakenings']})
        worn = (E.wear >= 50).mean() if minutes is not None else 1 - float(r[0, :, -1].mean())
        summary.update(epochs=len(E), worn_fraction=round(float(worn), 3),
                       raw_missing_fraction=round(float(r[0, :, -1].mean()), 3))
        if worn < 0.8:
            summary['warning'] = 'wristband worn < 80% of the night: the models were validated on worn nights only'
        return out, summary


def _read(path):
    return None if not isinstance(path, str) or not path or not os.path.exists(path) else pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description='Score Empatica nights (see README).')
    ap.add_argument('nights', help='CSV: night_id, bvp_csv, acc_csv, start, end [, minutes_csv]')
    ap.add_argument('--out', default='results')
    ap.add_argument('--model', default='raw', choices=['raw', 'acc'], help='raw: pulse + motion; acc: motion only')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sc = Scorer(model=a.model)
    rows = []
    for n in pd.read_csv(a.nights).itertuples():
        ep, s = sc.score_night(n.start, n.end, bvp=_read(getattr(n, 'bvp_csv', None)), acc=_read(n.acc_csv),
                               minutes=_read(getattr(n, 'minutes_csv', None)))
        ep.to_csv(f'{a.out}/{n.night_id}_epochs.csv', index=False)
        rows.append(dict(night_id=n.night_id, **s))
        print(f'{n.night_id}: TST {s["net_TST_min"]:.0f} min, SOL {s["net_SOL_min"]}, '
              f'WASO {s["net_WASO_min"]} min', flush=True)
    pd.DataFrame(rows).to_csv(f'{a.out}/night_summary.csv', index=False)


if __name__ == '__main__':
    main()
