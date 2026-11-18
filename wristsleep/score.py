"""
Score Empatica nights with the released models.

  deep network (raw pulse + motion + minute signals) -> sleep stages per 30-s epoch (the recommended output)
  LightGBM (minute signals)                          -> the baseline model of the paper

Both models are trained on five stages (Wake, N1, N2, N3, REM). The four-stage output (Wake, Light, N3, REM) is the
same hypnogram with N1 and N2 merged into Light.

Python:
    from wristsleep.score import Scorer
    sc = Scorer()
    epochs, summary = sc.score_night(minutes, start, end, bvp=bvp, acc=acc)

Command line (one row per night in nights.csv: night_id, minutes_csv, bvp_csv, acc_csv, start, end):
    python -m wristsleep.score nights.csv --out results/
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .inputs import epoch_table, lgbm_features, network_minute_inputs, raw_epochs
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
    def __init__(self, weights_dir=WEIGHTS, device='cpu'):
        meta = json.load(open(Path(weights_dir) / 'rawnet.json'))  # classes + input scaling of the fit set
        self.names = meta['names']
        self.med, self.iqr = np.asarray(meta['med']), np.asarray(meta['iqr'])
        self.net = RawNet(c_min=19, n_cls=len(self.names))
        self.net.load_state_dict(torch.load(Path(weights_dir) / 'rawnet.pt', map_location='cpu', weights_only=True))
        self.net.eval().to(device)
        self.device = device
        self.lgbm = None
        f = Path(weights_dir) / 'lgbm.txt'
        if f.exists():
            import lightgbm as lgb
            self.lgbm = lgb.Booster(model_file=str(f))
            self.lgbm_features = json.load(open(Path(weights_dir) / 'lgbm.json'))['features']

    def score_night(self, minutes, start, end, bvp=None, acc=None):
        """minutes: Empatica per-minute DataFrame; bvp: DataFrame (time, bvp); acc: DataFrame (time, x, y, z) in g;
        start/end: bedtime and rise time (Unix seconds or time strings with an offset).
        Returns (epochs DataFrame, night summary dict)."""
        start, end = to_unix(start), to_unix(end)
        E = epoch_table(minutes, start, end)
        z = torch.from_numpy(network_minute_inputs(E, self.med, self.iqr))[None]
        r = torch.from_numpy(raw_epochs(bvp, acc, start, len(E))).float()[None]
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
        if self.lgbm is not None:
            F = lgbm_features(E)[self.lgbm_features]
            l5 = self.lgbm.predict(F).argmax(1)
            out['stage_lgbm'] = np.array(NAMES4)[MERGE[l5]]
            m = night_measures(l5, self.names)
            summary.update({f'lgbm_{k}': m[k] for k in ['TST_min', 'SOL_min', 'WASO_min', 'awakenings']})
        worn = (E.wear >= 50).mean()
        summary.update(epochs=len(E), worn_fraction=round(float(worn), 3),
                       raw_missing_fraction=round(float(r[0, :, 4].mean()), 3))
        if worn < 0.8:
            summary['warning'] = 'wristband worn < 80% of the night: the models were validated on worn nights only'
        return out, summary


def _read(path):
    return None if not isinstance(path, str) or not path or not os.path.exists(path) else pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description='Score Empatica nights (see README).')
    ap.add_argument('nights', help='CSV: night_id, minutes_csv, bvp_csv, acc_csv, start, end')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sc = Scorer()
    rows = []
    for n in pd.read_csv(a.nights).itertuples():
        ep, s = sc.score_night(pd.read_csv(n.minutes_csv), n.start, n.end, bvp=_read(n.bvp_csv), acc=_read(n.acc_csv))
        ep.to_csv(f'{a.out}/{n.night_id}_epochs.csv', index=False)
        rows.append(dict(night_id=n.night_id, **s))
        print(f'{n.night_id}: TST {s["net_TST_min"]:.0f} min, SOL {s["net_SOL_min"]}, '
              f'WASO {s["net_WASO_min"]} min', flush=True)
    pd.DataFrame(rows).to_csv(f'{a.out}/night_summary.csv', index=False)


if __name__ == '__main__':
    main()
