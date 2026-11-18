"""
Which Dreem epochs to trust as labels. Two masks over wrist/epochs.parquet:

  mask_3h       : scored epochs in the first 3 h of the recording (Dreem scoring is most complete there)
  mask_quality  : scored epochs, any time of night, inside reliable 10-min stretches. A stretch is reliable when
                  <= 20 % of its epochs are unscored and <= 30 % of its sleep-scored epochs are noisy, where noisy =
                  EEG noise flag (reject on F7-O1 or F8-O2: clipping, SD > 150 uV or log 20-40 Hz power > 1.2) or
                  relative beta in the top 10 % for that Dreem stage. All epochs of a reliable stretch are kept,
                  including wake, so the rule does not filter out awakenings.

EEG markers come from cache/eeg_quality.parquet (pulled from the cluster EEG cache by prep/extract_eeg_quality.py),
joined on time (nearest within 15 s).

Output: wrist/label_masks.parquet (name, epoch, mask_3h, mask_quality, noisy, stretch_ok)
"""
import os

import numpy as np
import pandas as pd

P = os.environ.get('SOP_PROJ', 'data')  # project data folder (see training/README.md)
STRETCH = 20  # epochs (10 min)
MAX_UNSCORED, MAX_NOISY = 0.2, 0.3


def main():
    E = pd.read_parquet(f'{P}/wrist/epochs.parquet', columns=['SPID', 'name', 'epoch', 't', 'stage'])
    rec = pd.read_csv(f'{P}/qc/dreem_recordings.csv').dropna(subset=['dest']).set_index('name')
    q = pd.read_parquet(f'{P}/cache/eeg_quality.parquet')
    q['beta'] = q[['relative_power_beta_F7-O1', 'relative_power_beta_F8-O2']].mean(axis=1)
    q['reject'] = q[['reject_F7-O1', 'reject_F8-O2']].max(axis=1)
    q['t_utc'] = pd.to_datetime(q.time_utc, utc=True).astype('datetime64[ns, UTC]')
    out = []
    for name, e in E.groupby('name', sort=False):
        e = e.sort_values('epoch').copy()
        e['t_utc'] = e.t.dt.tz_localize(rec.loc[name, 'tz_str'], ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
        qq = q[q.name == name][['t_utc', 'beta', 'reject']].sort_values('t_utc')
        if len(qq) and e.t_utc.notna().any():
            m = pd.merge_asof(e[['epoch', 't_utc']].dropna().sort_values('t_utc'), qq, on='t_utc',
                              direction='nearest', tolerance=pd.Timedelta('15s'))
            e = e.merge(m[['epoch', 'beta', 'reject']], on='epoch', how='left')
        else:
            e['beta'], e['reject'] = np.nan, np.nan
        out.append(e)
    E = pd.concat(out, ignore_index=True)
    sleep = E.stage.isin([1, 2, 3, 4])
    hi = E[sleep].groupby('stage').beta.transform(lambda x: x >= x.quantile(0.9))
    E['noisy'] = np.nan
    E.loc[sleep, 'noisy'] = ((E.loc[sleep, 'reject'] == 1) | hi.fillna(False)).astype(float)
    E['stretch'] = E.epoch // STRETCH
    g = E.groupby(['name', 'stretch'])
    unsc = g.stage.transform(lambda x: x.isna().mean())
    noisy = g.noisy.transform('mean').fillna(0)
    no_eeg = g.beta.transform(lambda x: x.isna().mean()) > 0.5  # no EEG markers -> cannot vouch for it
    E['stretch_ok'] = (unsc <= MAX_UNSCORED) & (noisy <= MAX_NOISY) & ~no_eeg
    E['mask_3h'] = E.stage.notna() & (E.epoch < 360)
    E['mask_quality'] = E.stage.notna() & E.stretch_ok
    E[['name', 'epoch', 'mask_3h', 'mask_quality', 'noisy', 'stretch_ok']].to_parquet(f'{P}/wrist/label_masks.parquet',
                                                                                       index=False)
    lab = {0: 'Wake', 1: 'REM', 2: 'N1', 3: 'N2', 4: 'N3'}
    print(f'EEG markers matched on {E.beta.notna().mean():.1%} of epochs')
    for m in ['mask_3h', 'mask_quality']:
        s = E[E[m]]
        print(f'{m}: {len(s):,} labelled epochs, {s.name.nunique()} nights; '
              f'stages {s.stage.map(lab).value_counts(normalize=True).round(3).to_dict()}; '
              f'after 3 h: {(s.epoch >= 360).mean():.0%}')
    late = E.epoch >= 360
    print(f'reliable stretches: first 3 h {E[~late].stretch_ok.mean():.0%}, later {E[late].stretch_ok.mean():.0%}')


if __name__ == '__main__':
    main()
