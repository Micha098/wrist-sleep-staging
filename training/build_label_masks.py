"""
Which Dreem epochs to trust as labels. Two masks over wrist/epochs.parquet:

  mask_3h       : scored epochs in the first 3 h of the recording (Dreem scoring is most complete there)
  mask_quality  : scored epochs, any time of night, inside reliable 10-min stretches. A stretch is reliable when
                  <= 20 % of its epochs are unscored and <= 30 % of its sleep-scored epochs are noisy. All epochs of a
                  reliable stretch are kept, including wake, so the rule does not filter out awakenings.

Noisy sleep epoch (both F7-O1 and F8-O2 must be clean; a channel is noisy when any of the following holds):
  - SSP power filter, per child (SSP_analysis/dreem_processing/power_filter.py): kurtosis, max amplitude, variance or
    root total power above the lower channel's median + 3 x IQR (75th percentile - median), over the child's epochs
  - SSP power filter, global: the same feature above its 95th percentile over all epochs of that channel
  - log10 20-40 Hz power > 1.2 (broadband contact noise)
  - > 0.5 % of samples at |x| >= 500 uV
or, for the epoch, relative beta (mean of the two channels) in the top 10 % for its Dreem stage (muscle / contact
artefact).

EEG markers come from cache/eeg_quality.parquet (prep/extract_eeg_quality.py) and the power-filter features from
cache/eeg_power_filter_features.parquet (prep/extract_power_filter_features.py), both pulled from the cluster EEG
cache, joined on time (nearest within 15 s).

Output: wrist/label_masks.parquet (name, epoch, mask_3h, mask_quality, noisy, stretch_ok); WRIST_MASK_OUT overrides the
file (e.g. for a dry run).
"""
import os

import numpy as np
import pandas as pd

P = os.environ.get('SOP_PROJ', 'data')  # project data folder (see training/README.md)
OUT = os.environ.get('WRIST_MASK_OUT', f'{P}/wrist/label_masks.parquet')
STRETCH = 20  # epochs (10 min)
MAX_UNSCORED, MAX_NOISY = 0.2, 0.3
CH = ['F7-O1', 'F8-O2']
PF_FEAT = ['kurtosis', 'max_amplitude', 'variance', 'root_total_power']


def eeg_noisy(q):
    """Per EEG epoch: 1 when F7-O1 or F8-O2 is noisy (power filter, 20-40 Hz power or amplitude), else 0."""
    bad = {c: (q[f'noise_hf_{c}'] > 1.2).to_numpy() | q[f'noise_clip_{c}'].fillna(False).astype(bool).to_numpy()
           for c in CH}
    for _, idx in q.groupby('SPID').indices.items():  # power filter stage 1: per child
        d = q.iloc[idx]
        for f in PF_FEAT:
            cols = [f'{f}_{c}' for c in CH]
            med = d[cols].median().min()
            thr = med + 3 * (d[cols].quantile(0.75).min() - med)
            for c in CH:
                bad[c][idx] |= (d[f'{f}_{c}'] > thr).to_numpy()
    for c in CH:  # power filter stage 2: global
        for f in PF_FEAT:
            x = q[f'{f}_{c}']
            bad[c] |= (x > x.quantile(0.95)).to_numpy()
    return (bad[CH[0]] | bad[CH[1]]).astype(float)


def main():
    E = pd.read_parquet(f'{P}/wrist/epochs.parquet', columns=['SPID', 'name', 'epoch', 't', 'stage'])
    rec = pd.read_csv(f'{P}/qc/dreem_recordings.csv').dropna(subset=['dest']).set_index('name')
    q = pd.read_parquet(f'{P}/cache/eeg_quality.parquet', columns=['name', 'epoch', 'time_utc'] +
                        [f'{k}_{c}' for c in CH for k in ['relative_power_beta', 'noise_hf', 'noise_clip']])
    pf = pd.read_parquet(f'{P}/cache/eeg_power_filter_features.parquet',
                         columns=['name', 'epoch', 'SPID'] + [f'{f}_{c}' for c in CH for f in PF_FEAT])
    q = q.merge(pf, on=['name', 'epoch'], how='left', validate='1:1')
    q['beta'] = q[[f'relative_power_beta_{c}' for c in CH]].mean(axis=1)
    q['reject'] = eeg_noisy(q)
    q['t_utc'] = pd.to_datetime(q.time_utc, utc=True).astype('datetime64[ns, UTC]')
    q = q[['name', 't_utc', 'beta', 'reject']]
    by_night = {n: d.drop(columns='name').sort_values('t_utc') for n, d in q.groupby('name', sort=False)}
    del pf, q
    out = []
    for name, e in E.groupby('name', sort=False):
        e = e.sort_values('epoch').copy()
        e['t_utc'] = e.t.dt.tz_localize(rec.loc[name, 'tz_str'], ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
        qq = by_night.get(name)
        if qq is not None and len(qq) and e.t_utc.notna().any():
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
    E[['name', 'epoch', 'mask_3h', 'mask_quality', 'noisy', 'stretch_ok']].to_parquet(OUT, index=False)
    lab = {0: 'Wake', 1: 'REM', 2: 'N1', 3: 'N2', 4: 'N3'}
    print(f'EEG markers matched on {E.beta.notna().mean():.1%} of epochs')
    print('noisy sleep epochs by stage:', E[sleep].groupby(E.stage.map(lab)).noisy.mean().round(3).to_dict())
    for m in ['mask_3h', 'mask_quality']:
        s = E[E[m]]
        print(f'{m}: {len(s):,} labelled epochs, {s.name.nunique()} nights; '
              f'stages {s.stage.map(lab).value_counts(normalize=True).round(3).to_dict()}; '
              f'after 3 h: {(s.epoch >= 360).mean():.0%}')
    late = E.epoch >= 360
    print(f'reliable stretches: first 3 h {E[~late].stretch_ok.mean():.0%}, later {E[late].stretch_ok.mean():.0%}')


if __name__ == '__main__':
    main()
