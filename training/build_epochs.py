"""
Wrist-only sleep model, step 1: one row per 30-s Dreem epoch on nights the wristband was worn.

Labels : Dreem hypnogram (dreem/hypno_dreem), 0 Wake, 1 REM, 2 N1, 3 N2, 4 N3; other codes -> NaN (unscored)
Inputs : Empatica minute cache (SOL_analysis/sleep_prediction/cache/empatica_minute.parquet, local wall clock):
         hr, prv, temp, resp, act, steps, wear, emp_stage (Empatica's own sleep detection, baseline only)
Nights : every Dreem recording with the wristband worn (wear >= 50 %) on >= 80 % of it
         (cache/wear_coverage_dreem_nights.csv). Each 30-s epoch takes the values of its clock minute.

Output : wrist/epochs.parquet
"""
import os
import pandas as pd

RAW_DREEM_CODES = {'SLEEP-S0': 0, 'SLEEP-REM': 1, 'SLEEP-S1': 2, 'SLEEP-S2': 3, 'SLEEP-S3': 4}

P = os.environ.get('SOP_PROJ', 'data')  # project data folder (see training/README.md)
EMP = os.environ.get('WRIST_MINUTES', f'{P}/cache/empatica_minute.parquet')  # Empatica per-minute data, all children
MIN_WORN = 0.8
WRIST = ['hr', 'prv', 'temp', 'resp', 'act', 'steps', 'wear', 'emp_stage']


def read_dreem_hypno(f, t0_local):
    """As prep/stage1_eeg.read_dreem_hypno: harmonized tab files, or Dreem's raw export (times of day only)."""
    with open(f) as fh:
        head = fh.readline()
    if not head.startswith('Dreem Hypnogram Export'):
        d = pd.read_csv(f, sep='\t')
        d['t'] = pd.to_datetime(d['time'], utc=True)
        return d
    with open(f) as fh:
        skip = next(i for i, line in enumerate(fh) if line.startswith('Sleep Stage'))
    d = pd.read_csv(f, sep='\t', skiprows=skip)
    d['Sleep Stage'] = d['Sleep Stage'].map(RAW_DREEM_CODES)
    tod = pd.to_timedelta(d['Time [hh:mm:ss]'])
    day = (tod.diff() < pd.Timedelta(0)).cumsum()
    d['t'] = (t0_local.normalize() + tod + pd.to_timedelta(day, unit='D')).dt.tz_convert('UTC')
    return d


def main():
    cov = pd.read_csv(f'{P}/cache/wear_coverage_dreem_nights.csv')
    nights = cov[cov.worn >= MIN_WORN]
    rec = pd.read_csv(f'{P}/qc/dreem_recordings.csv').dropna(subset=['dest']).set_index('name')
    fam = pd.read_parquet(f'{P}/cache/samples.parquet')[['SPID', 'family_SPID']].drop_duplicates('SPID')
    emp = pd.read_parquet(EMP)
    emp = {k: v.set_index('t')[WRIST].sort_index() for k, v in emp.groupby('SPID')}
    out = []
    for r in nights.itertuples():
        f = f'{P}/dreem/hypno_dreem/{r.SPID}/dreem_hypno_{r.SPID}_{r.date}.csv'
        if not os.path.exists(f):
            continue
        t0 = pd.Timestamp(rec.loc[r.name, 'start_local'])
        h = read_dreem_hypno(f, t0).dropna(subset=['t']).sort_values('t')
        if len(h) and abs(h['t'].iloc[0] - t0.tz_convert('UTC') + pd.Timedelta(days=1)) < pd.Timedelta('5min'):
            h['t'] += pd.Timedelta(days=1)  # known one-day-early Dreem timestamps (as stage1_eeg)
        h['t'] = h['t'].dt.tz_convert(rec.loc[r.name, 'tz_str']).dt.tz_localize(None)
        h['stage'] = h['Sleep Stage'].where(h['Sleep Stage'].isin([0, 1, 2, 3, 4]))
        h = h[h.t <= h.t.iloc[0] + pd.Timedelta(hours=16)]  # recordings > 16 h: keep the night
        x = emp[r.SPID]
        x = x[~x.index.duplicated()]
        w = x.reindex(h.t.dt.floor('min')).reset_index(drop=True)
        e = pd.concat([h[['t', 'stage']].reset_index(drop=True), w], axis=1)
        e.insert(0, 'date', r.date)
        e.insert(0, 'name', r.name)
        e.insert(0, 'SPID', r.SPID)
        e['epoch'] = range(len(e))
        out.append(e)
    E = pd.concat(out, ignore_index=True).merge(fam, on='SPID', how='left')
    E.to_parquet(f'{P}/wrist/epochs.parquet', index=False)
    print(f'{len(E):,} epochs, {E.name.nunique()} nights, {E.SPID.nunique()} children, '
          f'{E.family_SPID.nunique()} families; scored {E.stage.notna().mean():.1%}, worn {(E.wear >= 50).mean():.1%}')
    print(E.stage.value_counts(normalize=True).sort_index().round(3).to_dict())


if __name__ == '__main__':
    main()
