"""
Wrist-only sleep model, step 4: LightGBM, the feature-based model the deep network is compared with, and Empatica's
own sleep detection (wake / sleep only).

  empatica : Empatica's own sleep detection (0 awake, 101 sleep, 102 short WASO, 300 long WASO; only 101 = sleep)
  lgbm     : LightGBM on each epoch's minute signals plus centred rolling summaries (5 / 15 / 31 min) of activity,
         HR, PRV and temperature, temperature relative to the night median, and position in the recording

Labels : Dreem, 5 classes: Wake, N1, N2, N3, REM (reported also as 4, N1 + N2 = Light), on the reliable 10-min
         stretches of build_label_masks.py (mask_quality). Inputs use the whole night.
Splits : 5 folds by family (siblings stay together); 20 % of training families for early stopping.
Scores : epoch level  - wake/sleep kappa and balanced accuracy, stage kappa, on reliable-stretch epochs
         night level  - TST, WASO (after the first sleep epoch), SOL (recording start to first sleep epoch),
                        number of awakenings (>= 2 consecutive wake epochs after sleep onset), whole nights
                        with >= 80 % scored epochs: mean bias, MAE, Pearson r and within-child r vs Dreem

Outputs: wrist/baseline_pred_quality_5st.parquet, baseline_scores_quality_5st.csv, baseline_nights_quality_5st.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

P = os.environ.get('SOP_PROJ', 'data')  # project data folder (see training/README.md)
CLASS = {0: 0, 2: 1, 3: 2, 4: 3, 1: 4}  # Dreem 0 W, 2 N1, 3 N2, 4 N3, 1 REM
NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
NAMES4 = ['Wake', 'Light', 'N3', 'REM']  # reporting view: N1 + N2 merged into Light (merge_5stage.py)
ST_TAG = '_5st'  # output-file suffix of the 5-stage models
SIG = ['act', 'hr', 'prv', 'temp']
WINDOWS = [10, 30, 62]  # in 30-s epochs: 5, 15, 31 min
LABELS = 'quality'  # reliable-stretch labels (build_label_masks.py mask_quality)


def features(E):
    E = E.sort_values(['name', 'epoch']).reset_index(drop=True)
    E.loc[E.wear < 50, SIG + ['resp', 'steps']] = np.nan
    g = E.groupby('name')
    F = E[['act', 'hr', 'prv', 'temp', 'resp', 'steps']].copy()
    F['temp_rel'] = E.temp - g.temp.transform('median')
    F['hr_rel'] = E.hr - g.hr.transform('median')
    F['pos_h'] = E.epoch / 120
    F['pos_frac'] = E.epoch / g.epoch.transform('max')
    for s in SIG:
        for w in WINDOWS:
            r = g[s].rolling(w, center=True, min_periods=1)
            F[f'{s}_mean{w}'] = r.mean().reset_index(level=0, drop=True)
            F[f'{s}_sd{w}'] = r.std().reset_index(level=0, drop=True)
        F[f'{s}_max{WINDOWS[1]}'] = g[s].rolling(WINDOWS[1], center=True, min_periods=1).max() \
            .reset_index(level=0, drop=True)
    F['act_zero_run'] = (E.act == 0).groupby((E.act != 0).cumsum()).cumcount()  # epochs since movement
    return E, F


def empatica_wake_sleep(E):
    """Empatica sleep detection: 0 awake, 101 sleep, 102 short WASO, 300 long WASO -> 0 wake / 1 sleep."""
    return np.where(E.emp_stage.isna() | (E.wear < 50), np.nan, np.where(E.emp_stage == 101, 1, 0))


def family_folds(E, k=5, seed=0):
    fam = E.family_SPID.fillna(E.SPID).drop_duplicates().sample(frac=1, random_state=seed).to_numpy()
    f = {x: i % k for i, x in enumerate(fam)}
    return E.family_SPID.fillna(E.SPID).map(f).to_numpy()


def night_measures(df, col):
    """df: one night's epochs (sorted) with stage column `col` (0 = Wake, NaN allowed). Sleep = class > 0."""
    s = df[col].to_numpy()
    sleep = np.where(np.isnan(s), np.nan, (s > 0).astype(float))
    idx = np.where(sleep == 1)[0]
    if len(idx) == 0:
        return dict(TST=0, SOL=np.nan, WASO=np.nan, n_awak=np.nan)
    so, last = idx[0], idx[-1]
    seg = sleep[so:last + 1]
    wake = (seg == 0).astype(int)
    starts = np.flatnonzero(np.diff(np.r_[0, wake]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[wake, 0]) == -1)
    n_awak = int(((ends - starts + 1) >= 2).sum())
    return dict(TST=np.nansum(sleep) / 2, SOL=so / 2, WASO=np.nansum(seg == 0) / 2, n_awak=n_awak)


def score(P_, models):
    """Epoch level on reliable-stretch epochs (P_ needs mask_quality); night level on whole nights."""
    rows = []
    s = P_[P_.mask_quality & P_.y.notna()]
    for m in models:
        k = s[m].notna()
        yt, yp = s.y[k].astype(int), s[m][k].astype(int)
        rows.append(dict(level='epoch', model=m, measure='wake/sleep kappa',
                         value=cohen_kappa_score(yt > 0, yp > 0)))
        rows.append(dict(level='epoch', model=m, measure='wake/sleep balanced acc',
                         value=balanced_accuracy_score(yt > 0, yp > 0)))
        rows.append(dict(level='epoch', model=m, measure='wake recall', value=((yp == 0) & (yt == 0)).sum() / (yt == 0).sum()))
        if m != 'empatica':
            rows.append(dict(level='epoch', model=m, measure='stage kappa', value=cohen_kappa_score(yt, yp)))
    good = P_.groupby('name').y.apply(lambda x: x.notna().mean()) >= 0.8
    nights = []
    for name, df in P_[P_.name.isin(good[good].index)].groupby('name'):
        r = dict(name=name, SPID=df.SPID.iloc[0])
        for m in ['y'] + models:
            for key, v in night_measures(df, m).items():
                r[f'{key}_{m}'] = v
        nights.append(r)
    N = pd.DataFrame(nights)
    for m in models:
        for key in ['TST', 'SOL', 'WASO', 'n_awak']:
            a, b = N[f'{key}_y'], N[f'{key}_{m}']
            k = a.notna() & b.notna()
            a, b, sp = a[k], b[k], N.SPID[k]
            wa, wb = a - a.groupby(sp).transform('mean'), b - b.groupby(sp).transform('mean')
            for meas, v in [('bias', (b - a).mean()), ('MAE', (b - a).abs().mean()), ('r', pearsonr(a, b)[0]),
                            ('within-child r', pearsonr(wa, wb)[0])]:
                rows.append(dict(level=f'night ({k.sum()} nights)', model=m, measure=f'{key} {meas}', value=v))
    return pd.DataFrame(rows), N


def main():
    import lightgbm as lgb  # lazy: rawnet.py imports this module on the cluster, which has no lightgbm
    E = pd.read_parquet(f'{P}/wrist/epochs.parquet')
    E, F = features(E)
    E['y'] = E.stage.map(CLASS)
    E = E.merge(pd.read_parquet(f'{P}/wrist/label_masks.parquet', columns=['name', 'epoch', 'mask_quality']),
                on=['name', 'epoch'], how='left', validate='1:1')
    E['mask_quality'] = E.mask_quality.fillna(False).astype(bool)
    E['fold'] = family_folds(E)
    E['empatica'] = empatica_wake_sleep(E)
    E['lgbm'] = np.nan
    proba = np.full((len(E), len(NAMES)), np.nan)
    for k in range(5):
        tr = (E.fold != k) & E.mask_quality
        te = (E.fold == k).to_numpy()
        fams = E.loc[tr, 'family_SPID'].fillna(E.loc[tr, 'SPID']).unique()
        val_f = set(pd.Series(fams).sample(frac=0.2, random_state=k))
        v = E.family_SPID.fillna(E.SPID).isin(val_f).to_numpy() & tr.to_numpy()
        fit = tr.to_numpy() & ~v
        m = lgb.LGBMClassifier(objective='multiclass', learning_rate=0.05, n_estimators=2000, num_leaves=63,
                               min_child_samples=200, subsample=0.5, subsample_freq=1, colsample_bytree=0.7,
                               class_weight='balanced', n_jobs=8, verbose=-1)
        m.fit(F[fit], E.y[fit], eval_set=[(F[v], E.y[v])], callbacks=[lgb.early_stopping(50, verbose=False)])
        proba[te] = m.predict_proba(F[te], num_iteration=m.best_iteration_)
        E.loc[te, 'lgbm'] = proba[te].argmax(1)
        print(f'fold {k}: {fit.sum():,} train epochs, best iteration {m.best_iteration_}', flush=True)
    E[[f'p_{c}' for c in NAMES]] = proba
    keep = ['SPID', 'family_SPID', 'name', 'date', 'epoch', 'stage', 'y', 'fold', 'mask_quality', 'empatica', 'lgbm'] + \
        [f'p_{c}' for c in NAMES]
    E[keep].to_parquet(f'{P}/wrist/baseline_pred_{LABELS}{ST_TAG}.parquet', index=False)
    S, N = score(E, ['empatica', 'lgbm'])
    S.to_csv(f'{P}/wrist/baseline_scores_{LABELS}{ST_TAG}.csv', index=False)
    N.to_csv(f'{P}/wrist/baseline_nights_{LABELS}{ST_TAG}.csv', index=False)
    print(S.pivot_table(index=['level', 'measure'], columns='model', values='value', sort=False).round(3).to_string())


if __name__ == '__main__':
    main()
