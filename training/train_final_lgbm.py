"""
Final LightGBM (quality labels) on all families, for release: 20 % of families for early stopping, no test set.
5 stages. Output: wrist/lgbm_final_quality_5st.txt (LightGBM model) + .json (feature list, classes).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline_lgbm import CLASS, NAMES, P, ST_TAG, features


def main():
    E = pd.read_parquet(f'{P}/wrist/epochs.parquet')
    E, F = features(E)
    E['y'] = E.stage.map(CLASS)
    E = E.merge(pd.read_parquet(f'{P}/wrist/label_masks.parquet', columns=['name', 'epoch', 'mask_quality']),
                on=['name', 'epoch'], how='left', validate='1:1')
    use = E.mask_quality.fillna(False).astype(bool).to_numpy() & E.y.notna().to_numpy()
    fam = E.family_SPID.fillna(E.SPID)
    val_f = set(pd.Series(fam[use].unique()).sample(frac=0.2, random_state=0))
    v = fam.isin(val_f).to_numpy() & use
    fit = use & ~v
    m = lgb.LGBMClassifier(objective='multiclass', learning_rate=0.05, n_estimators=2000, num_leaves=63,
                           min_child_samples=200, subsample=0.5, subsample_freq=1, colsample_bytree=0.7,
                           class_weight='balanced', n_jobs=8, verbose=-1)
    m.fit(F[fit], E.y[fit], eval_set=[(F[v], E.y[v])], callbacks=[lgb.early_stopping(50, verbose=False)])
    out = f'{P}/wrist/lgbm_final_quality{ST_TAG}'
    m.booster_.save_model(f'{out}.txt', num_iteration=m.best_iteration_)
    json.dump(dict(features=list(F.columns), classes=NAMES, best_iteration=int(m.best_iteration_),
                   n_fit_epochs=int(fit.sum()), n_val_epochs=int(v.sum())), open(f'{out}.json', 'w'), indent=1)
    print(f'saved {out}.txt: {fit.sum():,} fit epochs, best iteration {m.best_iteration_}')


if __name__ == '__main__':
    main()
