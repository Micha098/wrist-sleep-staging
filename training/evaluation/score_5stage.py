"""
Wrist-only sleep model: score the 5-stage models (Wake, N1, N2, N3, REM; WRIST_STAGES=5) against Dreem.

Models : lgbm (baseline_pred_quality_5st.parquet) and the raw-signal network (rawnet_pred/rawnet_pred_quality_5st_both_fold?.parquet)
Scores : per-stage precision / recall / F1, macro F1, kappa, row-% confusion matrices on reliable-label epochs and on all
         scored epochs; per-night minutes of N1 and N2 (and % of sleep) vs Dreem on nights with >= 80 % scored epochs
         (r, within-child r, bias), stage percentages on epochs scored by both.
Output : wrist/paper/5stage_epoch_metrics.csv, 5stage_confusion_<model>_<set>.csv, 5stage_night_agreement.csv
"""
import glob
import os

os.environ['WRIST_STAGES'] = '5'

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from baseline_lgbm import NAMES, P

W = f'{P}/wrist'
OUT = f'{W}/paper'
PC = [f'p_{c}' for c in NAMES]


def load():
    B = pd.read_parquet(f'{W}/baseline_pred_quality_5st.parquet',
                        columns=['SPID', 'name', 'epoch', 'y', 'mask_quality', 'lgbm'])
    fs = sorted(glob.glob(f'{W}/rawnet_pred/rawnet_pred_quality_5st_both_fold?.parquet'))
    assert len(fs) == 5, f'{len(fs)} rawnet folds'
    R = pd.concat([pd.read_parquet(f, columns=['name', 'epoch', 'rawnet']) for f in fs])
    D = B.merge(R, on=['name', 'epoch'], how='inner', validate='1:1').sort_values(['name', 'epoch'])
    return D.reset_index(drop=True)


def epoch_metrics(D):
    rows, conf = [], []
    for sname, m in [('reliable', D.mask_quality & D.y.notna()), ('all scored', D.y.notna())]:
        for mod in ['lgbm', 'rawnet']:
            yt, yp = D.y[m].astype(int), D[mod][m].astype(int)
            pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=range(5), zero_division=0)
            r = dict(set=sname, model=mod, epochs=len(yt), kappa=cohen_kappa_score(yt, yp), macro_F1=f1.mean(),
                     accuracy=(yt == yp).mean())
            for i, c in enumerate(NAMES):
                r[f'{c}_precision'], r[f'{c}_recall'], r[f'{c}_F1'], r[f'{c}_share'] = pr[i], rc[i], f1[i], sup[i] / sup.sum()
            rows.append(r)
            cm = confusion_matrix(yt, yp, labels=range(5))
            pd.DataFrame(cm / cm.sum(1, keepdims=True) * 100, index=[f'dreem_{c}' for c in NAMES],
                         columns=[f'model_{c}' for c in NAMES]).round(1) \
                .to_csv(f"{OUT}/5stage_confusion_{mod}_{sname.split()[0]}.csv")
    return pd.DataFrame(rows)


def night_agreement(D):
    good = D.groupby('name').y.apply(lambda x: x.notna().mean()) >= 0.8
    G = D[D.name.isin(good[good].index) & D.y.notna()]
    rows = []
    for mod in ['lgbm', 'rawnet']:
        per = []
        for name, g in G.groupby('name'):
            r = dict(name=name, SPID=g.SPID.iloc[0])
            for src, col in [('dreem', 'y'), ('model', mod)]:
                sl = g[col][g[col] > 0]
                for i, c in [(1, 'N1'), (2, 'N2'), (3, 'N3'), (4, 'REM')]:
                    r[f'{c}_min_{src}'] = (g[col] == i).sum() / 2
                    r[f'{c}_pct_{src}'] = (sl == i).mean() * 100 if len(sl) else np.nan
            per.append(r)
        N = pd.DataFrame(per)
        for c in ['N1', 'N2', 'N3', 'REM']:
            for unit in ['min', 'pct']:
                a, b = N[f'{c}_{unit}_dreem'], N[f'{c}_{unit}_model']
                k = a.notna() & b.notna()
                a, b, sp = a[k], b[k], N.SPID[k]
                wa, wb = a - a.groupby(sp).transform('mean'), b - b.groupby(sp).transform('mean')
                rows.append(dict(model=mod, stage=c, unit='minutes' if unit == 'min' else '% of sleep', nights=k.sum(),
                                 dreem_mean=a.mean(), model_mean=b.mean(), bias=(b - a).mean(),
                                 r=pearsonr(a, b)[0], within_child_r=pearsonr(wa, wb)[0]))
    return pd.DataFrame(rows)


def main():
    D = load()
    print(f'{D.name.nunique()} nights, {len(D):,} epochs')
    E = epoch_metrics(D)
    E.to_csv(f'{OUT}/5stage_epoch_metrics.csv', index=False)
    A = night_agreement(D)
    A.to_csv(f'{OUT}/5stage_night_agreement.csv', index=False)
    pd.set_option('display.width', 220)
    cols = ['set', 'model', 'kappa', 'macro_F1', 'accuracy'] + [f'{c}_F1' for c in NAMES]
    print(E[cols].round(3).to_string(index=False))
    print(E[['set', 'model'] + [f'N1_{m}' for m in ['precision', 'recall']] + [f'N2_{m}' for m in ['precision', 'recall']]]
          .round(3).to_string(index=False))
    for mod in ['lgbm', 'rawnet']:
        print(mod, 'reliable confusion (row %)')
        print(pd.read_csv(f'{OUT}/5stage_confusion_{mod}_reliable.csv', index_col=0).round(0).to_string())
    print(A.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
