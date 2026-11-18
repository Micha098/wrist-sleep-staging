"""
Numbers for the paper (deep net = raw-signal network vs LightGBM, both trained on reliable-stretch labels, 4 stages).

Output: wrist/paper/paper_night_agreement.csv (per model and measure: means, bias, 95 % limits of agreement, MAE, r,
        ICC(A,1), within-child r, child-mean r), paper_per_night_kappa.csv, paper_label_stats.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from baseline_lgbm import P, night_measures
from export_scored_nights import MEASURES, icc_a1
from scipy.stats import pearsonr

W = f'{P}/wrist'
OUT = f'{W}/paper'
MODELS = {'lgbm_q': 'LightGBM', 'raw_both': 'Deep net'}


def nights(A):
    rows = []
    for name, g in A.groupby('name', sort=True):
        r = dict(name=name, SPID=g.SPID.iloc[0], pct_scored=g.y.notna().mean() * 100)
        both = g[g.y.notna()]
        for src in ['y'] + list(MODELS):
            for k, v in night_measures(g, src).items():
                r[f'{k}_{src}'] = v
            bm = both[both[src].notna()]
            sl = bm[src][bm[src] > 0]
            for i, c in [(1, 'Light'), (2, 'N3'), (3, 'REM')]:
                r[f'pct_{c}_{src}'] = (sl == i).mean() * 100 if len(sl) else np.nan
            if src != 'y' and len(bm):
                r[f'kappa_{src}'] = cohen_kappa_score(bm.y.astype(int), bm[src].astype(int), labels=[0, 1, 2, 3])
                r[f'kappa_ws_{src}'] = cohen_kappa_score(bm.y > 0, bm[src] > 0)
        rows.append(r)
    return pd.DataFrame(rows)


def agreement(N):
    G = N[N.pct_scored >= 80]
    rows = []
    for m, lab in MODELS.items():
        for key, label in MEASURES:
            d = G[['SPID', f'{key}_y', f'{key}_{m}']].dropna()
            a, b = d[f'{key}_y'].to_numpy(), d[f'{key}_{m}'].to_numpy()
            diff = b - a
            sp = d.SPID
            wa, wb = a - d.groupby('SPID')[f'{key}_y'].transform('mean'), b - d.groupby('SPID')[f'{key}_{m}'].transform('mean')
            cm = d.groupby('SPID')[[f'{key}_y', f'{key}_{m}']].mean()[d.groupby('SPID').size() >= 3]
            rows.append(dict(model=lab, measure=label, nights=len(d), dreem_mean=a.mean(), dreem_sd=a.std(),
                             model_mean=b.mean(), model_sd=b.std(), bias=diff.mean(),
                             loa_low=diff.mean() - 1.96 * diff.std(), loa_high=diff.mean() + 1.96 * diff.std(),
                             MAE=np.abs(diff).mean(), r=pearsonr(a, b)[0], ICC_A1=icc_a1(a, b),
                             within_child_r=pearsonr(wa, wb)[0], child_mean_r=pearsonr(cm.iloc[:, 0], cm.iloc[:, 1])[0],
                             children=sp.nunique(), children_child_r=len(cm)))
    return pd.DataFrame(rows)


def label_stats():
    M = pd.read_parquet(f'{W}/label_masks.parquet')
    E = pd.read_parquet(f'{W}/epochs.parquet', columns=['name', 'epoch', 'stage'])
    E = E.merge(M, on=['name', 'epoch'])
    late = E.epoch >= 360
    lab = {0: 'Wake', 1: 'REM', 2: 'N1', 3: 'N2', 4: 'N3'}
    r = dict(epochs=len(E), scored=E.stage.notna().sum(), unscored_first3h=E[~late].stage.isna().mean(),
             unscored_after3h=E[late].stage.isna().mean(), reliable_epochs=E.mask_quality.sum(),
             reliable_share_of_scored=E.mask_quality.sum() / E.stage.notna().sum(),
             stretch_ok_first3h=E[~late].stretch_ok.mean(), stretch_ok_after3h=E[late].stretch_ok.mean(),
             reliable_after3h=(E[E.mask_quality].epoch >= 360).mean(),
             noisy_sleep=E.noisy.mean(), mask3h_epochs=E.mask_3h.sum())
    for k, v in E[E.mask_quality].stage.map(lab).value_counts(normalize=True).items():
        r[f'reliable_share_{k}'] = v
    for k, v in E[E.stage.notna()].stage.map(lab).value_counts(normalize=True).items():
        r[f'scored_share_{k}'] = v
    return pd.Series(r)


def main():
    A = pd.read_parquet(f'{W}/all_models_pred.parquet', columns=['SPID', 'name', 'epoch', 'y'] + list(MODELS))
    A = A.sort_values(['name', 'epoch'])
    N = nights(A)
    S = agreement(N)
    S.to_csv(f'{OUT}/paper_night_agreement.csv', index=False)
    G = N[N.pct_scored >= 80]
    K = pd.DataFrame([dict(model=lab, measure=f'{w} kappa per night', median=G[f'{c}_{m}'].median(),
                           q25=G[f'{c}_{m}'].quantile(.25), q75=G[f'{c}_{m}'].quantile(.75), nights=G[f'{c}_{m}'].notna().sum())
                      for m, lab in MODELS.items() for c, w in [('kappa', '4-stage'), ('kappa_ws', 'wake/sleep')]])
    K.to_csv(f'{OUT}/paper_per_night_kappa.csv', index=False)
    L = label_stats()
    L.to_csv(f'{OUT}/paper_label_stats.csv', header=['value'])
    pd.set_option('display.width', 250)
    print(S.round(2).to_string(index=False))
    print(K.round(2).to_string(index=False))
    print(L.round(3).to_string())


if __name__ == '__main__':
    main()
