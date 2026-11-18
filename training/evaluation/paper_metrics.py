"""
Wrist-only sleep model, step 6: numbers for the validation paper / lab talk.

1. Splits      : per family fold, nights / children / families / labelled epochs in fit, validation (early
                 stopping) and test sets (the seqnet.py / rawnet.py split).
2. Stage F1    : per-stage precision, recall, F1, macro F1, accuracy, kappa on held-out test predictions for
                 empatica (wake/sleep only), lgbm_q, net_q, raw_both, on three epoch sets:
                   reliable - quality-mask epochs, whole night (the training-label definition)
                   first3h  - all scored epochs in the first 3 h
                   all      - all scored epochs, whole night
                 + row-normalised confusion matrices on the reliable set.
3. Groups / age: per-child 4-stage kappa, macro F1 and wake F1 on reliable epochs (children with >= 200), and
                 per-night WASO error (whole nights with >= 80 % scored), by group4 and against age;
                 OLS: metric ~ asd + adhd + age + male.

Output: wrist/paper/*.csv
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

from baseline_lgbm import LABEL_EPOCHS, NAMES, P, score

W = f'{P}/wrist'
OUT = f'{W}/paper'
MODELS = ['empatica', 'lgbm_q', 'net_q', 'raw_both']
CHILD = os.environ.get('WRIST_CHILDREN', f'{P}/children.csv')  # SPID, group4, age, male, asd, adhd


def splits():
    E = pd.read_parquet(f'{W}/epochs.parquet', columns=['SPID', 'name', 'family_SPID', 'epoch', 'stage'])
    M = pd.read_parquet(f'{W}/label_masks.parquet', columns=['name', 'epoch', 'mask_quality'])
    E = E.merge(M, on=['name', 'epoch'], how='left')
    E['lab'] = E.mask_quality.fillna(False).astype(bool) & E.stage.notna() & (E.epoch < 1920)
    nights = E.groupby('name', sort=True).agg(SPID=('SPID', 'first'), family=('family_SPID', 'first'),
                                              lab=('lab', 'sum'), epochs=('epoch', 'size')).reset_index()
    nights['family'] = nights.family.fillna(nights.SPID)
    folds = pd.read_parquet(f'{W}/baseline_pred_3h.parquet', columns=['name', 'fold']).drop_duplicates('name')
    nights = nights.merge(folds, on='name', how='left')
    rows = []
    for k in range(5):
        tr = nights[nights.fold != k]
        fams = tr.family.drop_duplicates().sample(frac=0.2, random_state=k)
        part = {'fit': tr[~tr.family.isin(fams)], 'validation': tr[tr.family.isin(fams)], 'test': nights[nights.fold == k]}
        for s, d in part.items():
            rows.append(dict(fold=k, set=s, nights=len(d), children=d.SPID.nunique(), families=d.family.nunique(),
                             labelled_epochs=int(d.lab.sum()), labelled_hours=d.lab.sum() / 120))
    return pd.DataFrame(rows), nights


def stage_metrics(A):
    sets = {'reliable': A.mask_quality.fillna(False).astype(bool) & A.y.notna(),
            'first3h': (A.epoch < LABEL_EPOCHS) & A.y.notna(),
            'all': A.y.notna()}
    rows, conf = [], []
    for sname, m in sets.items():
        s = A[m]
        for mod in MODELS:
            k = s[mod].notna()
            yt, yp = s.y[k].astype(int).to_numpy(), s[mod][k].astype(int).to_numpy()
            ws_t, ws_p = (yt > 0).astype(int), (yp > 0).astype(int)
            base = dict(set=sname, model=mod, epochs=len(yt))
            rows.append(dict(base, measure='wake/sleep kappa', value=cohen_kappa_score(ws_t, ws_p)))
            rows.append(dict(base, measure='wake/sleep accuracy', value=accuracy_score(ws_t, ws_p)))
            rows.append(dict(base, measure='Wake F1 (wake/sleep)', value=f1_score(ws_t, ws_p, pos_label=0)))
            rows.append(dict(base, measure='Sleep F1 (wake/sleep)', value=f1_score(ws_t, ws_p, pos_label=1)))
            if mod == 'empatica':
                continue
            pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2, 3], zero_division=0)
            for i, c in enumerate(NAMES):
                rows += [dict(base, measure=f'{c} precision', value=pr[i]), dict(base, measure=f'{c} recall', value=rc[i]),
                         dict(base, measure=f'{c} F1', value=f1[i]), dict(base, measure=f'{c} share', value=sup[i] / sup.sum())]
            rows += [dict(base, measure='macro F1', value=f1.mean()),
                     dict(base, measure='4-stage accuracy', value=accuracy_score(yt, yp)),
                     dict(base, measure='4-stage kappa', value=cohen_kappa_score(yt, yp))]
            if sname == 'reliable':
                cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
                cm = cm / cm.sum(1, keepdims=True)
                for i, a in enumerate(NAMES):
                    conf.append(dict(model=mod, dreem=a, **{f'pred_{b}': cm[i, j] for j, b in enumerate(NAMES)}))
    return pd.DataFrame(rows), pd.DataFrame(conf)


def child_metrics(A, C):
    s = A[A.mask_quality.fillna(False).astype(bool) & A.y.notna()]
    rows = []
    for spid, d in s.groupby('SPID'):
        if len(d) < 200:
            continue
        for mod in ['lgbm_q', 'net_q', 'raw_both']:
            k = d[mod].notna()
            yt, yp = d.y[k].astype(int), d[mod][k].astype(int)
            rows.append(dict(SPID=spid, model=mod, epochs=len(yt), kappa=cohen_kappa_score(yt, yp),
                             macro_f1=f1_score(yt, yp, labels=[0, 1, 2, 3], average='macro', zero_division=0),
                             wake_f1=f1_score(yt > 0, yp > 0, pos_label=False, zero_division=0)))
    return pd.DataFrame(rows).merge(C, on='SPID', how='left')


def night_errors(A, C):
    _, N = score(A, ['empatica', 'lgbm_q', 'raw_both'], max_epoch=10**6)
    rows = []
    for mod in ['empatica', 'lgbm_q', 'raw_both']:
        for key in ['WASO', 'n_awak', 'TST']:
            d = N[['name', 'SPID', f'{key}_y', f'{key}_{mod}']].dropna()
            rows.append(pd.DataFrame(dict(name=d.name, SPID=d.SPID, model=mod, measure=key,
                                          err=d[f'{key}_{mod}'] - d[f'{key}_y'])))
    E = pd.concat(rows)
    E['abs_err'] = E.err.abs()
    return E.merge(C, on='SPID', how='left')


def group_tests(df, metrics, by_model):
    rows = []
    for mod, d in df.groupby(by_model):
        d = d.dropna(subset=['group4', 'age'])
        for met in metrics:
            g = d.groupby('group4')[met].agg(['mean', 'std', 'count'])
            fit = smf.ols(f'{met} ~ asd + adhd + age + male', d).fit()
            r = dict(model=mod, metric=met, n=len(d))
            for grp, x in g.iterrows():
                r[f'{grp} mean'], r[f'{grp} sd'], r[f'{grp} n'] = x['mean'], x['std'], x['count']
            for v in ['asd', 'adhd', 'age']:
                r[f'b_{v}'], r[f'p_{v}'] = fit.params[v], fit.pvalues[v]
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    S, nights = splits()
    S.to_csv(f'{OUT}/splits_by_fold.csv', index=False)
    print(S.groupby('set')[['nights', 'children', 'families', 'labelled_hours']].mean().round(0))
    print('total', len(nights), 'nights', nights.SPID.nunique(), 'children', nights.family.nunique(), 'families')

    A = pd.read_parquet(f'{W}/all_models_pred.parquet')
    M, CM = stage_metrics(A)
    M.to_csv(f'{OUT}/stage_metrics.csv', index=False)
    CM.to_csv(f'{OUT}/confusion_reliable.csv', index=False)
    pd.set_option('display.width', 220)
    print(M.pivot_table(index=['set', 'measure'], columns='model', values='value', sort=False)[MODELS].round(3).to_string())
    print(CM.round(2).to_string(index=False))

    C = pd.read_csv(CHILD, usecols=['SPID', 'group4', 'age', 'male', 'asd', 'adhd'])
    K = child_metrics(A, C)
    K.to_csv(f'{OUT}/child_metrics.csv', index=False)
    G = group_tests(K, ['kappa', 'macro_f1', 'wake_f1'], 'model')
    NE = night_errors(A, C)
    NE.to_csv(f'{OUT}/night_errors.csv', index=False)
    ch = NE.groupby(['model', 'measure', 'SPID']).agg(err=('err', 'mean'), abs_err=('abs_err', 'mean')).reset_index()
    ch = ch.merge(C, on='SPID', how='left')
    ch['model'] = ch.model + ' ' + ch.measure
    G2 = group_tests(ch, ['err', 'abs_err'], 'model')
    G = pd.concat([G, G2])
    G.to_csv(f'{OUT}/group_age_tests.csv', index=False)
    cols = ['model', 'metric', 'n'] + [f'{g} mean' for g in ['TD', 'ADHD', 'ASD', 'ASD+ADHD']] + \
           ['b_asd', 'p_asd', 'b_adhd', 'p_adhd', 'b_age', 'p_age']
    print(G[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
