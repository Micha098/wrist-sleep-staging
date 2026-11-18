"""
Wrist-only sleep model, step 7: export the deep-net (raw-signal network) scoring of every paired night, and its
agreement with Dreem.

Predictions: rawnet both inputs, quality labels, held-out family folds (each night scored by a model that never saw
that family). Nights: the 1,977 Dreem + Empatica nights (wristband worn >= 80 %), first 16 h.

Output folder (OUT):
  nights/<SPID>/<night>.csv   per 30-s epoch: time (local), Dreem stage (5- and 4-class), model stage, model
                              probabilities, Empatica's own sleep code, reliable-label flag
  all_epochs.parquet          the same for all nights
  night_measures.csv          per night: Dreem and model TST, SOL, WASO, awakenings, % Light / N3 / REM of sleep,
                              epoch agreement (kappa, accuracy), % epochs scored by Dreem, group, age
  agreement_epochs.csv        epoch level: kappa, accuracy, per-stage precision / recall / F1 (all scored, reliable)
  confusion_*.csv             confusion matrices (counts, row %)
  agreement_measures.csv      night level per measure: means, bias, 95 % limits of agreement, MAE, r, ICC(A,1),
                              within-child r, child-mean r
  figures/*.png               confusion matrix, per-night kappa, scatter and Bland-Altman plots, example hypnograms
  README.md                   summary
Night measures as baseline_lgbm.night_measures (sleep = any non-wake stage; WASO and awakenings between the first
and last sleep epoch; awakenings = >= 2 consecutive wake epochs; SOL from the start of the recording); stage
percentages on epochs scored by both. Night-level agreement on nights with >= 80 % of epochs scored by Dreem.
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from baseline_lgbm import NAMES, P, night_measures

W = f'{P}/wrist'
OUT = os.environ.get('WRIST_EXPORT', 'scored_nights')
CHILD = os.environ.get('WRIST_CHILDREN', f'{P}/children.csv')  # SPID, group4, age, male, asd, adhd
DREEM5 = {0: 'Wake', 1: 'REM', 2: 'N1', 3: 'N2', 4: 'N3'}
NAMES5 = ['Wake', 'N1', 'N2', 'N3', 'REM']
PC = [f'p_{c}' for c in NAMES5]                 # 5-stage network probabilities
MEASURES = [('TST', 'Total sleep time (min)'), ('SOL', 'Sleep onset latency (min)'), ('WASO', 'WASO (min)'),
            ('n_awak', 'Awakenings (n)'), ('pct_Light', 'Light sleep (% of sleep)'),
            ('pct_N3', 'N3 (% of sleep)'), ('pct_REM', 'REM (% of sleep)')]
STAGE_COL = {'Wake': '#E8962E', 'Light': '#7FA7D9', 'N3': '#3D3B8E', 'REM': '#B5838D'}


def load():
    A = pd.read_parquet(f'{W}/eval_pred.parquet',
                        columns=['SPID', 'name', 'date', 'epoch', 'stage', 'y', 'empatica', 'mask_quality', 'raw_5m'])
    R = pd.concat([pd.read_parquet(f, columns=['name', 'epoch', 'rawnet'] + PC)
                   for f in sorted(glob.glob(f'{W}/rawnet_pred/rawnet_pred_quality_5st_both_fold?.parquet'))])
    T = pd.read_parquet(f'{W}/epochs.parquet', columns=['name', 'epoch', 't', 'emp_stage', 'wear'])
    D = A.merge(R, on=['name', 'epoch'], how='inner', validate='1:1').merge(T, on=['name', 'epoch'], how='left',
                                                                            validate='1:1')
    D = D.sort_values(['name', 'epoch']).reset_index(drop=True)
    D['dreem_stage5'] = D.stage.map(DREEM5)
    D['dreem_stage'] = D.y.map(dict(enumerate(NAMES)))
    D['model_stage'] = D.raw_5m.map(dict(enumerate(NAMES)))          # N1+N2 merged into Light
    D['model_stage5'] = D.rawnet.map(dict(enumerate(NAMES5)))
    D['reliable_label'] = D.mask_quality.fillna(False).astype(bool)
    return D


def export_nights(D):
    cols = ['epoch', 't', 'dreem_stage5', 'dreem_stage', 'model_stage5', 'model_stage'] + PC + ['emp_stage', 'wear', 'reliable_label']
    out = D[['SPID', 'name', 'date'] + cols].rename(columns={'t': 'time_local', 'emp_stage': 'empatica_sleep_code',
                                                          'wear': 'wear_pct'})
    out[PC] = out[PC].round(4)
    out.to_parquet(f'{OUT}/all_epochs.parquet', index=False)
    for (spid, name), g in out.groupby(['SPID', 'name']):
        os.makedirs(f'{OUT}/nights/{spid}', exist_ok=True)
        g.drop(columns=['SPID', 'name']).to_csv(f"{OUT}/nights/{spid}/{name.replace('.edf', '')}.csv", index=False)


def night_table(D):
    rows = []
    for name, g in D.groupby('name', sort=True):
        r = dict(name=name, SPID=g.SPID.iloc[0], date=g.date.iloc[0], start_local=g.t.iloc[0], epochs=len(g),
                 pct_scored=g.y.notna().mean() * 100)
        for src, col in [('dreem', 'y'), ('model', 'raw_5m')]:
            for k, v in night_measures(g, col).items():
                r[f'{k}_{src}'] = v
        both = g[g.y.notna()]
        for src, col in [('dreem', 'y'), ('model', 'raw_5m')]:
            sl = both[col][both[col] > 0]
            for i, c in [(1, 'Light'), (2, 'N3'), (3, 'REM')]:
                r[f'pct_{c}_{src}'] = (sl == i).mean() * 100 if len(sl) else np.nan
        if len(both) > 0:
            r['kappa'] = cohen_kappa_score(both.y.astype(int), both.raw_5m.astype(int), labels=[0, 1, 2, 3])
            r['accuracy'] = accuracy_score(both.y.astype(int), both.raw_5m.astype(int))
            r['kappa_wake_sleep'] = cohen_kappa_score(both.y > 0, both.raw_5m > 0)
        rows.append(r)
    N = pd.DataFrame(rows)
    C = pd.read_csv(CHILD, usecols=['SPID', 'group4', 'age', 'male'])
    return N.merge(C, on='SPID', how='left')


def icc_a1(a, b):
    """ICC(A,1): two-way, absolute agreement, single measure (McGraw & Wong)."""
    X = np.c_[a, b]
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def measure_agreement(N):
    G = N[N.pct_scored >= 80]
    rows = []
    for key, label in MEASURES:
        d = G[['SPID', f'{key}_dreem', f'{key}_model']].dropna()
        a, b = d[f'{key}_dreem'].to_numpy(), d[f'{key}_model'].to_numpy()
        diff = b - a
        wa = a - d.groupby('SPID')[f'{key}_dreem'].transform('mean')
        wb = b - d.groupby('SPID')[f'{key}_model'].transform('mean')
        cm = d.groupby('SPID')[[f'{key}_dreem', f'{key}_model']].mean()
        cm = cm[d.groupby('SPID').size() >= 3]
        rows.append(dict(measure=label, key=key, nights=len(d), children=d.SPID.nunique(),
                         dreem_mean=a.mean(), dreem_sd=a.std(), model_mean=b.mean(), model_sd=b.std(),
                         bias=diff.mean(), loa_low=diff.mean() - 1.96 * diff.std(),
                         loa_high=diff.mean() + 1.96 * diff.std(), MAE=np.abs(diff).mean(),
                         r=pearsonr(a, b)[0], ICC_A1=icc_a1(a, b), within_child_r=pearsonr(wa, wb)[0],
                         child_mean_r=pearsonr(cm.iloc[:, 0], cm.iloc[:, 1])[0], children_for_child_r=len(cm)))
    return pd.DataFrame(rows), G


def epoch_agreement(D):
    rows, cms = [], {}
    for sname, m in [('all scored epochs', D.y.notna()), ('reliable-label epochs', D.reliable_label & D.y.notna())]:
        yt, yp = D.y[m].astype(int), D.raw_5m[m].astype(int)
        pr, rc, f1, sup = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2, 3], zero_division=0)
        r = dict(epochs=len(yt), set=sname, kappa=cohen_kappa_score(yt, yp), accuracy=accuracy_score(yt, yp),
                 macro_F1=f1.mean(), wake_sleep_kappa=cohen_kappa_score(yt > 0, yp > 0))
        for i, c in enumerate(NAMES):
            r[f'{c}_precision'], r[f'{c}_recall'], r[f'{c}_F1'] = pr[i], rc[i], f1[i]
        rows.append(r)
        cms[sname] = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    return pd.DataFrame(rows), cms


def fig_confusion(cm, path, title):
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(pct, cmap='Purples', vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{pct[i, j]:.0f}%\n({cm[i, j]:,})', ha='center', va='center', fontsize=8.5,
                    color='white' if pct[i, j] > 55 else 'black')
    ax.set_xticks(range(4), NAMES)
    ax.set_yticks(range(4), NAMES)
    ax.set_xlabel('Deep net (wrist)')
    ax.set_ylabel('Dreem EEG')
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_kappa(N, path):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, lab in [(axs[0], 'kappa_wake_sleep', 'Wake/sleep kappa per night'),
                         (axs[1], 'kappa', '4-stage kappa per night')]:
        v = N[col].dropna()
        ax.hist(v, bins=40, color='#3D3B8E')
        ax.axvline(v.median(), color='#E8962E', lw=2)
        ax.set_title(f'{lab}\nmedian {v.median():.2f} (IQR {v.quantile(.25):.2f}–{v.quantile(.75):.2f}), '
                     f'{len(v):,} nights', fontsize=10)
        ax.set_xlim(-0.2, 1)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_scatter(G, S, path):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
    for ax, (key, label) in zip(axs.flat, MEASURES):
        d = G[[f'{key}_dreem', f'{key}_model']].dropna()
        a, b = d.iloc[:, 0], d.iloc[:, 1]
        ax.scatter(a, b, s=6, alpha=0.35, color='#3D3B8E', edgecolors='none')
        lo, hi = min(a.min(), b.min()), max(a.quantile(.995), b.quantile(.995))
        ax.plot([lo, hi], [lo, hi], color='#E8962E', lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        s = S.set_index('key').loc[key]
        ax.set_title(f'{label}\nr {s.r:.2f} · ICC {s.ICC_A1:.2f} · within-child r {s.within_child_r:.2f}',
                     fontsize=9.5)
        ax.set_xlabel('Dreem')
        ax.set_ylabel('Deep net')
        ax.spines[['top', 'right']].set_visible(False)
    axs.flat[-1].axis('off')
    fig.suptitle(f'Night measures, deep net vs Dreem ({len(G):,} nights with ≥ 80% scored)', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bland_altman(G, S, path):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
    for ax, (key, label) in zip(axs.flat, MEASURES):
        d = G[[f'{key}_dreem', f'{key}_model']].dropna()
        a, b = d.iloc[:, 0], d.iloc[:, 1]
        s = S.set_index('key').loc[key]
        ax.scatter((a + b) / 2, b - a, s=6, alpha=0.35, color='#3D3B8E', edgecolors='none')
        for v, ls in [(s.bias, '-'), (s.loa_low, '--'), (s.loa_high, '--')]:
            ax.axhline(v, color='#E8962E', ls=ls, lw=1.2)
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_title(f'{label}\nbias {s.bias:+.1f}, 95% LoA {s.loa_low:.1f} to {s.loa_high:.1f}', fontsize=9.5)
        ax.set_xlabel('Mean of Dreem and deep net')
        ax.set_ylabel('Deep net − Dreem')
        ax.spines[['top', 'right']].set_visible(False)
    axs.flat[-1].axis('off')
    fig.suptitle('Bland–Altman: deep net − Dreem', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_hypnograms(D, N, path):
    G = N[N.pct_scored >= 80].dropna(subset=['kappa']).sort_values('kappa').reset_index(drop=True)
    pick = [G.iloc[int(q * (len(G) - 1))] for q in (0.1, 0.5, 0.9)]
    level = {'Wake': 3, 'REM': 2, 'Light': 1, 'N3': 0}
    fig, axs = plt.subplots(len(pick), 1, figsize=(13, 2.6 * len(pick)))
    for ax, r, pc in zip(axs, pick, ['10th', '50th', '90th']):
        g = D[D.name == r['name']]
        h = (g.epoch / 120).to_numpy()
        for col, off, lab, c in [('dreem_stage', 0.12, 'Dreem', '#2B2D5C'), ('model_stage', -0.12, 'Deep net',
                                                                               '#E8962E')]:
            ax.step(h, g[col].map(level) + off, where='post', lw=1.1, color=c, label=lab)
        ax.set_yticks([0, 1, 2, 3], ['N3', 'Light', 'REM', 'Wake'])
        ax.set_xlim(0, h.max())
        ax.set_title(f"{r.SPID} · {r.date} · 4-stage kappa {r.kappa:.2f} (night at the "
                     f"{pc} percentile)", fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axs[-1].set_xlabel('Hours from recording start')
    axs[0].legend(loc='upper right', ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def readme(D, N, E, S, path):
    G = N[N.pct_scored >= 80]
    e_all, e_rel = E.iloc[0], E.iloc[1]
    L = [
        '# Wrist deep-net sleep scoring of the paired Dreem + Empatica nights', '',
        f'{N.name.nunique():,} nights from {N.SPID.nunique()} children, scored every 30 s by the raw-signal deep '
        'network (raw pulse + motion + Empatica minute signals), trained on 5 stages (Wake, N1, N2, N3, REM); its '
        '4-stage output is the same hypnogram with N1 and N2 merged into Light. Each night was scored by a model trained on other '
        'families only (5-fold cross-validation by family), so these are out-of-sample scores.', '',
        '## Files', '',
        '- `nights/<SPID>/<night>.csv`: one row per 30-s epoch: `time_local`, Dreem stage (`dreem_stage5`: Wake / '
        'N1 / N2 / N3 / REM; `dreem_stage`: 4 classes, empty = unscored by Dreem), `model_stage5` (Wake / N1 / N2 / N3 / REM), `model_stage` '
        '(4 classes, N1 + N2 = Light), 5-stage model probabilities `p_Wake, p_N1, p_N2, p_N3, p_REM`, Empatica\'s own sleep code (0 awake, 101 sleep, 102 short WASO, 300 long '
        'WASO), `wear_pct`, `reliable_label` (epoch in a reliable EEG stretch).',
        '- `all_epochs.parquet`: all nights in one table.',
        '- `night_measures.csv`: per night, Dreem and deep-net TST, SOL, WASO, awakenings, % Light / N3 / REM, '
        'per-night kappa and accuracy, % of epochs scored by Dreem, group and age.',
        '- `agreement_epochs.csv`, `confusion_*.csv`, `agreement_measures.csv`, `figures/`.', '',
        '## Epoch agreement (4 classes: Wake, Light = N1 + N2, N3, REM)', '',
        '| Epochs | n | Kappa | Accuracy | Macro F1 | Wake F1 | Light F1 | N3 F1 | REM F1 | Wake/sleep kappa |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for _, e in E.iterrows():
        L.append(f"| {e.set} | {e.epochs:,} | {e.kappa:.2f} | {e.accuracy:.0%} | {e.macro_F1:.2f} | {e.Wake_F1:.2f} "
                 f"| {e.Light_F1:.2f} | {e.N3_F1:.2f} | {e.REM_F1:.2f} | {e.wake_sleep_kappa:.2f} |")
    L += ['', f"Per night (nights ≥ 80% scored): 4-stage kappa median {G.kappa.median():.2f} "
              f"(IQR {G.kappa.quantile(.25):.2f}–{G.kappa.quantile(.75):.2f}); wake/sleep kappa median "
              f"{G.kappa_wake_sleep.median():.2f}.", '',
          '![confusion](figures/confusion_all_scored.png)', '', '![kappa](figures/per_night_kappa.png)', '',
          f'## Sleep measures ({len(G):,} nights with ≥ 80% of epochs scored by Dreem)', '',
          '| Measure | Dreem mean (SD) | Deep net mean (SD) | Bias | 95% limits of agreement | MAE | r | ICC(A,1) | '
          'Within-child r | Child-mean r |', '|---|---|---|---|---|---|---|---|---|---|']
    for _, s in S.iterrows():
        L.append(f'| {s.measure} | {s.dreem_mean:.1f} ({s.dreem_sd:.1f}) | {s.model_mean:.1f} ({s.model_sd:.1f}) | '
                 f'{s.bias:+.1f} | {s.loa_low:.1f} to {s.loa_high:.1f} | {s.MAE:.1f} | {s.r:.2f} | {s.ICC_A1:.2f} | '
                 f'{s.within_child_r:.2f} | {s.child_mean_r:.2f} |')
    L += ['', 'Bias = deep net − Dreem. ICC(A,1) = absolute agreement. Within-child r: both measures centred on each '
              "child's own mean (does the model track which nights were better or worse for that child?). Child-mean "
              'r: correlation of per-child averages (children with ≥ 3 nights), relevant for between-child and group '
              'comparisons.', '',
          '![scatter](figures/night_measures_scatter.png)', '', '![bland-altman](figures/night_measures_bland_altman.png)',
          '', '## Example nights (10th, 50th and 90th percentile of per-night kappa)', '',
          '![hypnograms](figures/example_hypnograms.png)', '',
          '## Notes', '',
          '- Definitions: sleep = any non-wake stage; SOL from the recording start to the first sleep epoch; WASO = '
          'wake between the first and last sleep epoch; awakenings = runs of ≥ 2 wake epochs (≥ 1 min) after sleep '
          'onset. Stage percentages are of sleep, on epochs scored by both.',
          '- Dreem leaves ~11% of epochs unscored early in the night and ~41% late, so Dreem itself is an imperfect '
          'reference late at night; that is why night measures use nights with ≥ 80% scored epochs.',
          '- The deep net smooths over short (1–2 min) awakenings: it under-counts WASO (about 9 min) and '
          'awakenings and over-counts TST. Night-to-night changes in WASO are tracked moderately (within-child r '
          f"{S.set_index('key').loc['WASO', 'within_child_r']:.2f}).",
          '- N1 is over-counted by the 5-stage model (see the paper); use N1 minutes as a relative index only.',
          '- Only the first 16 h of each recording are scored.', '',
          'Generated by `training/evaluation/export_scored_nights.py`.']
    open(path, 'w').write('\n'.join(L))


def main():
    os.makedirs(f'{OUT}/figures', exist_ok=True)
    D = load()
    print(f'{D.name.nunique()} nights, {len(D):,} epochs', flush=True)
    export_nights(D)
    N = night_table(D)
    N.to_csv(f'{OUT}/night_measures.csv', index=False)
    E, cms = epoch_agreement(D)
    E.to_csv(f'{OUT}/agreement_epochs.csv', index=False)
    for k, cm in cms.items():
        tag = k.split()[0]
        pd.DataFrame(cm, index=[f'dreem_{c}' for c in NAMES], columns=[f'model_{c}' for c in NAMES]) \
            .to_csv(f'{OUT}/confusion_{tag}_counts.csv')
        pd.DataFrame(cm / cm.sum(1, keepdims=True) * 100, index=[f'dreem_{c}' for c in NAMES],
                     columns=[f'model_{c}' for c in NAMES]).round(1).to_csv(f'{OUT}/confusion_{tag}_row_pct.csv')
    S, G = measure_agreement(N)
    S.to_csv(f'{OUT}/agreement_measures.csv', index=False)
    fig_confusion(cms['all scored epochs'], f'{OUT}/figures/confusion_all_scored.png',
                  'All scored epochs (row %, counts)')
    fig_confusion(cms['reliable-label epochs'], f'{OUT}/figures/confusion_reliable.png',
                  'Reliable-label epochs (row %, counts)')
    fig_kappa(G, f'{OUT}/figures/per_night_kappa.png')
    fig_scatter(G, S, f'{OUT}/figures/night_measures_scatter.png')
    fig_bland_altman(G, S, f'{OUT}/figures/night_measures_bland_altman.png')
    fig_hypnograms(D, N, f'{OUT}/figures/example_hypnograms.png')
    readme(D, N, E, S, f'{OUT}/README.md')
    pd.set_option('display.width', 250)
    print(E.round(3).to_string(index=False))
    print(S.drop(columns=['key']).round(2).to_string(index=False))
    print(G[['kappa', 'kappa_wake_sleep']].describe().round(2))


if __name__ == '__main__':
    main()
