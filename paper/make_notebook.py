"""Writes paper/figures.ipynb (run once; edit the notebook itself afterwards)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Wrist-only sleep staging: paper figures

This notebook rebuilds every figure of the paper from the saved model predictions and result tables. Run it top to
bottom. Each section says what the figure shows and how it is computed.

**Inputs** (in the project data folder, `SOP_PROJ`; participant data are not public):
- `wrist/eval_pred.parquet`: one row per 30-s epoch with the Dreem stage (`y`), the reliable-label flag
  (`mask_quality`), and the out-of-fold predictions of LightGBM (`lgbm_5m`) and the deep network (`raw_5m`): both models are trained on five
  stages, and these columns are their hypnograms with N1 and N2 merged into Light (`training/merge_5stage.py`);
- `wrist/baseline_pred_quality_5st.parquet` and `wrist/rawnet_pred/rawnet_pred_quality_5st_raw_fold*.parquet`: the five-stage predictions;
- `wrist/paper/child_metrics.csv` and `group_age_tests.csv`: per-child metrics and group regressions (`training/evaluation/paper_metrics.py`).

All predictions are out-of-fold: each night was scored by a model trained on other families only.""")

code("""import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

P = os.environ.get('SOP_PROJ', 'data')                    # project data folder
FIG = os.environ.get('WRIST_FIGURES', 'figures')          # where figures are written
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'training'))
sys.path.insert(0, os.path.abspath('../training'))
from baseline_lgbm import night_measures                  # the same night-measure definitions as the paper

NAMES = ['Wake', 'Light', 'N3', 'REM']
MODELS = {'lgbm_5m': 'LightGBM', 'raw_5m': 'Deep network'}
COL = {'LightGBM': '#9a9a9a', 'Deep network': '#2f6fd6'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})

A = pd.read_parquet(f'{P}/wrist/eval_pred.parquet',
                    columns=['SPID', 'name', 'date', 'epoch', 'y', 'mask_quality'] + list(MODELS)).sort_values(['name', 'epoch'])
A['reliable'] = A.mask_quality.fillna(False).astype(bool) & A.y.notna()
print(f"{A.name.nunique():,} nights, {A.SPID.nunique()} children, {len(A):,} epochs, "
      f"{A.y.notna().sum():,} scored, {A.reliable.sum():,} reliable")""")

md("""## Figure 1. Analysis pipeline

A schematic of the steps in Methods: both devices are aligned on 30-s epochs, reliable EEG stretches give the labels,
both models get the same labels and family folds; LightGBM reads Empatica's minute signals, the deep network raw pulse
and motion plus time of night.""")
code("""boxes = [  # (x, y, w, label, highlight)
    (0.02, 0.86, 0.46, 'Dreem EEG headband\\nautomatic 30-s stages', False),
    (0.52, 0.86, 0.46, 'Empatica wristband\\nminute signals; raw pulse + motion (64 Hz)', False),
    (0.15, 0.70, 0.70, 'Align on 30-s epochs\\n1,977 nights worn >= 80% of the night', False),
    (0.02, 0.54, 0.46, 'Reliable-stretch labels (5 stages)\\n10 min: <= 20% unscored, <= 30% noisy', False),
    (0.52, 0.54, 0.46, 'Wrist inputs\\nLightGBM: 9 minute signals\\ndeep net: raw pulse + motion, time of night', False),
    (0.02, 0.38, 0.46, 'LightGBM (baseline)\\n39 hand-crafted features', False),
    (0.52, 0.38, 0.46, 'Deep network\\nCNN encoder > TCN > transformer', True),
    (0.15, 0.22, 0.70, '5-fold cross-validation by family\\nfit / validation / test families', False),
    (0.10, 0.06, 0.80, 'Agreement with Dreem\\nepochs (kappa, F1) - nights (TST, SOL, WASO) - autism, ADHD, age', False)]
fig, ax = plt.subplots(figsize=(8, 8.5)); ax.axis('off')
for x, y, w, lab, hi in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, 0.1, boxstyle='round,pad=0.005,rounding_size=0.015',
                                fc='#e8f0fc' if hi else 'white', ec='#2f6fd6' if hi else '#888', lw=1.6 if hi else 1))
    ax.text(x + w / 2, y + 0.05, lab, ha='center', va='center', fontsize=8.5)
arrow = dict(arrowstyle='-|>', color='#888', lw=1)
for (x0, y0), (x1, y1) in [((0.26, 0.86), (0.26, 0.80)), ((0.74, 0.86), (0.74, 0.80)), ((0.40, 0.70), (0.26, 0.64)),
                           ((0.60, 0.70), (0.74, 0.64)), ((0.26, 0.54), (0.26, 0.48)), ((0.74, 0.54), (0.74, 0.48)),
                           ((0.40, 0.54), (0.66, 0.48)), ((0.60, 0.54), (0.34, 0.48)), ((0.26, 0.38), (0.40, 0.32)),
                           ((0.74, 0.38), (0.60, 0.32)), ((0.5, 0.22), (0.5, 0.16))]:
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=arrow)
fig.savefig(f'{FIG}/fig1_pipeline.png', dpi=200, bbox_inches='tight'); plt.show()""")

md("""## Figure 2. F1 per stage (four stages, reliable epochs)

For each stage, F1 is the harmonic mean of precision (of the epochs called that stage, the share that were) and recall
(of the true epochs of that stage, the share found). Macro F1 is the unweighted mean over the four stages.""")
code("""R = A[A.reliable]
rows = []
for m, lab in MODELS.items():
    k = R[m].notna()
    pr, rc, f1, _ = precision_recall_fscore_support(R.y[k].astype(int), R[m][k].astype(int), labels=range(4), zero_division=0)
    rows += [dict(model=lab, stage=s, precision=pr[i], recall=rc[i], F1=f1[i]) for i, s in enumerate(NAMES)]
    rows.append(dict(model=lab, stage='Macro F1', F1=f1.mean()))
S = pd.DataFrame(rows)
display(S.pivot(index='stage', columns='model', values='F1').loc[NAMES + ['Macro F1']].round(3))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
cats = NAMES + ['Macro F1']; x = np.arange(len(cats)) + np.r_[0, 0, 0, 0, 0.4]
for j, lab in enumerate(MODELS.values()):
    v = S[S.model == lab].set_index('stage').F1.loc[cats]
    b = ax.bar(x + (j - 0.5) * 0.38, v, 0.36, color=COL[lab], label=lab)
    ax.bar_label(b, fmt='%.2f', fontsize=8.5)
ax.set_xticks(x, cats); ax.set_ylim(0, 1); ax.set_ylabel('F1'); ax.legend(frameon=False, ncol=2, loc='upper right')
ax.set_title('F1 per stage on reliable-label epochs')
fig.savefig(f'{FIG}/fig2_f1_per_stage.png', dpi=200, bbox_inches='tight'); plt.show()""")

md("""## Figure 3. Confusion matrices (four stages, reliable epochs)

Rows are the Dreem stage and sum to 100%; columns are the model's prediction. The diagonal is agreement.""")
code("""fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, (m, lab) in zip(axs, MODELS.items()):
    k = R[m].notna()
    cm = confusion_matrix(R.y[k].astype(int), R[m][k].astype(int), labels=range(4))
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{pct[i, j]:.0f}%', ha='center', va='center', color='white' if pct[i, j] > 55 else 'black')
    ax.set_xticks(range(4), NAMES); ax.set_yticks(range(4), NAMES)
    ax.set_xlabel(lab); ax.set_ylabel('Dreem EEG')
    ax.set_title(f'{lab}: kappa {cohen_kappa_score(R.y[k].astype(int), R[m][k].astype(int)):.2f}')
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_confusion.png', dpi=200, bbox_inches='tight'); plt.show()""")

md("""## Night-level measures

From each hypnogram (Dreem, LightGBM, deep network): TST, SOL, WASO and awakenings (as in Methods 2.9), and the
percentage of sleep in Light, N3 and REM on epochs scored by both. Agreement uses nights with >= 80% of epochs scored.""")
code("""MEAS = [('TST', 'Total sleep time (min)'), ('SOL', 'Sleep onset latency (min)'), ('WASO', 'WASO (min)'),
        ('n_awak', 'Awakenings (n)'), ('pct_Light', 'Light (% of sleep)'), ('pct_N3', 'N3 (% of sleep)'), ('pct_REM', 'REM (% of sleep)')]
rows = []
for name, g in A.groupby('name', sort=False):
    r = dict(name=name, SPID=g.SPID.iloc[0], pct_scored=g.y.notna().mean() * 100)
    both = g[g.y.notna()]
    for src in ['y'] + list(MODELS):
        r.update({f'{k}_{src}': v for k, v in night_measures(g, src).items()})
        b = both[both[src].notna()]; sl = b[src][b[src] > 0]
        for i, c in [(1, 'Light'), (2, 'N3'), (3, 'REM')]:
            r[f'pct_{c}_{src}'] = (sl == i).mean() * 100 if len(sl) else np.nan
    rows.append(r)
N = pd.DataFrame(rows)
G = N[N.pct_scored >= 80]
print(f'{len(G):,} nights with >= 80% scored, {G.SPID.nunique()} children')""")

md("""## Figure 4. Night measures: model vs Dreem

One dot per night; the orange line is identity. Panel titles give Pearson r and the within-child r (both values centred
on each child's own mean: does the model track which nights were better or worse for that child?).""")
code("""def within_r(d, a, b):
    wa = d[a] - d.groupby('SPID')[a].transform('mean'); wb = d[b] - d.groupby('SPID')[b].transform('mean')
    return pearsonr(wa, wb)[0]

fig, axs = plt.subplots(2, 7, figsize=(22, 6.5))
for row, (m, lab) in enumerate(MODELS.items()):
    for ax, (key, title) in zip(axs[row], MEAS):
        d = G[['SPID', f'{key}_y', f'{key}_{m}']].dropna()
        a, b = d[f'{key}_y'], d[f'{key}_{m}']
        ax.scatter(a, b, s=5, alpha=0.35, color=COL[lab], edgecolors='none')
        lo, hi = min(a.min(), b.min()), max(a.quantile(.995), b.quantile(.995))
        ax.plot([lo, hi], [lo, hi], color='#e8962e', lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f'{title}\\nr {pearsonr(a, b)[0]:.2f} | within-child r {within_r(d, f"{key}_y", f"{key}_{m}"):.2f}', fontsize=9)
        ax.set_xlabel('Dreem'); ax.set_ylabel(lab if key == 'TST' else '')
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_night_scatter.png', dpi=200, bbox_inches='tight'); plt.show()""")

md("""## Figure 5. Bland-Altman plots

Difference (model - Dreem) against the mean of the two, per night. Solid line: bias; dashed: 95% limits of agreement
(bias +- 1.96 SD of the differences).""")
code("""fig, axs = plt.subplots(2, 7, figsize=(22, 6.5))
for row, (m, lab) in enumerate(MODELS.items()):
    for ax, (key, title) in zip(axs[row], MEAS):
        d = G[[f'{key}_y', f'{key}_{m}']].dropna(); a, b = d.iloc[:, 0], d.iloc[:, 1]
        diff = b - a; bias, sd = diff.mean(), diff.std()
        ax.scatter((a + b) / 2, diff, s=5, alpha=0.35, color=COL[lab], edgecolors='none')
        for v, ls in [(bias, '-'), (bias - 1.96 * sd, '--'), (bias + 1.96 * sd, '--')]:
            ax.axhline(v, color='#e8962e', ls=ls, lw=1)
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_title(f'{title}\\nbias {bias:+.1f}, LoA {bias - 1.96 * sd:.1f} to {bias + 1.96 * sd:.1f}', fontsize=9)
        ax.set_ylabel(f'{lab} - Dreem' if key == 'TST' else '')
fig.tight_layout(); fig.savefig(f'{FIG}/fig5_bland_altman.png', dpi=200, bbox_inches='tight'); plt.show()""")

md("""## Figure 6. Five stages (N1 and N2 separated)

The same models retrained with five classes. Left: F1 per stage on reliable epochs; right: the deep network's confusion matrix.""")
code("""N5 = ['Wake', 'N1', 'N2', 'N3', 'REM']
B5 = pd.read_parquet(f'{P}/wrist/baseline_pred_quality_5st.parquet', columns=['name', 'epoch', 'y', 'mask_quality', 'lgbm'])
R5 = pd.concat([pd.read_parquet(f, columns=['name', 'epoch', 'rawnet'])
                for f in sorted(glob.glob(f'{P}/wrist/rawnet_pred/rawnet_pred_quality_5st_raw_fold?.parquet'))])
D5 = B5.merge(R5, on=['name', 'epoch'])
D5 = D5[D5.mask_quality & D5.y.notna()]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
for j, (m, lab) in enumerate([('lgbm', 'LightGBM'), ('rawnet', 'Deep network')]):
    f1 = precision_recall_fscore_support(D5.y.astype(int), D5[m].astype(int), labels=range(5), zero_division=0)[2]
    v = np.r_[f1, f1.mean()]; x = np.arange(6) + np.r_[0, 0, 0, 0, 0, 0.4]
    b = axs[0].bar(x + (j - 0.5) * 0.38, v, 0.36, color=COL[lab], label=f'{lab} (kappa {cohen_kappa_score(D5.y.astype(int), D5[m].astype(int)):.2f})')
    axs[0].bar_label(b, fmt='%.2f', fontsize=8)
axs[0].set_xticks(np.arange(6) + np.r_[0, 0, 0, 0, 0, 0.4], N5 + ['Macro F1']); axs[0].set_ylim(0, 1)
axs[0].set_ylabel('F1'); axs[0].legend(frameon=False, loc='upper right', fontsize=8.5); axs[0].set_title('F1 per stage, five classes')
cm = confusion_matrix(D5.y.astype(int), D5.rawnet.astype(int), labels=range(5)); pct = cm / cm.sum(1, keepdims=True) * 100
axs[1].imshow(pct, cmap='Blues', vmin=0, vmax=100)
for i in range(5):
    for j in range(5):
        axs[1].text(j, i, f'{pct[i, j]:.0f}%', ha='center', va='center', fontsize=9, color='white' if pct[i, j] > 55 else 'black')
axs[1].set_xticks(range(5), N5); axs[1].set_yticks(range(5), N5); axs[1].set_xlabel('Deep network'); axs[1].set_ylabel('Dreem EEG')
fig.tight_layout(); fig.savefig(f'{FIG}/fig6_five_stages.png', dpi=200, bbox_inches='tight'); plt.show()""")

md("""## Figure 7. Accuracy by group and age

Per-child four-stage kappa on reliable epochs (children with >= 200 reliable epochs), from `paper_metrics.py`. Left: by
group; right: against age. Regression coefficients (autism, ADHD, age, sex) are in `group_age_tests.csv`.""")
code("""K = pd.read_csv(f'{P}/wrist/paper/child_metrics.csv')
K = K[K.model.isin(MODELS)].assign(model=lambda d: d.model.map(MODELS)).dropna(subset=['group4'])
groups = ['TD', 'ADHD', 'ASD', 'ASD+ADHD']
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
rng = np.random.default_rng(0)
for j, lab in enumerate(MODELS.values()):
    for i, g in enumerate(groups):
        v = K[(K.model == lab) & (K.group4 == g)].kappa
        axs[0].scatter(i + (j - 0.5) * 0.36 + rng.uniform(-0.08, 0.08, len(v)), v, s=9, alpha=0.5, color=COL[lab])
        axs[0].plot([i + (j - 0.5) * 0.36 - 0.12, i + (j - 0.5) * 0.36 + 0.12], [v.mean()] * 2, color='black', lw=2)
    d = K[K.model == lab]
    axs[1].scatter(d.age, d.kappa, s=9, alpha=0.5, color=COL[lab], label=lab)
    c = np.polyfit(d.age, d.kappa, 1); xx = np.linspace(d.age.min(), d.age.max(), 2)
    axs[1].plot(xx, np.polyval(c, xx), color=COL[lab], lw=2)
axs[0].legend(handles=[plt.Line2D([], [], marker='o', ls='', color=COL[l], label=l) for l in MODELS.values()], frameon=False, loc='lower left')
axs[0].set_xticks(range(4), groups); axs[0].set_ylabel('Per-child 4-stage kappa'); axs[0].set_title('By group (bars = means)')
axs[1].set_xlabel('Age (years)'); axs[1].set_title('Against age'); axs[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/fig7_groups_age.png', dpi=200, bbox_inches='tight'); plt.show()
T = pd.read_csv(f'{P}/wrist/paper/group_age_tests.csv')
display(T[T.metric.isin(['kappa', 'macro_f1']) & T.model.isin(MODELS)][['model', 'metric', 'n', 'b_asd', 'p_asd', 'b_adhd', 'p_adhd', 'b_age', 'p_age']].round(3))""")

md("""## Figure 8. Example nights

Hypnograms of three nights at the 10th, 50th and 90th percentile of the deep network's per-night kappa (nights with
>= 80% scored), with Dreem (top trace), the deep network and LightGBM.""")
code("""kap = {name: cohen_kappa_score(g.y.astype(int), g.raw_5m.astype(int), labels=range(4))
       for name, g in A[A.name.isin(G.name)].dropna(subset=['y', 'raw_5m']).groupby('name')}
kap = pd.Series(kap).sort_values()
pick = [kap.index[int(q * (len(kap) - 1))] for q in (0.1, 0.5, 0.9)]
level = {0: 3, 3: 2, 1: 1, 2: 0}  # Wake, REM, Light, N3 from top to bottom
fig, axs = plt.subplots(3, 1, figsize=(12, 7.5))
for ax, name, q in zip(axs, pick, ['10th', '50th', '90th']):
    g = A[A.name == name]; h = g.epoch / 120
    for col, off, lab, c in [('y', 0.16, 'Dreem', 'black'), ('raw_5m', 0, 'Deep network', COL['Deep network']), ('lgbm_5m', -0.16, 'LightGBM', '#e8962e')]:
        ax.step(h, g[col].map(level) + off, where='post', lw=1, color=c, label=lab)
    ax.set_yticks([0, 1, 2, 3], ['N3', 'Light', 'REM', 'Wake']); ax.set_xlim(0, h.max())
    ax.set_title(f'{q} percentile night: deep-network kappa {kap[name]:.2f}', fontsize=10)
axs[-1].set_xlabel('Hours from the start of the recording')
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0))
fig.savefig(f'{FIG}/fig8_example_nights.png', dpi=200, bbox_inches='tight'); plt.show()""")

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbf.write(nb, 'figures.ipynb')
print('wrote figures.ipynb')
