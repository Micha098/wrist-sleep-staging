# Training and evaluation pipeline

These are the scripts that produced the released models and every number in the paper. They need the paired Dreem +
Empatica recordings (not public). Set `SOP_PROJ` to the project data folder; paths inside it are as below. Run the
scripts from this folder (`cd training`), so they can import each other.

| Step | Script | Input -> output |
|---|---|---|
| 1. Epoch table | `build_epochs.py` | Dreem hypnograms + Empatica per-minute data -> `wrist/epochs.parquet` (one row per 30-s epoch on nights worn >= 80%) |
| 2. Reference labels | `prep/extract_power_filter_features.py`, `build_label_masks.py` | Dreem stages + EEG quality markers (`cache/eeg_quality.parquet`) + SSP power-filter features (`cache/eeg_power_filter_features.parquet`) -> `wrist/label_masks.parquet` (reliable 10-min stretches; noisy = power filter, 20-40 Hz power, ±500 µV or top-10% beta on F7-O1 or F8-O2). The cluster jobs read their own copy of `wrist/label_masks.parquet`: copy it to the cluster project folder after rebuilding |
| 3. Raw signals | `prep/cut_bvp_nights.py`, `prep/cut_acc_nights.py`, `prep/pack_raw.py` | raw 64 Hz CSVs -> per-night 32 Hz arrays -> `cache/raw_pack/raw.npy` [epochs, 5, 960] |
| 4. LightGBM | `baseline_lgbm.py` | reliable-stretch labels -> out-of-fold predictions `wrist/baseline_pred_quality_5st.parquet` |
| 5. Deep network | `rawnet.py` via `rawnet.sbatch` (one family fold per GPU job) | `sbatch --array=0-4 rawnet.sbatch` -> `wrist/rawnet_pred/rawnet_pred_quality_5st_raw_fold<k>.parquet` (copy them into `rawnet_pred/`); time of night and its scaling come from `minute_inputs.py` |
| 5b. Four stages | `merge_5stage.py` | joins labels and both models' out-of-fold predictions; N1 + N2 merged into Light -> `wrist/eval_pred.parquet` (used by all evaluation scripts and the figures notebook) |
| 6. Final models | `train_final_lgbm.py`; `rawnet.py` with `RAWNET_FOLD=-1`; `export_weights.py` | all families, 20% for early stopping -> `wrist/lgbm_final_quality_5st.txt`, `wrist/rawnet_final_quality_5st_raw.pt` -> `weights/` |
| 7. Evaluation | `evaluation/paper_metrics.py`, `evaluation/paper_tables.py`, `evaluation/score_5stage.py`, `evaluation/export_scored_nights.py` | out-of-fold predictions -> `wrist/paper/*.csv` (all tables of the paper) |

Environment variables: `SOP_PROJ` (data folder), `RAWNET_FOLD`, `RAWNET_SEED`, `RAWNET_MAX_EPOCHS`,
`WRIST_CHILDREN` (CSV with SPID, group4, age, male, asd, adhd for the group analyses).

Both models are trained and evaluated at the epoch level on the reliable stretches only; night-level measures (TST,
SOL, WASO, awakenings, stage percentages) use whole nights with >= 80% of epochs scored by Dreem.

Folds: families are assigned to 5 folds at random (`baseline_lgbm.family_folds`, seed 0); both models use the same test
folds. Within the training families of each fold, 20% of families (random, seed = fold) are used for early stopping.

Computing: LightGBM on a laptop CPU (minutes per fold). The deep network on one NVIDIA A100 or H100 (PyTorch 2.6,
CUDA 12.5, cuDNN 8.9; 4–11 min per fold).
