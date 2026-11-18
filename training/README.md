# Training and evaluation pipeline

These are the scripts that produced the released models and every number in the paper. They need the paired Dreem +
Empatica recordings (not public). Set `SOP_PROJ` to the project data folder; paths inside it are as below. Run the
scripts from this folder (`cd training`), so they can import each other.

| Step | Script | Input -> output |
|---|---|---|
| 1. Epoch table | `build_epochs.py` | Dreem hypnograms + Empatica per-minute data -> `wrist/epochs.parquet` (one row per 30-s epoch on nights worn >= 80%) |
| 2. Reference labels | `build_label_masks.py` | Dreem stages + EEG quality markers (`cache/eeg_quality.parquet`) -> `wrist/label_masks.parquet` (reliable 10-min stretches) |
| 3. Raw signals | `prep/cut_bvp_nights.py`, `prep/cut_acc_nights.py`, `prep/pack_raw.py` | raw 64 Hz CSVs -> per-night 32 Hz arrays -> `cache/raw_pack/raw.npy` [epochs, 5, 960] |
| 4. LightGBM | `baseline_lgbm.py` | `WRIST_LABELS=quality` (`WRIST_STAGES=5` for N1/N2) -> out-of-fold predictions `wrist/baseline_pred_quality*.parquet` |
| 5. Deep network | `rawnet.py` via `rawnet.sbatch` (one family fold per GPU job) | `sbatch --array=0-4 rawnet.sbatch` -> `wrist/rawnet_pred_quality_both_fold<k>.parquet`; minute-input scaling comes from `seqnet.py` |
| 6. Final models | `train_final_lgbm.py`; `rawnet.py` with `RAWNET_FOLD=-1` | all families, 20% for early stopping -> `wrist/lgbm_final_quality*.txt`, `wrist/rawnet_final_quality*_both.pt` |
| 7. Evaluation | `evaluation/paper_metrics.py`, `evaluation/paper_tables.py`, `evaluation/score_5stage.py`, `evaluation/export_scored_nights.py` | out-of-fold predictions -> `wrist/paper/*.csv` (all tables of the paper) |

Environment variables: `SOP_PROJ` (data folder), `WRIST_LABELS` (`quality` = reliable stretches, used for the paper),
`WRIST_STAGES` (4 or 5), `RAWNET_FOLD`, `RAWNET_INPUT` (`both` | `raw` | `minute`), `RAWNET_SEED`, `RAWNET_MAX_EPOCHS`,
`WRIST_CHILDREN` (CSV with SPID, group4, age, male, asd, adhd for the group analyses).

Folds: families are assigned to 5 folds at random (`baseline_lgbm.family_folds`, seed 0); both models use the same test
folds. Within the training families of each fold, 20% of families (random, seed = fold) are used for early stopping.

Computing: LightGBM on a laptop CPU (minutes per fold). The deep network on one NVIDIA A100 or H100 (PyTorch 2.6,
CUDA 12.5, cuDNN 8.9; 5–9 min per fold).
