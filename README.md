# wrist-sleep-staging

Sleep staging from an Empatica EmbracePlus wristband alone, for children and adolescents.

Two models, trained on 1,977 nights from 185 children aged 10–18 (typically developing, autistic, ADHD, autistic + ADHD)
who wore the wristband together with a Dreem EEG headband:

| Model | Input | |
|---|---|---|
| **Deep network** (`rawnet`) | raw pulse (BVP) + accelerometer at 32 Hz and time since the start of the night, over the whole night | the recommended model |
| **LightGBM** (`lgbm`) | Empatica's per-minute signals with rolling features | the baseline of the paper |

Both models are trained on five stages (Wake, N1, N2, N3, REM). The four-stage output (Wake, Light, N3, REM) is the
same hypnogram with N1 and N2 merged into Light, so the two outputs always agree.

Agreement with the EEG headband in held-out families (5-fold cross-validation by family). Epoch-level measures are on
reliable stretches of the EEG reference (10-min windows with complete scoring and clean EEG on both channels, by the
SSP Dreem power filter); night-level measures on whole nights. Empatica's own sleep detection is shown for comparison
(wake/sleep only):

| | Deep network | LightGBM | Empatica |
|---|---|---|---|
| Wake/sleep kappa | **0.85** | **0.85** | 0.69 |
| F1 sleep / wake | 0.98 / 0.87 | 0.98 / 0.87 | 0.94 / 0.75 |
| 4-stage Cohen's kappa | **0.71** | 0.62 | – |
| Macro F1 (4 stages) | **0.79** | 0.73 | – |
| F1 light / N3 / REM | **0.72 / 0.84 / 0.75** | 0.64 / 0.79 / 0.60 | – |
| 5-stage kappa (N1 F1) | **0.68** (0.36) | 0.59 (0.27) | – |
| Sleep onset latency: bias, r per night | −1.4 min, **0.88** | +0.9 min, 0.82 | −1.9 min, 0.88 |
| Total sleep time: bias, r per night | +28 min, 0.93 | +22 min, 0.93 | −37 min, 0.85 |
| WASO: bias, within-child r | −8.6 min, 0.44 | −6.5 min, 0.55 | +52 min, 0.35 |

Sleep and wake are separated with high agreement by both models, well above the wristband's own algorithm, and the
deep network identifies deep and REM sleep well. Sleep onset latency and total sleep time track the EEG closely from
night to night. Both models score some quiet wake as sleep, so total sleep time is over-estimated and WASO
under-estimated by a similar amount on every night. Accuracy did not differ between autistic and non-autistic children
or with age. See the paper [citation to be added] for the full validation.

## Install

```bash
git clone https://github.com/Micha098/wrist-sleep-staging.git
cd wrist-sleep-staging
pip install -e .          # numpy, pandas, torch, lightgbm
```

Scoring runs on a CPU (a few seconds per night).

## Score your nights

You need, per night:

1. **Raw signals**: CSVs `time,bvp` and `time,x,y,z` (Unix seconds; accelerometer in g, 64 Hz), covering the night and
   the hour before it. Convert Empatica's raw Avro files to this layout first.
2. **Start and end of the night**: bedtime (lights off) and rise time, with a UTC offset, e.g. `2024-05-03T21:30:00-04:00`.
   The network uses time since the start of the night; SOL is counted from the start you give.
3. **Per-minute data** (optional): Empatica's "aggregated per minute" digital biomarkers in one CSV with the columns
   `timestamp_unix` (ms or s), `pulse_rate_bpm`, `prv_rmssd_ms`, `temperature_celsius`, `respiratory_rate_brpm`,
   `activity_counts`, `step_counts`, `wearing_detection_percentage`. Needed only for the LightGBM baseline and for
   Empatica's wear detection; the deep network does not use it.

```bash
python -m wristsleep.score nights.csv --out results/
```

`nights.csv` (`minutes_csv` may be left out):

```csv
night_id,bvp_csv,acc_csv,start,end,minutes_csv
child01_2024-05-03,data/child01_bvp.csv,data/child01_acc.csv,2024-05-03T21:30:00-04:00,2024-05-04T07:10:00-04:00,data/child01_minutes.csv
```

Outputs: `results/<night_id>_epochs.csv` (one row per 30-s epoch: deep-network `stage` (Wake/Light/N3/REM) and `stage5`
(Wake/N1/N2/N3/REM), its five probabilities, LightGBM stage, wear) and `results/night_summary.csv` (deep-network TST,
SOL, WASO, awakenings, minutes and % of sleep in N1, N2, Light, N3 and REM; LightGBM TST, SOL, WASO, awakenings when
the per-minute file is given; wear (from the per-minute file, otherwise raw-data coverage) and a warning when the band
was worn < 80% of the night).

From Python:

```python
from wristsleep.score import Scorer
sc = Scorer()
epochs, summary = sc.score_night(start, end, bvp=bvp_df, acc=acc_df, minutes=minutes_df)  # minutes optional
```

**Recommended use:** the deep network for sleep/wake, stages, SOL, TST and WASO. Because the TST and WASO biases are
consistent, these measures compare well between nights and children.

## What the models expect

The scoring code rebuilds the inputs exactly as in training (`wristsleep/inputs.py`; tests compare it with the training
code):

- 30-s epochs from the start of the night, up to 16 h; time of night in hours since the start.
- Per-minute signals (LightGBM only): each epoch takes the values of the clock minute it starts in; minutes with < 50%
  wear are missing.
- Raw signals: linear interpolation to a 64 Hz grid (gaps > 1 s stay missing), averaged in pairs to 32 Hz, 960 samples
  per epoch; BVP robust-scaled per night (median / IQR) and clipped to ±10; accelerometer in g clipped to ±4.

**Validated population:** children and adolescents aged 10–18, Empatica EmbracePlus, home nights with the band worn
≥ 80% of the night. Outside these conditions (younger children, other devices, partly worn nights) accuracy is unknown.

## Weights

`weights/` holds the final models, trained on all 185 children (20% of families held out for early stopping):

| File | Model |
|---|---|
| `rawnet.pt`, `rawnet.json` | deep network (PyTorch state dict; JSON = classes and input scaling) |
| `lgbm.txt`, `lgbm.json` | LightGBM (LightGBM text model; JSON = feature order) |

The weights are loaded with `torch.load(..., weights_only=True)`: they contain tensors only, no code.

## Reproduce the training and the paper

`training/` contains the scripts that produced the models and every number in the paper, in the order they run
(see `training/README.md`). They need the paired EEG and wristband recordings, which are not public. `paper/figures.ipynb`
rebuilds all paper figures from the saved predictions.

## Citation

[Hacohen M., et al. Wrist-only sleep staging in autistic and typically developing children (in preparation).]

## License

MIT (see `LICENSE`).
