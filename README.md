# wrist-sleep-staging

Sleep staging from an Empatica EmbracePlus wristband alone, for children and adolescents.

Two models, trained on 1,977 nights from 185 children aged 10–18 (typically developing, autistic, ADHD, autistic + ADHD)
who wore the wristband together with a Dreem EEG headband:

| Model | Input | |
|---|---|---|
| **Deep network** (`rawnet`) | raw pulse (BVP) + accelerometer at 32 Hz, and Empatica's per-minute signals, over the whole night | the recommended model |
| **LightGBM** (`lgbm`) | Empatica's per-minute signals with rolling features | the baseline of the paper |

Both models are trained on five stages (Wake, N1, N2, N3, REM). The four-stage output (Wake, Light, N3, REM) is the
same hypnogram with N1 and N2 merged into Light, so the two outputs always agree.

Agreement with the EEG headband in held-out families (5-fold cross-validation by family):

| | Deep network | LightGBM |
|---|---|---|
| 4-stage Cohen's kappa, reliable epochs | **0.71** | 0.61 |
| Macro F1 (4 stages) | **0.80** | 0.72 |
| Wake/sleep kappa | **0.86** | 0.84 |
| 5-stage kappa (N1 F1) | **0.68** (0.37) | 0.58 (0.28) |
| Sleep onset latency: bias, r per night | −0.6 min, 0.86 | +0.4 min, 0.81 |
| Total sleep time: bias, r per night | +27 min, 0.93 | +24 min, 0.93 |
| WASO: bias, within-child r | −8.5 min, 0.53 | −7.1 min, 0.55 |

Both models over-estimate total sleep time and under-estimate WASO and awakenings (about one in eight wake epochs is
scored as sleep, mostly as N1); use these as relative measures (night-to-night, child-to-child) rather than absolute
durations. Accuracy did not differ between autistic and non-autistic children or with age. Per-night percentages of
light, deep and REM sleep agree poorly with the EEG (r 0.16–0.43) and should not be used to compare groups. See the paper
[citation to be added] for the full validation.

## Install

```bash
git clone https://github.com/Micha098/wrist-sleep-staging.git
cd wrist-sleep-staging
pip install -e .          # numpy, pandas, torch, lightgbm
```

Scoring runs on a CPU (a few seconds per night).

## Score your nights

You need, per night:

1. **Per-minute data**: Empatica's "aggregated per minute" digital biomarkers, merged into one CSV with the columns
   `timestamp_unix` (ms or s), `pulse_rate_bpm`, `prv_rmssd_ms`, `temperature_celsius`, `respiratory_rate_brpm`,
   `activity_counts`, `step_counts`, `wearing_detection_percentage`. These are Empatica's own column names.
2. **Raw signals** (recommended): CSVs `time,bvp` and `time,x,y,z` (Unix seconds; accelerometer in g, 64 Hz), covering the
   night and the hour before it. Convert Empatica's raw Avro files to this layout first. Without raw signals the deep
   network still runs, but it was validated with them.
3. **Start and end of the night**: bedtime (lights off) and rise time, with a UTC offset, e.g. `2024-05-03T21:30:00-04:00`.
   The models use time since the start of the night, so the start should be when the child went to bed.

```bash
python -m wristsleep.score nights.csv --out results/
```

`nights.csv`:

```csv
night_id,minutes_csv,bvp_csv,acc_csv,start,end
child01_2024-05-03,data/child01_minutes.csv,data/child01_bvp.csv,data/child01_acc.csv,2024-05-03T21:30:00-04:00,2024-05-04T07:10:00-04:00
```

Outputs: `results/<night_id>_epochs.csv` (one row per 30-s epoch: deep-network `stage` (Wake/Light/N3/REM) and `stage5`
(Wake/N1/N2/N3/REM), its five probabilities, LightGBM stage, wear) and `results/night_summary.csv` (deep-network TST,
SOL, WASO, awakenings, minutes and % of sleep in N1, N2, Light, N3 and REM; LightGBM TST, SOL, WASO, awakenings; wear
and raw-data coverage, and a warning when the band was worn < 80% of the night).

From Python:

```python
from wristsleep.score import Scorer
sc = Scorer()
epochs, summary = sc.score_night(minutes_df, start, end, bvp=bvp_df, acc=acc_df)
```

**Recommended use:** the deep network for stages, SOL, TST and WASO; TST and WASO as relative measures (see above).
Use N1 minutes as a relative index of fragmentation (they track night-to-night change, within-child r 0.71), not as
absolute minutes (N1 is over-counted about 2.6-fold).

## What the models expect

The scoring code rebuilds the inputs exactly as in training (`wristsleep/inputs.py`; tests compare it with the training
code):

- 30-s epochs from the start of the night, up to 16 h; each epoch takes the values of the clock minute it starts in.
- Minutes with < 50% wear are missing; movement and steps are log-transformed; heart rate and temperature are also given
  relative to the night's median; time of night in hours.
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
