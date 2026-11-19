"""Per-epoch features of the SSP power filter (dreem_processing/power_filter.py) from cache/eeg/<SPID>/*.parquet ->
cache/eeg_power_filter_features.parquet (read-only pull): kurtosis, max amplitude, variance, root total power on
F7-O1, F7-O2, F8-O1, F8-O2."""
import glob, os
import pandas as pd, pyarrow.parquet as pq
C = '/mnt/ceph/users/mhacohen/Sleep_Onset_prediction_model/cache'
CH = ['F7-O1', 'F7-O2', 'F8-O1', 'F8-O2']
FEAT = ['kurtosis', 'max_amplitude', 'variance', 'root_total_power']
want = ['epoch', 'time_utc', 'SPID', 'stage_dreem'] + [f'{f}_{c}' for c in CH for f in FEAT]
out = []
for f in sorted(glob.glob(f'{C}/eeg/SP*/*.parquet')):
    have = set(pq.read_schema(f).names)
    d = pd.read_parquet(f, columns=[c for c in want if c in have])
    d.insert(0, 'name', os.path.basename(f).replace('.parquet', '.edf'))
    out.append(d)
q = pd.concat(out, ignore_index=True)
for c in q.columns:
    if q[c].dtype == 'float64':
        q[c] = q[c].astype('float32')
q.to_parquet(f'{C}/eeg_power_filter_features.parquet', index=False)
print(len(out), 'files', q.shape)
