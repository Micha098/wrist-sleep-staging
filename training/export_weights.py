"""
Convert the final training checkpoints (rawnet.py with RAWNET_FOLD=-1, WRIST_STAGES=5) into the released format:
weights/rawnet.pt (tensor-only state dict, loads with weights_only=True) + weights/rawnet.json (classes, input order,
input scaling). LightGBM: copies wrist/lgbm_final_quality_5st.txt/.json to weights/lgbm.*.
    python export_weights.py <SOP_PROJ>/wrist ../weights
"""
import json
import shutil
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(f'{src}/rawnet_final_quality_5st_both.pt', map_location='cpu', weights_only=False)  # our own file
torch.save({k: v.contiguous() for k, v in ck['state_dict'].items()}, f'{dst}/rawnet.pt')
json.dump(dict(names=list(ck['names']), inputs=list(ck['cont']), med=[float(x) for x in ck['med']],
               iqr=[float(x) for x in ck['iqr']], best_epoch=int(ck['best_epoch']), val_loss=float(ck['val_loss']),
               n_fit_nights=int(ck['n_fit']), n_val_nights=int(ck['n_val'])), open(f'{dst}/rawnet.json', 'w'), indent=1)
for ext in ['txt', 'json']:
    shutil.copy(f'{src}/lgbm_final_quality_5st.{ext}', f'{dst}/lgbm.{ext}')
print('exported')
