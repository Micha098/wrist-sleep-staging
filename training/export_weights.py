"""
Convert the final training checkpoints (rawnet.py with RAWNET_FOLD=-1) into the released format:
weights/rawnet.pt (tensor-only state dict, loads with weights_only=True) + weights/rawnet.json (classes, input mode
(`raw`: raw pulse + motion + time of night), input order, input scaling). LightGBM: copies wrist/lgbm_final_quality_5st.txt/.json to weights/lgbm.*.
    python export_weights.py <SOP_PROJ>/wrist ../weights
    python export_weights.py <SOP_PROJ>/wrist ../weights acc   (motion-only network -> weights/rawnet_acc.*)
"""
import json
import shutil
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
inp = sys.argv[3] if len(sys.argv) > 3 else 'raw'
stem = 'rawnet' if inp == 'raw' else f'rawnet_{inp}'
ck = torch.load(f'{src}/rawnet_final_quality_5st_{inp}.pt', map_location='cpu', weights_only=False)  # our own file
torch.save({k: v.contiguous() for k, v in ck['state_dict'].items()}, f'{dst}/{stem}.pt')
json.dump(dict(names=list(ck['names']), input=ck['input'], inputs=list(ck['cont']), med=[float(x) for x in ck['med']],
               iqr=[float(x) for x in ck['iqr']], best_epoch=int(ck['best_epoch']), val_loss=float(ck['val_loss']),
               n_fit_nights=int(ck['n_fit']), n_val_nights=int(ck['n_val'])), open(f'{dst}/{stem}.json', 'w'), indent=1)
if inp == 'raw':
    for ext in ['txt', 'json']:
        shutil.copy(f'{src}/lgbm_final_quality_5st.{ext}', f'{dst}/lgbm.{ext}')
print('exported')
