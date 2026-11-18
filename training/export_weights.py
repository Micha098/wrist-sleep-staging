"""
Convert the final training checkpoints (rawnet.py with RAWNET_FOLD=-1) into the released format:
weights/rawnet[_5st].pt (tensor-only state dict, loads with weights_only=True) + weights/rawnet[_5st].json (classes,
input order, input scaling). LightGBM: copies wrist/lgbm_final_quality[_5st].txt/.json to weights/lgbm[_5st].*.
    python export_weights.py <SOP_PROJ>/wrist ../weights
"""
import json
import shutil
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
for tag in ['', '_5st']:
    ck = torch.load(f'{src}/rawnet_final_quality{tag}_both.pt', map_location='cpu', weights_only=False)  # our own file
    torch.save({k: v.contiguous() for k, v in ck['state_dict'].items()}, f'{dst}/rawnet{tag}.pt')
    json.dump(dict(names=list(ck['names']), inputs=list(ck['cont']), med=[float(x) for x in ck['med']],
                   iqr=[float(x) for x in ck['iqr']], best_epoch=int(ck['best_epoch']), val_loss=float(ck['val_loss']),
                   n_fit_nights=int(ck['n_fit']), n_val_nights=int(ck['n_val'])), open(f'{dst}/rawnet{tag}.json', 'w'), indent=1)
    for ext in ['txt', 'json']:
        shutil.copy(f'{src}/lgbm_final_quality{tag}.{ext}', f'{dst}/lgbm{tag}.{ext}')
    print('exported', tag or '4 stages')
