"""
Wrist-only sleep model: the evaluation table of the paper.

Both models are trained on five stages (Wake, N1, N2, N3, REM). Their four-stage output is the
5-stage hypnogram with N1 and N2 merged into Light (hard labels), so the 4- and 5-stage outputs always agree.

Input : baseline_pred_quality_5st.parquet (labels, reliable-stretch mask, Empatica sleep code, LightGBM out-of-fold predictions)
        rawnet_pred/rawnet_pred_quality_5st_raw_fold?.parquet (deep-network out-of-fold predictions)
Output: wrist/eval_pred.parquet, one row per 30-s epoch: SPID, name, date, epoch, stage (Dreem code), y (4-class Dreem
        label), mask_quality, empatica, lgbm_5m and raw_5m (4-class predictions), lgbm_5 and raw_5 (5-class)
"""
import glob

import numpy as np
import pandas as pd

from baseline_lgbm import P

W = f'{P}/wrist'
MERGE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}        # Wake, N1, N2, N3, REM -> Wake, Light, N3, REM


def main():
    B = pd.read_parquet(f'{W}/baseline_pred_quality_5st.parquet',
                        columns=['SPID', 'name', 'date', 'epoch', 'stage', 'y', 'mask_quality', 'empatica', 'lgbm'])
    fs = sorted(glob.glob(f'{W}/rawnet_pred/rawnet_pred_quality_5st_raw_fold?.parquet'))
    assert len(fs) == 5
    R = pd.concat([pd.read_parquet(f, columns=['name', 'epoch', 'rawnet']) for f in fs])
    A = B.rename(columns={'lgbm': 'lgbm_5', 'y': 'y5'}).merge(R.rename(columns={'rawnet': 'raw_5'}), on=['name', 'epoch'],
                                                               how='left', validate='1:1')
    A['y'] = A.y5.map(MERGE)
    A['lgbm_5m'] = A.lgbm_5.map(MERGE)
    A['raw_5m'] = A.raw_5.map(MERGE)
    A = A.drop(columns='y5').sort_values(['name', 'epoch']).reset_index(drop=True)
    print(f'{A.name.nunique():,} nights, {len(A):,} epochs; deep-network coverage {A.raw_5m.notna().mean():.4f}')
    A.to_parquet(f'{W}/eval_pred.parquet', index=False, compression=None)


if __name__ == '__main__':
    main()
