"""
Wrist-only sleep model, step 3: sequence network over the whole night.

Input  : per 30-s epoch, the minute wristband signals (log activity, log steps, HR, PRV, temperature, respiration,
         HR and temperature relative to the night median, worn flag, missing flags, hours since recording start),
         scaled by median / IQR of the training nights, clipped to +-5, missing -> 0. First 6 h of the night
         (labels end at 3 h; the rest is context).
Model  : average pairs of epochs to minute resolution (the wrist data are per minute) -> 1-D CNN (local patterns, dilations up to ~8 min) -> 2-layer bidirectional GRU (the night as a sequence)
         -> per-minute 4-class output (Wake, Light, N3, REM), repeated for its two 30-s epochs.
Labels : Dreem, training mask by WRIST_LABELS (3h or quality, as baseline_lgbm.py); loss is masked elsewhere.
         Class-weighted CE.
Splits : the same 5 family folds as baseline_lgbm.py (read from baseline_pred_3h.parquet); 20 % of training
         families for early stopping on validation loss.

Outputs: wrist/seqnet_pred_<labels>[_bvp].parquet (SEQNET_BVP=1 adds the raw-BVP beat features) (+ scores with baseline_lgbm.score / score_late)
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_lgbm import CLASS, LABELS, NAMES, P, ST_TAG

# CPU is ~3x faster than MPS for this GRU on the M5; set SEQNET_DEV to override
DEV = torch.device(os.environ.get('SEQNET_DEV', 'cuda' if torch.cuda.is_available() else 'cpu'))
# input crop: 6 h is enough context for 3-h labels; quality labels run all night
MAX_T = int(float(os.environ.get('SEQNET_CROP_H', '6' if LABELS == '3h' else '16')) * 120)
SEEDS = int(os.environ.get('SEQNET_SEEDS', '1'))
BVP = os.environ.get('SEQNET_BVP', '0') == '1'  # add beat features from raw BVP (bvp_epoch_features.py)
BVP_FEATS = ['bvp_hr', 'bvp_ibi_sd', 'bvp_rmssd', 'bvp_amp', 'bvp_amp_cv', 'bvp_valid']
FOLDS = [int(k) for k in os.environ.get('SEQNET_FOLDS', '0,1,2,3,4').split(',')]
MAX_EPOCHS = int(os.environ.get('SEQNET_MAX_EPOCHS', '40'))
TAG = LABELS + ST_TAG + ('_bvp' if BVP else '') + ('' if FOLDS == [0, 1, 2, 3, 4] else '_folds' + ''.join(map(str, FOLDS)))
# cuDNN in the cluster `module load python` torch fails on GRU weights (as train/option2.py); native CUDA kernels
torch.backends.cudnn.enabled = DEV.type != 'cuda'


def night_arrays():
    E = pd.read_parquet(f'{P}/wrist/epochs.parquet').sort_values(['name', 'epoch']).reset_index(drop=True)
    E = E.merge(pd.read_parquet(f'{P}/wrist/label_masks.parquet', columns=['name', 'epoch', f'mask_{LABELS}']),
                on=['name', 'epoch'], how='left', validate='1:1').rename(columns={f'mask_{LABELS}': 'use'})
    E['use'] = E.use.fillna(False).astype(bool)
    if BVP:
        E = E.merge(pd.read_parquet(f'{P}/wrist/bvp_epoch_features.parquet'), on=['name', 'epoch'], how='left',
                    validate='1:1')
    worn = E.wear >= 50
    sig = ['act', 'steps', 'hr', 'prv', 'temp', 'resp']
    E.loc[~worn, sig] = np.nan
    E['act'], E['steps'] = np.log1p(E.act), np.log1p(E.steps)
    g = E.groupby('name')
    E['hr_rel'] = E.hr - g.hr.transform('median')
    E['temp_rel'] = E.temp - g.temp.transform('median')
    E['pos_h'] = E.epoch / 120
    cont = sig + ['hr_rel', 'temp_rel', 'pos_h'] + (BVP_FEATS if BVP else [])
    names = E.name.unique()
    N, T = len(names), MAX_T
    X = np.full((N, T, len(cont)), np.nan, np.float32)
    W = np.zeros((N, T), np.float32)
    Y = np.full((N, T), -1, np.int64)
    L = np.zeros(N, np.int64)
    meta = []
    for i, (name, d) in enumerate(E.groupby('name', sort=False)):
        n = min(len(d), T)
        X[i, :n] = d[cont].to_numpy(np.float32)[:n]
        W[i, :n] = (d.wear >= 50).to_numpy(np.float32)[:n]
        y = d.stage.map(CLASS).to_numpy()[:n]
        lab = ~np.isnan(y) & d.use.to_numpy()[:n]
        Y[i, :n][lab] = y[lab].astype(np.int64)
        L[i] = n
        meta.append(dict(name=name, SPID=d.SPID.iloc[0], family=d.family_SPID.fillna(d.SPID).iloc[0], n=n))
    return X, W, Y, L, pd.DataFrame(meta), cont


class Net(nn.Module):
    def __init__(self, c_in, d=64, n_cls=len(NAMES), p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, d, 5, padding=2), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(d, d, 5, padding=4, dilation=2), nn.GELU(), nn.Dropout(p),
            nn.Conv1d(d, d, 5, padding=8, dilation=4), nn.GELU(),
        )
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=p)
        self.out = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, n_cls))

    def forward(self, x):
        T = x.shape[1]
        h = F.avg_pool1d(x.transpose(1, 2), 2, ceil_mode=True)  # 30-s epochs -> minutes
        h = self.conv(h).transpose(1, 2)
        h, _ = self.gru(h)
        return self.out(h).repeat_interleave(2, dim=1)[:, :T]


def scale_stats(X, tr):
    """Per-input median and IQR (SD, then 1, when the IQR is ~0) over the nights `tr`."""
    flat = X[tr].reshape(-1, X.shape[2])
    med = np.nanmedian(flat, 0)
    iqr = np.nanpercentile(flat, 75, 0) - np.nanpercentile(flat, 25, 0)
    sd = np.nanstd(flat, 0)
    return med, np.where(iqr > 1e-6, iqr, np.where(sd > 1e-6, sd, 1.0))


def scale(X, W, tr):
    med, iqr = scale_stats(X, tr)
    Z = np.clip((X - med) / iqr, -5, 5)
    miss = np.isnan(Z).astype(np.float32)
    Z = np.nan_to_num(Z, nan=0.0)
    return np.concatenate([Z, miss, W[..., None]], axis=2).astype(np.float32)


def run_fold(Z, Y, L, fit, val, test, seed, max_epochs=40, patience=6, bs=16):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Zt, Yt = torch.from_numpy(Z).to(DEV), torch.from_numpy(Y).to(DEV)
    counts = np.bincount(Y[fit][Y[fit] >= 0], minlength=len(NAMES))
    wts = torch.tensor(counts.sum() / (len(NAMES) * np.maximum(counts, 1)), dtype=torch.float32, device=DEV)
    net = Net(Z.shape[2]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    def batches(idx):
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            t = int(L[b].max())
            yield b, Zt[b, :t], Yt[b, :t]

    def loss_on(idx):
        net.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b, x, y in batches(idx):
                lo = net(x)
                m = y >= 0
                tot += F.cross_entropy(lo[m], y[m], weight=wts, reduction='sum').item()
                n += int(m.sum())
        return tot / max(n, 1)

    best, best_state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(max_epochs):
        net.train()
        for b, x, y in batches(rng.permutation(fit)):
            lo = net(x)
            m = y >= 0
            if not m.any():
                continue
            loss = F.cross_entropy(lo[m], y[m], weight=wts)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        v = loss_on(val)
        if v < best - 1e-4:
            best, bad, best_ep = v, 0, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    probs = {}
    with torch.no_grad():
        for b, x, _ in batches(test):
            pr = torch.softmax(net(x), -1).cpu().numpy()
            for j, i in enumerate(b):
                probs[i] = pr[j, :L[i]]
    return probs, best_ep, best


def main():
    t0 = time.time()
    X, W, Y, L, meta, cont = night_arrays()
    folds = pd.read_parquet(f'{P}/wrist/baseline_pred_3h.parquet', columns=['name', 'fold']).drop_duplicates('name')
    meta = meta.merge(folds, on='name', how='left')
    assert meta.fold.notna().all()
    print(f'{len(meta)} nights, {X.shape[2]} signals, device {DEV}, arrays in {time.time() - t0:.0f}s', flush=True)
    out = []
    for k in FOLDS:
        tr = np.where(meta.fold != k)[0]
        test = np.where(meta.fold == k)[0]
        fams = meta.family.iloc[tr].drop_duplicates().sample(frac=0.2, random_state=k)
        val = tr[meta.family.iloc[tr].isin(fams).to_numpy()]
        fit = np.setdiff1d(tr, val)
        Z = scale(X, W, fit)
        acc = {}
        for seed in range(SEEDS):
            probs, be, bv = run_fold(Z, Y, L, fit, val, test, seed, max_epochs=MAX_EPOCHS)
            for i, p in probs.items():
                acc[i] = acc.get(i, 0) + p / SEEDS
            print(f'fold {k} seed {seed}: fit {len(fit)} val {len(val)} test {len(test)} nights, best epoch {be}, '
                  f'val loss {bv:.3f}, {time.time() - t0:.0f}s', flush=True)
        for i, p in acc.items():
            df = pd.DataFrame(p, columns=[f'p_{c}' for c in NAMES])
            df.insert(0, 'epoch', np.arange(L[i]))
            df.insert(0, 'name', meta.name.iloc[i])
            out.append(df)
    pr = pd.concat(out, ignore_index=True)
    pr['seqnet'] = pr[[f'p_{c}' for c in NAMES]].to_numpy().argmax(1)
    pr.to_parquet(f'{P}/wrist/seqnet_pred_{TAG}.parquet', index=False, compression=None)  # cluster pyarrow: no snappy
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
