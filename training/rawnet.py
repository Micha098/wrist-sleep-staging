"""
Wrist-only sleep model, step 4: raw-signal network (raw BVP + ACC) on the cluster GPU, one family fold per job.

Input  : per 30-s epoch
           raw    cache/raw_pack/raw.npy (prep/pack_raw.py): 5 x 960 samples (32 Hz BVP, ACC x/y/z, missing flag)
           time   hours since the start of the recording, scaled as in minute_inputs.py
Model  : epoch encoder (strided 1-D CNN on the raw 30 s, mean + SD pooling) -> concat with time of night ->
         residual dilated temporal CNN over epochs (dilations 1-16, ~ +-1 h) -> 2-layer transformer encoder over
         the night -> 5 classes per epoch (Wake, N1, N2, N3, REM).
Labels / splits / loss: reliable-stretch labels (mask_quality), the 5 family folds of baseline_pred_quality_5st.parquet, 20 % of training
         families for early stopping, class-weighted CE. Training on random 8-h crops; validation and
         test on whole nights (up to 16 h).

Env    : RAWNET_FOLD=-1 trains the final model on all families (20 % of families for early stopping, no test set)
         and saves wrist/rawnet_final_quality_5st_raw.pt (weights, input scaling, config).
         RAWNET_FOLD (0-4, or SLURM_ARRAY_TASK_ID), RAWNET_SEED, RAWNET_MAX_EPOCHS (30), RAWNET_BS (8)
Output : wrist/rawnet_pred_quality_5st_raw_fold<k>.parquet (name, epoch, p_*)
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_lgbm import LABELS, NAMES, P, ST_TAG
from minute_inputs import night_arrays, scale, scale_stats

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True
FOLD = int(os.environ.get('RAWNET_FOLD', os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
SEED = int(os.environ.get('RAWNET_SEED', '0'))
MAX_EPOCHS = int(os.environ.get('RAWNET_MAX_EPOCHS', '30'))
INPUT = NAME = 'raw'  # raw pulse + motion + time of night
POS_H = 8  # column of hours since recording start in the minute inputs (minute_inputs.night_arrays `cont`)
BS = int(os.environ.get('RAWNET_BS', '8'))
CROP = 960  # 8 h training crops
PATIENCE = 5
TAG = f'{LABELS}{ST_TAG}_{NAME}_fold{FOLD}' + (f'_seed{SEED}' if SEED else '')


class Encoder(nn.Module):
    """30 s x 5 channels at 32 Hz -> d."""

    def __init__(self, c_in=5, d=64):
        super().__init__()
        layers, c = [], c_in
        for c_out, k in [(32, 9), (48, 9), (64, 7), (96, 7), (128, 5)]:
            layers += [nn.Conv1d(c, c_out, k, stride=2, padding=k // 2), nn.BatchNorm1d(c_out), nn.GELU()]
            c = c_out
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(2 * c, d)

    def forward(self, x):  # [n, 5, 960]
        h = self.conv(x)
        return self.proj(torch.cat([h.mean(-1), h.std(-1)], -1))


class ResTCN(nn.Module):
    def __init__(self, d, dil, p=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 5, padding=2 * dil, dilation=dil)
        self.c2 = nn.Conv1d(d, d, 1)
        self.norm = nn.GroupNorm(1, d)
        self.drop = nn.Dropout(p)

    def forward(self, h):  # [B, d, T]
        return h + self.drop(self.c2(F.gelu(self.norm(self.c1(h)))))


class RawNet(nn.Module):
    def __init__(self, c_min, d=128, n_cls=len(NAMES)):
        super().__init__()
        self.enc = Encoder(c_in=5, d=64)
        self.inp = nn.Linear(64 + 1, d)
        self.tcn = nn.Sequential(*[ResTCN(d, k) for k in [1, 2, 4, 8, 16]])
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, n_cls)

    def forward(self, raw, zmin, pad):  # raw [B, T, 5, 960], zmin [B, T, c], pad [B, T] True = padding
        B, T = raw.shape[:2]
        parts = [self.enc(raw.reshape(B * T, *raw.shape[2:])).reshape(B, T, -1)]
        parts.append(zmin[..., POS_H:POS_H + 1])
        h = self.inp(torch.cat(parts, -1))
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        h = self.tf(h, src_key_padding_mask=pad)
        return self.out(h)


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X, W, Y, L, meta, cont = night_arrays()
    assert cont[POS_H] == 'pos_h', cont
    folds = pd.read_parquet(f'{P}/wrist/baseline_pred_quality_5st.parquet', columns=['name', 'fold']).drop_duplicates('name')
    meta = meta.merge(folds, on='name', how='left')
    idx = pd.read_parquet(f'{P}/cache/raw_pack/index.parquet').set_index('name').loc[meta.name]
    assert (idx.n.to_numpy() == L).all(), 'raw pack and minute arrays disagree on night length'
    rows = idx.row.to_numpy()
    raw = torch.from_numpy(np.load(f'{P}/cache/raw_pack/raw.npy'))  # read fully into RAM (~21 GB)
    print(f'{len(meta)} nights; device {DEV}; input {NAME}; data in {time.time() - t0:.0f}s', flush=True)

    tr = np.where(meta.fold != FOLD)[0]  # FOLD -1: every night
    test = np.where(meta.fold == FOLD)[0]
    fams = meta.family.iloc[tr].drop_duplicates().sample(frac=0.2, random_state=max(FOLD, 0))
    val = tr[meta.family.iloc[tr].isin(fams).to_numpy()]
    fit = np.setdiff1d(tr, val)
    Z = torch.from_numpy(scale(X, W, fit))
    med, iqr = scale_stats(X, fit)
    Yt = torch.from_numpy(Y)
    counts = np.bincount(Y[fit][Y[fit] >= 0], minlength=len(NAMES))
    wts = torch.tensor(counts.sum() / (len(NAMES) * np.maximum(counts, 1)), dtype=torch.float32, device=DEV)

    def batch(b, crop):
        starts = [int(rng.integers(0, max(1, L[i] - CROP + 1))) if crop else 0 for i in b]
        lens = [min(L[i] - s, CROP) if crop else L[i] for i, s in zip(b, starts)]
        T = max(lens)
        r = torch.zeros(len(b), T, 5, 960, dtype=torch.float16)
        z = torch.zeros(len(b), T, Z.shape[2])
        y = torch.full((len(b), T), -1, dtype=torch.long)
        pad = torch.ones(len(b), T, dtype=torch.bool)
        for j, (i, s, n) in enumerate(zip(b, starts, lens)):
            r[j, :n] = raw[rows[i] + s:rows[i] + s + n]
            z[j, :n], y[j, :n], pad[j, :n] = Z[i, s:s + n], Yt[i, s:s + n], False
        to = lambda t: t.to(DEV, non_blocking=True)
        return to(r).float(), to(z), to(y), to(pad), lens

    net = RawNet(Z.shape[2]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS)
    amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=DEV.type == 'cuda')

    def evaluate(ix, keep=False):
        net.eval()
        tot, n, probs = 0.0, 0, {}
        with torch.no_grad():
            for k in range(0, len(ix), max(1, BS // 2)):
                b = ix[k:k + max(1, BS // 2)]
                r, z, y, pad, lens = batch(b, crop=False)
                with torch.autocast(**amp):
                    lo = net(r, z, pad).float()
                m = y >= 0
                tot += F.cross_entropy(lo[m], y[m], weight=wts, reduction='sum').item()
                n += int(m.sum())
                if keep:
                    pr = torch.softmax(lo, -1).cpu().numpy()
                    for j, i in enumerate(b):
                        probs[i] = pr[j, :lens[j]]
        return tot / max(n, 1), probs

    best, best_state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(MAX_EPOCHS):
        net.train()
        te = time.time()
        perm = rng.permutation(fit)
        for k in range(0, len(perm), BS):
            r, z, y, pad, _ = batch(perm[k:k + BS], crop=True)
            m = y >= 0
            if not m.any():
                continue
            with torch.autocast(**amp):
                lo = net(r, z, pad).float()
            loss = F.cross_entropy(lo[m], y[m], weight=wts)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        v, _ = evaluate(val)
        print(f'epoch {ep}: val loss {v:.4f}, {time.time() - te:.0f}s', flush=True)
        if v < best - 1e-4:
            best, bad, best_ep = v, 0, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    if FOLD == -1:
        torch.save(dict(state_dict={k: t.cpu() for k, t in best_state.items()}, names=NAMES, cont=cont, med=med, iqr=iqr,
                        input=INPUT, labels=LABELS, best_epoch=best_ep, val_loss=best, n_fit=len(fit), n_val=len(val)),
                   f'{P}/wrist/rawnet_final_{LABELS}{ST_TAG}_{NAME}.pt')
        print(f'final model: fit {len(fit)} val {len(val)} nights, best epoch {best_ep}, val loss {best:.4f}, '
              f'done in {time.time() - t0:.0f}s', flush=True)
        return
    _, probs = evaluate(test, keep=True)
    out = []
    for i, p in probs.items():
        df = pd.DataFrame(p, columns=[f'p_{c}' for c in NAMES])
        df.insert(0, 'epoch', np.arange(L[i]))
        df.insert(0, 'name', meta.name.iloc[i])
        out.append(df)
    pr = pd.concat(out, ignore_index=True)
    pr['rawnet'] = pr[[f'p_{c}' for c in NAMES]].to_numpy().argmax(1)
    pr.to_parquet(f'{P}/wrist/rawnet_pred_{TAG}.parquet', index=False, compression=None)  # cluster pyarrow: no snappy
    print(f'fold {FOLD}: fit {len(fit)} val {len(val)} test {len(test)} nights, best epoch {best_ep}, '
          f'val loss {best:.4f}, done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
