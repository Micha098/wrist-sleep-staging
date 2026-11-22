"""The raw-signal network (identical layers and names to training/rawnet.py, so trained weights load directly)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """One 30-s epoch, 5 channels x 960 samples at 32 Hz -> d features."""

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
    """Epoch encoder (raw pulse + motion) + per-epoch inputs -> dilated temporal CNN (~+-30 min) -> transformer over the
    night -> stage logits per epoch. Per-epoch inputs: the scaled hours since the start of the night only (c_min=1,
    the released model), or all 19 minute inputs (c_min=19, training option RAWNET_INPUT=both). c_raw: 5 raw channels
    (pulse + motion + missing flag) or 4 (motion only: x/y/z + missing flag)."""

    def __init__(self, c_min=19, n_cls=4, d=128, use_raw=True, use_minute=True, c_raw=5):
        super().__init__()
        self.use_raw, self.use_minute = use_raw, use_minute
        self.enc = Encoder(c_in=c_raw, d=64) if use_raw else None
        self.inp = nn.Linear((64 if use_raw else 0) + (c_min if use_minute else 0), d)
        self.tcn = nn.Sequential(*[ResTCN(d, k) for k in [1, 2, 4, 8, 16]])
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, n_cls)

    def forward(self, raw, zmin, pad):  # raw [B, T, 5, 960], zmin [B, T, c], pad [B, T] True = padding
        parts = []
        if self.use_raw:
            B, T = raw.shape[:2]
            parts.append(self.enc(raw.reshape(B * T, *raw.shape[2:])).reshape(B, T, -1))
        if self.use_minute:
            parts.append(zmin)
        h = self.inp(torch.cat(parts, -1))
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        h = self.tf(h, src_key_padding_mask=pad)
        return self.out(h)
