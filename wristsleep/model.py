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
    """Epoch encoder (raw pulse + motion) + scaled hours since the start of the night -> dilated temporal CNN
    (~+-30 min) -> transformer over the night -> stage logits per epoch. c_raw: 5 raw channels (pulse + motion +
    missing flag) or 2 (pulse only: BVP + its missing flag)."""

    def __init__(self, n_cls=5, d=128, c_raw=5):
        super().__init__()
        self.enc = Encoder(c_in=c_raw, d=64)
        self.inp = nn.Linear(64 + 1, d)
        self.tcn = nn.Sequential(*[ResTCN(d, k) for k in [1, 2, 4, 8, 16]])
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, n_cls)

    def forward(self, raw, ztime, pad):  # raw [B, T, 5, 960], ztime [B, T, 1], pad [B, T] True = padding
        B, T = raw.shape[:2]
        h = self.inp(torch.cat([self.enc(raw.reshape(B * T, *raw.shape[2:])).reshape(B, T, -1), ztime], -1))
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        h = self.tf(h, src_key_padding_mask=pad)
        return self.out(h)
