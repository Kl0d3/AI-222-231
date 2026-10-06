"""Tiny CNN Voice Command Model (VCM).

Designed to run in real-time on a Raspberry Pi 5 (4 GB RAM). Input is a
(1, N_MELS, N_FRAMES) log-mel spectrogram; output is 12 intent logits.
Parameter count is ~180K (~0.7 MB fp32) so it fits comfortably on a Pi and
runs well under 10 ms per command on a Pi 5 CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_INTENTS = 12


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyVCM(nn.Module):
    """3-stage CNN: 32 -> 64 -> 128 channels, global-pool, FC head.

    Input:  (B, 1, N_FRAMES, N_MELS)  e.g. (B, 1, 99, 40)
    Output: (B, N_INTENTS)
    """

    def __init__(self, n_mels=40, n_frames=99, n_intents=N_INTENTS,
                 base=32, dropout=0.4):
        super().__init__()
        assert n_mels % 8 == 0, "n_mels must be divisible by 8 (pooling)"
        self.n_mels = n_mels
        self.n_frames = n_frames
        b = base
        self.stage1 = nn.Sequential(
            ConvBlock(1, b, k=3, s=1, p=1),
            ConvBlock(b, b, k=3, s=1, p=1),
            nn.MaxPool2d(2),
        )                                   # (B,b,F/2,M/2)
        self.stage2 = nn.Sequential(
            ConvBlock(b, 2 * b, k=3, s=1, p=1),
            ConvBlock(2 * b, 2 * b, k=3, s=1, p=1),
            nn.MaxPool2d(2),
        )                                   # (B,2b,F/4,M/4)
        self.stage3 = nn.Sequential(
            ConvBlock(2 * b, 4 * b, k=3, s=1, p=1),
            ConvBlock(4 * b, 4 * b, k=3, s=1, p=1),
            nn.AdaptiveAvgPool2d(1),
        )                                   # (B,4b,1,1)
        feat = 4 * b
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_intents),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
