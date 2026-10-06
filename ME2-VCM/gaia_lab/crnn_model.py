"""CRNN voice-command model.

Convolutional front-end (log-mel spectrogram) + bidirectional GRU temporal
encoder + classification head. Designed for the AI231/MEX2 19-intent dataset
(100 speakers, clean+noisy). Runs comfortably on a Raspberry Pi 5 CPU.

Input:  (B, 1, N_FRAMES, N_MELS)  e.g. (B, 1, 99, 40)
Output: (B, N_INTENTS)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CRNN(nn.Module):
    """CNN encoder -> BiGRU temporal model -> FC head.

    The conv stages compress the (frames, mels) image; the GRU then models the
    temporal evolution of the command, which is what lets it generalise to
    unseen phrasings/speakers better than a pure CNN.
    """

    def __init__(self, n_mels=40, n_frames=99, n_intents=19,
                 base=32, rnn_hidden=128, dropout=0.4):
        super().__init__()
        assert n_mels % 8 == 0, "n_mels must be divisible by 8 (pooling)"
        self.n_mels = n_mels
        self.n_frames = n_frames
        b = base
        # After two max-pools: frames -> F/4, mels -> M/4
        self.cnn = nn.Sequential(
            nn.Conv2d(1, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(b, 2 * b, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * b), nn.ReLU(inplace=True),
            nn.Conv2d(2 * b, 2 * b, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * b), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # frame dim after pooling
        self.tf = n_frames // 4
        self.tm = n_mels // 4
        feat_dim = 2 * b * self.tm                  # per-frame vector width

        self.rnn = nn.GRU(
            input_size=feat_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        rnn_out = 2 * rnn_hidden
        self.head = nn.Sequential(
            nn.LayerNorm(rnn_out),
            nn.Linear(rnn_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_intents),
        )

    def forward(self, x):
        # x: (B, 1, N_FRAMES, N_MELS)
        h = self.cnn(x)                              # (B, 2b, tf, tm)
        B, C, tf, tm = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, tf, C * tm)  # (B, tf, feat)
        h, _ = self.rnn(h)                           # (B, tf, 2*rnn_hidden)
        h = h[:, -1, :]                              # last timestep
        return self.head(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class CRNNBlock(CRNN):
    """Same topology as CRNN, but the conv stages are wrapped in ConvBlock
    modules so the state_dict keys are ``cnn.N.conv.*`` / ``cnn.N.bn.*``.

    The ``*_gen`` / ``*_maxgen`` checkpoints were saved with this layout, so
    they can't be loaded into the flat ``CRNN``. This subclass produces the
    exact same key names and identical numerics (ConvBlock is just
    act(bn(conv(x))) with the same hyper-params), so it loads those weights
    cleanly and runs them.
    """

    def __init__(self, n_mels=40, n_frames=99, n_intents=19,
                 base=32, rnn_hidden=128, dropout=0.4):
        # Build the parent (flat) first so self.tf/self.tm/feat dims exist,
        # then swap in the block-wrapped CNN.
        super().__init__(n_mels, n_frames, n_intents, base, rnn_hidden, dropout)
        b = base
        self.cnn = nn.Sequential(
            ConvBlock(1, b, k=3, s=1, p=1),
            ConvBlock(b, b, k=3, s=1, p=1),
            nn.MaxPool2d(2),
            ConvBlock(b, 2 * b, k=3, s=1, p=1),
            ConvBlock(2 * b, 2 * b, k=3, s=1, p=1),
            nn.MaxPool2d(2),
        )
