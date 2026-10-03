#!/usr/bin/env python3
"""Check alternative architectures for generalization."""
import torch
import torch.nn as nn
from crnn_model import CRNN, count_params

# Current model
m = CRNN(n_mels=40, n_frames=99, n_intents=20)
print(f"Current CRNN (base=32, rnn=128): {count_params(m):,} params")

# 1. Deeper CNN (no RNN)
class DeepCNN(nn.Module):
    def __init__(self, n_mels=40, n_frames=99, n_intents=20, base=48, dropout=0.4):
        super().__init__()
        b = base
        self.cnn = nn.Sequential(
            nn.Conv2d(1, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(b, 2*b, 3, padding=1), nn.BatchNorm2d(2*b), nn.ReLU(),
            nn.Conv2d(2*b, 2*b, 3, padding=1), nn.BatchNorm2d(2*b), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2*b, 4*b, 3, padding=1), nn.BatchNorm2d(4*b), nn.ReLU(),
            nn.Conv2d(4*b, 4*b, 3, padding=1), nn.BatchNorm2d(4*b), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*b, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_intents),
        )
    def forward(self, x):
        return self.head(self.cnn(x))

dc = DeepCNN()
print(f"DeepCNN (base=48, 6 conv): {count_params(dc):,} params")

# 2. CRNN with wider GRU
class WideCRNN(nn.Module):
    def __init__(self, n_mels=40, n_frames=99, n_intents=20, base=32, rnn_hidden=256, dropout=0.4):
        super().__init__()
        b = base
        self.cnn = nn.Sequential(
            nn.Conv2d(1, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(b, 2*b, 3, padding=1), nn.BatchNorm2d(2*b), nn.ReLU(),
            nn.Conv2d(2*b, 2*b, 3, padding=1), nn.BatchNorm2d(2*b), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.tf = n_frames // 4
        self.tm = n_mels // 4
        feat_dim = 2 * b * self.tm
        self.rnn = nn.GRU(feat_dim, rnn_hidden, num_layers=2, batch_first=True,
                         bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(2*rnn_hidden),
            nn.Linear(2*rnn_hidden, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_intents),
        )
    def forward(self, x):
        h = self.cnn(x)
        B, C, tf, tm = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, tf, C*tm)
        h, _ = self.rnn(h)
        h = h[:, -1, :]
        return self.head(h)

wc = WideCRNN()
print(f"WideCRNN (base=32, rnn=256): {count_params(wc):,} params")

# 3. CRNN with higher base
class BigCRNN(nn.Module):
    def __init__(self, n_mels=40, n_frames=99, n_intents=20, base=48, rnn_hidden=128, dropout=0.4):
        super().__init__()
        b = base
        self.cnn = nn.Sequential(
            nn.Conv2d(1, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(b, 2*b, 3, padding=1), nn.BatchNorm2d(2*b), nn.ReLU(),
            nn.Conv2d(2*b, 2*b, 3, padding=1), nn.BatchNorm2d(2*b), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.tf = n_frames // 4
        self.tm = n_mels // 4
        feat_dim = 2 * b * self.tm
        self.rnn = nn.GRU(feat_dim, rnn_hidden, num_layers=2, batch_first=True,
                         bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(2*rnn_hidden),
            nn.Linear(2*rnn_hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_intents),
        )
    def forward(self, x):
        h = self.cnn(x)
        B, C, tf, tm = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, tf, C*tm)
        h, _ = self.rnn(h)
        h = h[:, -1, :]
        return self.head(h)

bc = BigCRNN()
print(f"BigCRNN (base=48, rnn=128): {count_params(bc):,} params")

# Forward pass timing on CPU (Pi simulation)
import time
x = torch.randn(1, 1, 99, 40)
for name, model in [("Current CRNN", m), ("DeepCNN", dc), ("WideCRNN", wc), ("BigCRNN", bc)]:
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(x)
        t0 = time.time()
        n_iter = 20
        for _ in range(n_iter):
            _ = model(x)
        elapsed = (time.time() - t0) / n_iter * 1000
    print(f"  {name}: {elapsed:.1f} ms/forward (CPU)")
