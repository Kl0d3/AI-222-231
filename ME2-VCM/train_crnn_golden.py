#!/usr/bin/env python3
"""Train CRNN on the ME2 v2 "golden" dataset (airimonda/ai231-me2-voice-commands v2).

Goal: maximum generalization across speakers, accents, phrasings, noise,
and out-of-scope rejection.

Data (16 kHz, parquet with inline audio bytes):
  HF train          10,733  (20 classes; SLURP/SNIPS/FSC/CommonVoice/real_voice + synthetic)
  supp_fil         14,120  (Filipino-accent synthetic voices)
  supp_synth        5,856  (extra synthetic voices)
  neg_train         1,000  (synthetic OUT_OF_SCOPE)
  Mark MEX2         ~1,444 (real human recordings, 12 intents -> HF 20-class map)
  HF test            4,443  (speaker-disjoint; held out)
  neg_test            250  (OOS held out)

Strategy (3 phases):
  Phase 1: Pre-train on HF train only (10% held out for validation)
           - Aggressive SpecAugment + waveform aug (shift/gain/background-noise)
           - Mixup (alpha=0.3), label smoothing 0.15, warmup + cosine LR
  Phase 2: Fine-tune on the FULL golden pool
           (HF train + supp_fil + supp_synth + neg_train + Mark MEX2)
           - Lower LR (1e-4), fewer epochs, SWA over last checkpoints
  Phase 3: Evaluate on HF test + neg_test (OOS) + Mark MEX2, per-intent
           confusion breakdown. Save checkpoint with serving metadata.

Usage:
    python train_crnn_golden.py [--device cuda] [--seed 42]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import random
import sys
import time
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 400
HOP = 160
N_FRAMES = 99

HF_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2/data"
SUPP_FIL_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2/supplemental_fil"
SUPP_SYNTH_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2/supplemental_synth"
NEG_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2/synthetic_negatives"
MARK_DIR = (
    "/home/kent.justin.canja/sandbox/"
    "ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/"
    "ME2_ Voice Controlled Smart Device/vcm_dataset"
)

INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER", "OUT_OF_SCOPE",
]
N_INTENTS = len(INTENTS)
INTENT_TO_IDX = {s: i for i, s in enumerate(INTENTS)}

# Map Mark's 12 intents to HF 20-class labels
MARK_TO_HF = {
    "dim_lights": "BRIGHTNESS",
    "get_time": "TIME",
    "get_weather": "WEATHER",
    "light_off": "LIGHT_OFF",
    "light_on": "LIGHT_ON",
    "make_call": "CALL",
    "manage_reminders": "CREATE_REMINDER",
    "media_control": "PAUSE",
    "play_music": "PLAY_MUSIC",
    "set_alarm": "ALARM",
    "set_temperature": "TEMPERATURE",
    "set_timer": "TIMER",
}


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------
def load_audio_from_bytes(data: bytes) -> np.ndarray:
    """Load audio from raw bytes, resample to 16 kHz mono."""
    from scipy.signal import resample_poly
    import math
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return x


def load_wav(path: str) -> np.ndarray:
    """Load wav file as float32 mono @ 16 kHz."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return x


def _mel_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filterbank (no external deps)."""
    def hz_to_mel(h):
        return 2595.0 * np.log10(1.0 + h / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low, high = 0.0, sr / 2.0
    mel_low, mel_high = hz_to_mel(low), hz_to_mel(high)
    pts = mel_to_hz(np.linspace(mel_low, mel_high, n_filters + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft)
    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_filters):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_MEL_FB = None
def get_mel_fb() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE)
    return _MEL_FB


def compute_mel(x: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram: (N_FRAMES, N_MELS).

    Matches vcm/features.py exactly: STFT over the WHOLE (onset-trimmed)
    clip, then crop the first N_FRAMES frames (or right-pad if shorter).
    power = |FFT|^2 + 1e-10 (inside the log), then log(mel + 1e-5).
    """
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    win = np.hanning(N_FFT).astype(np.float32)
    hop = HOP
    n_frames = 1 + (len(x) - N_FFT) // hop
    if n_frames <= 0:
        x = np.pad(x, (0, N_FFT))
        n_frames = 1

    shape = (n_frames, N_FFT)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    frames = frames * win
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2) + 1e-10

    # Mel filterbank
    mel_fb = get_mel_fb()
    mel_spec = power @ mel_fb.T  # (T, N_MELS)
    # Log (same eps as serving)
    mel_spec = np.log(mel_spec + 1e-5)
    # Crop / pad to N_FRAMES (onset-aligned: take the FIRST frames)
    if mel_spec.shape[0] > N_FRAMES:
        mel_spec = mel_spec[:N_FRAMES]
    elif mel_spec.shape[0] < N_FRAMES:
        mel_spec = np.pad(mel_spec, ((0, N_FRAMES - mel_spec.shape[0]), (0, 0)),
                          mode="constant")
    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
    return mel_spec.astype(np.float32)


def detect_onset(x: np.ndarray, sr: int = SAMPLE_RATE) -> int:
    """Detect speech onset, skipping initial clicks/transients.

    Mirrors vcm/features.py::_find_onset (the serving pipeline) so training
    and inference crop the waveform identically.
    """
    hop = 160  # 10 ms
    frame_len = hop
    n_frames = len(x) // frame_len
    if n_frames < 5:
        return 0

    energies = []
    for i in range(n_frames):
        chunk = x[i * frame_len:(i + 1) * frame_len]
        energies.append(np.sqrt(np.mean(chunk ** 2)))
    energies = np.array(energies)

    # Detect initial transient (click)
    first_100ms = energies[:10]
    rest = energies[10:] if len(energies) > 10 else first_100ms
    first_rms = first_100ms.mean()
    rest_rms = rest.mean()
    start_search = 0
    if rest_rms > 1e-6 and first_rms > 2.0 * rest_rms:
        start_search = 10  # skip the click

    # Adaptive threshold: 12% of peak, but at least an absolute floor
    peak = energies.max()
    threshold = max(peak * 0.12, 0.003)
    above = energies >= threshold

    # Find the first SUSTAINED region: >= 5 consecutive hops above threshold
    min_sustained = 5  # 50 ms of continuous energy = speech, not a click
    onset_hop = -1
    run = 0
    for i in range(start_search, len(above)):
        if above[i]:
            run += 1
            if run >= min_sustained:
                onset_hop = i - min_sustained + 1  # start of the run
                break
        else:
            run = 0

    if onset_hop < 0:
        # No sustained region found - fall back to first above-threshold hop
        first_above = np.where(above[start_search:])[0]
        if len(first_above) == 0:
            return 0
        onset_hop = start_search + first_above[0]

    # Go back one hop to catch the attack transient
    onset_sample = max(0, (onset_hop - 1) * hop)
    return onset_sample


def preprocess(x: np.ndarray) -> np.ndarray:
    """Full preprocessing: onset crop, compute mel.

    Matches vcm/features.py::preprocess (the serving pipeline) exactly:
    onset-aligned crop with 40 ms pre-roll, NO waveform normalization
    (the mel is mean/std-normalized instead), full-clip STFT then crop.
    """
    # Onset detection (only if clip is longer than the target window)
    onset = 0
    if len(x) > N_FRAMES * HOP:
        onset = detect_onset(x)
    # Trim leading silence (keep 40 ms pre-roll for natural attack)
    pre_roll = min(40 * (SAMPLE_RATE // 1000), onset)  # 40 ms
    x = x[max(0, onset - pre_roll):]

    # Compute mel (no waveform scaling — identical to serving)
    mel = compute_mel(x)
    return mel  # (N_FRAMES, N_MELS)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def spec_augment_strong(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aggressive SpecAugment: 4 time masks (up to 7 frames), 3 freq masks (up to 25 mels)."""
    img = img.copy()
    h, w = img.shape  # (99, 40)
    # Time masks: 4 masks, up to 7 frames
    for _ in range(4):
        t0 = int(rng.integers(0, h))
        tw = int(rng.integers(1, 8))
        img[t0:min(h, t0 + tw), :] = 0.0
    # Freq masks: 3 masks, up to 25 mels
    for _ in range(3):
        f0 = int(rng.integers(0, w))
        fw = int(rng.integers(1, 26))
        img[:, f0:min(w, f0 + fw)] = 0.0
    return img


def augment_waveform(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Waveform-level augmentation: time shift + gain + background noise."""
    # Random time shift (±20 ms)
    shift = int(rng.integers(-int(0.020 * SAMPLE_RATE), int(0.020 * SAMPLE_RATE)))
    if shift > 0:
        x = np.concatenate([np.zeros(shift, np.float32), x[:-shift]])
    elif shift < 0:
        x = x[-shift:]
        x = np.concatenate([x, np.zeros(-shift, np.float32)])
    # Random gain (±6 dB)
    x = x * float(10 ** (rng.uniform(-6.0, 6.0) / 20.0))
    # Background noise (60% of clips): pink-ish noise at SNR 15-40 dB
    if rng.random() < 0.6:
        snr_db = float(rng.uniform(15.0, 40.0))
        sig_power = np.mean(x ** 2) + 1e-12
        noise = rng.standard_normal(len(x)).astype(np.float32)
        # crude low-pass to make it less harsh
        kernel = np.ones(4) / 4.0
        noise = np.convolve(noise, kernel, mode="same")
        noise *= np.sqrt(sig_power / (np.mean(noise ** 2) + 1e-12) / 10 ** (snr_db / 10.0))
        x = x + noise
    return x


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.3, n_cls: int = 20):
    """Apply mixup to a batch. Returns mixed x, soft probability targets."""
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    # Soft probability targets
    soft_a = torch.zeros(batch_size, n_cls, device=x.device)
    soft_b = torch.zeros(batch_size, n_cls, device=x.device)
    soft_a.scatter_(1, y.unsqueeze(1), 1.0)
    soft_b.scatter_(1, y[index].unsqueeze(1), 1.0)
    mixed_y = lam * soft_a + (1 - lam) * soft_b
    return mixed_x, mixed_y


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class HFDataset(Dataset):
    """HF AI231-ME2 parquet dataset."""

    def __init__(self, parquet_files: List[str], augment: bool = False,
                 seed: int = 0):
        self.items = []
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                audio_data = row["audio"]["bytes"]
                command = str(row["command"]).strip().upper()
                if command not in INTENT_TO_IDX:
                    continue
                self.items.append((audio_data, INTENT_TO_IDX[command]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_data, label = self.items[idx]
        x = load_audio_from_bytes(audio_data)

        if self.augment:
            x = augment_waveform(x, self.rng)

        mel = preprocess(x)  # (N_FRAMES, N_MELS) = (99, 40)

        if self.augment:
            mel = spec_augment_strong(mel, self.rng)

        # Reshape to (1, N_FRAMES, N_MELS) for Conv2d
        tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, 99, 40)
        return tensor, label


class MarkDataset(Dataset):
    """Mark's 1,444 clips mapped to HF 20-class labels."""

    def __init__(self, augment: bool = False, seed: int = 0):
        self.items = []
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        meta = pd.read_csv(os.path.join(MARK_DIR, "metadata.csv"))
        for _, row in meta.iterrows():
            fname = row["file_name"]
            intent_mark = str(row["intent"]).strip()
            if intent_mark not in MARK_TO_HF:
                continue
            hf_label = MARK_TO_HF[intent_mark]
            if hf_label not in INTENT_TO_IDX:
                continue
            path = os.path.join(MARK_DIR, fname)
            if not os.path.isfile(path):
                path = os.path.join(MARK_DIR, "real_voice_audio", fname)
                if not os.path.isfile(path):
                    continue
            self.items.append((path, INTENT_TO_IDX[hf_label]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x = load_wav(path)

        if self.augment:
            x = augment_waveform(x, self.rng)

        mel = preprocess(x)  # (N_FRAMES, N_MELS) = (99, 40)

        if self.augment:
            mel = spec_augment_strong(mel, self.rng)

        tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, 99, 40)
        return tensor, label


class MixedDataset(Dataset):
    """Concatenate any number of datasets (HF + supplemental + Mark...)."""

    def __init__(self, *datasets):
        self.datasets = list(datasets)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        for d in self.datasets:
            if idx < len(d):
                return d[idx]
            idx -= len(d)
        raise IndexError(idx)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CRNN(nn.Module):
    """CNN encoder -> BiGRU temporal model -> FC head."""

    def __init__(self, n_mels=40, n_frames=99, n_intents=20,
                 base=32, rnn_hidden=128, dropout=0.4):
        super().__init__()
        assert n_mels % 8 == 0
        self.n_mels = n_mels
        self.n_frames = n_frames
        b = base
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
        self.tf = n_frames // 4
        self.tm = n_mels // 4
        feat_dim = 2 * b * self.tm

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
        h = self.cnn(x)
        B, C, tf, tm = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, tf, C * tm)
        h, _ = self.rnn(h)
        h = h[:, -1, :]
        return self.head(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def swa_average(model: nn.Module, states: List[dict], device: torch.device):
    """Average model weights from multiple checkpoint states."""
    avg_state = {}
    for state in states:
        for k, v in state.items():
            if k not in avg_state:
                avg_state[k] = v.clone().float()
            else:
                avg_state[k] += v.float()
    for k in avg_state:
        avg_state[k] /= len(states)
    model.load_state_dict(avg_state)


def evaluate(model, loader, device, tag=""):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / max(total, 1)
    if tag:
        print(f"  [{tag}] acc={acc:.4f} ({correct}/{total})")
    return acc, total


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup:
            frac = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            frac = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * frac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-epochs", type=int, default=25)
    ap.add_argument("--phase2-epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-phase2", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.15)
    ap.add_argument("--mixup-alpha", type=float, default=0.3)
    ap.add_argument("--swa-window", type=int, default=7)
    ap.add_argument("--grad-clip", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="models/crnn_hf_20_golden.pth")
    args = ap.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Params: {count_params(CRNN()):,}")

    # Load data
    print("\nLoading HF v2 golden data...")
    train_files = sorted(glob.glob(os.path.join(HF_DIR, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(HF_DIR, "test-*.parquet")))
    supp_fil_files = sorted(glob.glob(os.path.join(SUPP_FIL_DIR, "*.parquet")))
    supp_synth_files = sorted(glob.glob(os.path.join(SUPP_SYNTH_DIR, "*.parquet")))
    neg_train_files = sorted(glob.glob(os.path.join(NEG_DIR, "train-*.parquet")))
    neg_test_files = sorted(glob.glob(os.path.join(NEG_DIR, "test-*.parquet")))
    print(f"  HF train parquets: {len(train_files)}, test: {len(test_files)}")
    print(f"  supp_fil: {len(supp_fil_files)}, supp_synth: {len(supp_synth_files)}, "
          f"neg_train: {len(neg_train_files)}, neg_test: {len(neg_test_files)}")

    # Phase 1: HF only (with val split)
    train_ds_p1 = HFDataset(train_files, augment=True, seed=args.seed)
    # Hold out 10% for validation
    n_val = int(0.10 * len(train_ds_p1))
    n_train = len(train_ds_p1) - n_val
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(train_ds_p1)), [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Create subset datasets
    class SubsetDS(Dataset):
        def __init__(self, ds, indices):
            self.ds = ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.ds[self.indices[i]]

    train_sub = SubsetDS(train_ds_p1, train_idx)
    val_sub = SubsetDS(train_ds_p1, val_idx)

    train_loader = DataLoader(train_sub, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"  Phase 1 train: {len(train_sub)}, val: {len(val_sub)}")

    # Phase 2: FULL golden pool (HF train + Filipino + synthetic + OOS negatives + Mark)
    print("\nLoading supplemental + Mark data...")
    supp_fil_ds = HFDataset(supp_fil_files, augment=True, seed=args.seed + 1)
    supp_synth_ds = HFDataset(supp_synth_files, augment=True, seed=args.seed + 2)
    neg_train_ds = HFDataset(neg_train_files, augment=True, seed=args.seed + 3)
    mark_ds = MarkDataset(augment=True, seed=args.seed)
    print(f"  supp_fil: {len(supp_fil_ds)}, supp_synth: {len(supp_synth_ds)}, "
          f"neg_train: {len(neg_train_ds)}, Mark: {len(mark_ds)}")

    p2_train_ds = MixedDataset(train_ds_p1, supp_fil_ds, supp_synth_ds,
                               neg_train_ds, mark_ds)
    p2_train_loader = DataLoader(p2_train_ds, batch_size=args.batch, shuffle=True,
                                 num_workers=args.workers, pin_memory=True)
    print(f"  Phase 2 train: {len(p2_train_ds)}")

    # Load HF test
    test_ds = HFDataset(test_files, augment=False, seed=args.seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    print(f"  HF test: {len(test_ds)}")

    # OOS test (held out)
    neg_test_ds = HFDataset(neg_test_files, augment=False, seed=args.seed)
    neg_test_loader = DataLoader(neg_test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    print(f"  OOS test: {len(neg_test_ds)}")

    # Mark validation (held out)
    mark_val_ds = MarkDataset(augment=False, seed=args.seed)
    mark_val_loader = DataLoader(mark_val_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    # Create model
    model = CRNN(n_mels=N_MELS, n_frames=N_FRAMES, n_intents=N_INTENTS)
    model = model.to(device)
    print(f"\nModel: {count_params(model):,} params")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ===================================================================
    # PHASE 1: Pre-train on HF only
    # ===================================================================
    print(f"\n{'=' * 64}")
    print(f"  PHASE 1: Pre-train on HF ({args.phase1_epochs} epochs, lr={args.lr})")
    print(f"{'=' * 64}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.phase1_epochs)

    best_val = 0.0
    best_epoch = 0
    patience_counter = 0
    swa_states = []
    history = []

    for epoch in range(1, args.phase1_epochs + 1):
        scheduler.step(epoch - 1)
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Mixup
            x, soft_y = mixup_batch(x, y, args.mixup_alpha, N_INTENTS)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.kl_div(
                F.log_softmax(logits, dim=1), soft_y, reduction="batchmean"
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validate
        val_acc, _ = evaluate(model, val_loader, device, tag=f"P1 ep{epoch}")

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  ep{epoch:2d} | loss={train_loss:.4f} acc={train_acc:.4f} "
              f"val={val_acc:.4f} lr={lr_now:.2e} time={elapsed:.0f}s")

        history.append({
            "phase": 1, "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "lr": lr_now,
        })

        # Track best
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "val_acc": val_acc, "phase": 1,
            }, args.out)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # SWA tracking
        if epoch > args.phase1_epochs - args.swa_window:
            swa_states.append({k: v.clone() for k, v in model.state_dict().items()})

    # Apply SWA
    if swa_states:
        print(f"\n  Applying SWA (averaging {len(swa_states)} checkpoints)...")
        swa_average(model, swa_states, device)
        swa_val_acc, _ = evaluate(model, val_loader, device, tag="P1 SWA")
        print(f"  SWA val_acc={swa_val_acc:.4f} (best single={best_val:.4f})")
        if swa_val_acc > best_val:
            best_val = swa_val_acc
            torch.save({
                "model": model.state_dict(), "epoch": best_epoch,
                "val_acc": swa_val_acc, "phase": 1, "swa": True,
            }, args.out)
            print(f"  ★ SWA model is better, saved.")

    # ===================================================================
    # PHASE 2: Fine-tune on FULL golden pool
    # ===================================================================
    print(f"\n{'=' * 64}")
    print(f"  PHASE 2: Fine-tune FULL golden pool ({args.phase2_epochs} epochs, lr={args.lr_phase2})")
    print(f"{'=' * 64}")

    print(f"  Pool: {len(p2_train_ds)} "
          f"(HF {len(train_ds_p1)} + fil {len(supp_fil_ds)} + synth {len(supp_synth_ds)} "
          f"+ OOS {len(neg_train_ds)} + Mark {len(mark_ds)})")

    # Reset optimizer with lower LR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr_phase2, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineScheduler(optimizer, 2, args.phase2_epochs)

    best_val_p2 = 0.0
    best_epoch_p2 = 0
    patience_counter = 0
    swa_states_p2 = []

    for epoch in range(1, args.phase2_epochs + 1):
        scheduler.step(epoch - 1)
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        t0 = time.time()
        for x, y in p2_train_loader:
            x = x.to(device)
            y = y.to(device)

            x, soft_y = mixup_batch(x, y, args.mixup_alpha, N_INTENTS)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.kl_div(
                F.log_softmax(logits, dim=1), soft_y, reduction="batchmean"
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validate on HF val
        val_acc, _ = evaluate(model, val_loader, device, tag=f"P2 ep{epoch}")

        # Also track Mark accuracy
        mark_acc, _ = evaluate(model, mark_val_loader, device, tag=f"P2 ep{epoch} Mark")

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  ep{epoch:2d} | loss={train_loss:.4f} acc={train_acc:.4f} "
              f"val={val_acc:.4f} mark={mark_acc:.4f} lr={lr_now:.2e} time={elapsed:.0f}s")

        history.append({
            "phase": 2, "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "mark_acc": mark_acc, "lr": lr_now,
        })

        if val_acc > best_val_p2:
            best_val_p2 = val_acc
            best_epoch_p2 = epoch
            patience_counter = 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "val_acc": val_acc, "phase": 2,
            }, args.out)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch > args.phase2_epochs - args.swa_window:
            swa_states_p2.append({k: v.clone() for k, v in model.state_dict().items()})

    # Apply SWA for Phase 2
    if swa_states_p2:
        print(f"\n  Applying SWA P2 (averaging {len(swa_states_p2)} checkpoints)...")
        swa_average(model, swa_states_p2, device)
        swa_val_acc, _ = evaluate(model, val_loader, device, tag="P2 SWA")
        swa_mark_acc, _ = evaluate(model, mark_val_loader, device, tag="P2 SWA Mark")
        print(f"  SWA val_acc={swa_val_acc:.4f} mark_acc={swa_mark_acc:.4f}")
        if swa_val_acc > best_val_p2:
            best_val_p2 = swa_val_acc
            torch.save({
                "model": model.state_dict(), "epoch": best_epoch_p2,
                "val_acc": swa_val_acc, "phase": 2, "swa": True,
            }, args.out)
            print(f"  ★ SWA P2 model is better, saved.")

    # ===================================================================
    # PHASE 3: Final evaluation
    # ===================================================================
    print(f"\n{'=' * 64}")
    print(f"  PHASE 3: Final Evaluation")
    print(f"{'=' * 64}")

    # Load best model
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])

    # HF Test
    hf_test_acc, hf_test_total = evaluate(model, test_loader, device, tag="HF TEST")

    # OOS Test (should be rejected as OUT_OF_SCOPE)
    oos_acc, oos_total = evaluate(model, neg_test_loader, device, tag="OOS TEST")

    # Mark Validation
    mark_final_acc, mark_total = evaluate(model, mark_val_loader, device, tag="MARK VAL")

    # Per-intent HF test breakdown + confusion
    print("\n  HF test per-intent:")
    hf_correct = {INTENTS[i]: [0, 0] for i in range(N_INTENTS)}
    confusion = {}
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            for i in range(y.size(0)):
                ti = INTENTS[y[i].item()]
                pi = INTENTS[preds[i].item()]
                hf_correct[ti][1] += 1
                if pi == ti:
                    hf_correct[ti][0] += 1
                else:
                    confusion.setdefault(ti, {}).setdefault(pi, 0)
                    confusion[ti][pi] += 1

    hf_per_intent = {}
    for intent in sorted(hf_correct.keys()):
        c, t = hf_correct[intent]
        if t > 0:
            hf_per_intent[intent] = [c, t]
            print(f"    {intent:20s}: {c}/{t} ({100*c/t:.1f}%)")

    # Per-intent Mark breakdown
    print("\n  Mark per-intent:")
    mark_correct = {INTENTS[i]: [0, 0] for i in range(N_INTENTS)}
    model.eval()
    with torch.no_grad():
        for x, y in mark_val_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            for i in range(y.size(0)):
                intent = INTENTS[y[i].item()]
                mark_correct[intent][1] += 1
                if preds[i].item() == y[i].item():
                    mark_correct[intent][0] += 1

    mark_per_intent = {}
    for intent in sorted(mark_correct.keys()):
        c, t = mark_correct[intent]
        if t > 0:
            mark_per_intent[intent] = [c, t]
            print(f"    {intent:20s}: {c}/{t} ({100*c/t:.1f}%)")

    # Save eval results
    eval_result = {
        "config": vars(args),
        "n_params": count_params(model),
        "phase1_best_val": best_val,
        "phase1_best_epoch": best_epoch,
        "phase2_best_val": best_val_p2,
        "phase2_best_epoch": best_epoch_p2,
        "hf_test_acc": hf_test_acc,
        "oos_test_acc": oos_acc,
        "mark_validation_acc": mark_final_acc,
        "hf_per_intent": hf_per_intent,
        "mark_per_intent": mark_per_intent,
        "confusion_top": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:5])
                          for k, v in confusion.items()},
        "history": history,
    }
    eval_path = args.out.replace(".pth", "_eval.json")
    with open(eval_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"\n  Eval saved -> {eval_path}")

    # Add metadata to checkpoint
    ckpt["intents"] = INTENTS
    ckpt["n_mels"] = N_MELS
    ckpt["n_frames"] = N_FRAMES
    ckpt["hf_test_acc"] = hf_test_acc
    ckpt["oos_test_acc"] = oos_acc
    ckpt["mark_validation_acc"] = mark_final_acc
    ckpt["dataset"] = "airimonda/ai231-me2-voice-commands v2 (golden)"
    torch.save(ckpt, args.out)
    print(f"  Checkpoint updated -> {args.out}")

    print(f"\n{'=' * 64}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 64}")
    print(f"  HF Test Accuracy:    {hf_test_acc:.4f} ({hf_test_total} clips)")
    print(f"  OOS Test Accuracy:   {oos_acc:.4f} ({oos_total} clips)")
    print(f"  Mark Validation:     {mark_final_acc:.4f} ({mark_total} clips)")
    print(f"  Phase 1 Best Val:    {best_val:.4f} (epoch {best_epoch})")
    print(f"  Phase 2 Best Val:    {best_val_p2:.4f} (epoch {best_epoch_p2})")
    print(f"  Model:               {count_params(model):,} params")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
