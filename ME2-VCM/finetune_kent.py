"""Fine-tune the golden CRNN-HF-20 on Kent's real voice recordings.

Why: the golden model (models/crnn_hf_20.pth) scored ~41% on Kent's real
voice because his accent + speaking pace are out-of-distribution vs the
synthetic/Mark training data. This script loads the golden checkpoint and
continues training on a small pool:

  - Kent's real clips (real_data/, 16 kHz mono)
  - Mark MEX2 clips (kept in the pool to reduce catastrophic forgetting)
  - HF train parquets (kept in the pool, downsampled — forgetting regularizer)

Kent clips get aggressive augmentation (time shift, gain, noise, time-stretch)
so the model generalizes across his pace rather than memorizing.

Validation: 20% of Kent's clips (speaker = Kent, so this measures exactly
what we care about). The rest of the pool is train-only.

Output: models/crnn_hf20_kent.pth (keeps the golden ckpt untouched).

Usage:
  python finetune_kent.py --epochs 12 --device cuda
"""
from __future__ import annotations

import argparse
import glob
import io
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (identical to train_crnn_golden.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 400
HOP = 160
N_FRAMES = 99

HERE = os.path.dirname(os.path.abspath(__file__))
KENT_DIR = "/home/kent.justin.canja/sandbox/vcm/real_data"  # recording UI save dir (organized by intent)
KENT_META = os.path.join(KENT_DIR, "real_metadata.csv")
MARK_DIR = (
    "/home/kent.justin.canja/sandbox/"
    "ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/"
    "ME2_ Voice Controlled Smart Device/vcm_dataset"
)
HF_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2/data"
CKPT_GOLDEN = os.environ.get(
    "FINETUNE_SRC",
    os.path.join(HERE, "models", "crnn_hf_20_golden.pth"),  # the SERVED checkpoint
)
CKPT_OUT = os.path.join(HERE, "models", "crnn_hf20_kent.pth")

INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER", "OUT_OF_SCOPE",
]
N_INTENTS = len(INTENTS)
INTENT_TO_IDX = {s: i for i, s in enumerate(INTENTS)}

# Kent's 12 device classes -> the HF class(es) that count as correct.
KENT_TO_HF = {
    "dim_lights":       {"BRIGHTNESS"},
    "get_time":         {"TIME"},
    "get_weather":      {"WEATHER"},
    "light_off":        {"LIGHT_OFF"},
    "light_on":         {"LIGHT_ON"},
    "make_call":        {"CALL"},
    "manage_reminders": {"CREATE_REMINDER", "LIST_REMINDERS"},
    "media_control":    {"PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN"},
    "play_music":       {"PLAY_MUSIC"},
    "set_alarm":        {"ALARM"},
    "set_temperature":  {"TEMPERATURE"},
    "set_timer":        {"TIMER"},
}
# Single canonical label per Kent class (training target).
KENT_CANON = {
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

# Mark's 12 intents -> HF 20-class labels (same as train_crnn_golden.py)
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
# Audio utilities (match train_crnn_golden.py)
# ---------------------------------------------------------------------------
def load_wav(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return x


def _mel_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low, high = 20.0, sr / 2.0
    mels = np.linspace(hz2mel(low), hz2mel(high), n_filters + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
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
    """Log-mel spectrogram (N_FRAMES, N_MELS). Matches train_crnn_golden.py."""
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    win = np.hanning(N_FFT).astype(np.float32)
    hop = HOP
    n_frames = 1 + (len(x) - N_FFT) // hop
    if n_frames <= 0:
        x = np.pad(x, (0, N_FFT))
        n_frames = 1
    sig = x[: N_FFT + (n_frames - 1) * hop]
    frames = np.array(
        [sig[i * hop: i * hop + N_FFT] * win for i in range(n_frames)],
        dtype=np.float32,
    )
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-10
    mel = spec @ get_mel_fb().T
    mel = np.log(mel + 1e-5).astype(np.float32)
    if mel.shape[0] < N_FRAMES:
        mel = np.pad(mel, ((0, N_FRAMES - mel.shape[0]), (0, 0)))
    else:
        mel = mel[:N_FRAMES]
    return mel


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def augment_waveform(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shift = int(rng.integers(-int(0.020 * SAMPLE_RATE), int(0.020 * SAMPLE_RATE)))
    if shift > 0:
        x = np.concatenate([np.zeros(shift, np.float32), x[:-shift]])
    elif shift < 0:
        x = x[-shift:]
        x = np.concatenate([x, np.zeros(-shift, np.float32)])
    x = x * float(10 ** (rng.uniform(-6.0, 6.0) / 20.0))
    if rng.random() < 0.6:
        snr_db = float(rng.uniform(15.0, 40.0))
        sig_power = np.mean(x ** 2) + 1e-12
        noise = rng.standard_normal(len(x)).astype(np.float32)
        kernel = np.ones(4) / 4.0
        noise = np.convolve(noise, kernel, mode="same")
        noise *= np.sqrt(sig_power / (np.mean(noise ** 2) + 1e-12) / 10 ** (snr_db / 10.0))
        x = x + noise
    return x


def time_stretch(x: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3:
        return x
    num, den = 100, int(round(100.0 / factor))
    g = math.gcd(num, den)
    x = resample_poly(x, num // g, den // g)
    target = len(x)
    if len(x) < target:
        x = np.pad(x, (0, target - len(x)))
    else:
        x = x[:target]
    return x.astype(np.float32)


def augment_kent(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        x = time_stretch(x, float(rng.uniform(0.85, 1.15)))
    return augment_waveform(x, rng)


# ---------------------------------------------------------------------------
# Model — MUST match the golden checkpoint's param names exactly.
# The golden ckpt uses a Block(conv, bn, relu) wrapper: cnn.0.conv / cnn.0.bn
# ---------------------------------------------------------------------------
class CRNN(nn.Module):
    """FLAT Sequential CRNN — must match vcm/crnn_model.py CRNN exactly
    (this is the architecture the SERVER serves with crnn_hf_20_golden.pth)."""

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
        # x: (B, T, M)
        x = x.unsqueeze(1)  # (B, 1, T, M)
        h = self.cnn(x)
        B, C, tf, tm = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, tf, C * tm)
        h, _ = self.rnn(h)
        h = h[:, -1, :]
        return self.head(h)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class KentDataset(Dataset):
    def __init__(self, items, augment: bool = False, seed: int = 0):
        self.items = items
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, kent_class = self.items[idx]
        x = load_wav(path)
        if self.augment:
            x = augment_kent(x, self.rng)
        mel = compute_mel(x)
        label = INTENT_TO_IDX[KENT_CANON[kent_class]]
        return torch.from_numpy(mel), label


class MarkDataset(Dataset):
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
        mel = compute_mel(x)
        return torch.from_numpy(mel), label


class HFPoolDataset(Dataset):
    def __init__(self, parquet_files, max_items: int, augment: bool = True, seed: int = 0):
        import pyarrow.parquet as pq
        self.items = []
        rng = np.random.default_rng(seed)
        for pf in parquet_files:
            t = pq.read_table(pf)
            d = t.to_pydict()
            n = len(d["audio"])
            for i in range(n):
                audio_struct = d["audio"][i]
                audio_bytes = audio_struct["bytes"] if isinstance(audio_struct, dict) else audio_struct
                command = str(d["command"][i]).strip().upper()
                if command not in INTENT_TO_IDX:
                    continue
                self.items.append((pf, i, audio_bytes, INTENT_TO_IDX[command]))
        if len(self.items) > max_items:
            idx = rng.choice(len(self.items), size=max_items, replace=False)
            self.items = [self.items[i] for i in sorted(idx)]
        self.augment = augment
        self.rng = np.random.default_rng(seed + 1000)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pf, i, audio_bytes, label = self.items[idx]
        x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SAMPLE_RATE:
            g = math.gcd(sr, SAMPLE_RATE)
            x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
        if self.augment:
            x = augment_waveform(x, self.rng)
        mel = compute_mel(x)
        return torch.from_numpy(mel), label


class MixedDataset(Dataset):
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


class SubsetDS(Dataset):
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.ds[self.indices[idx]]


# ---------------------------------------------------------------------------
# Scheduler + mixup
# ---------------------------------------------------------------------------
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.warmup = warmup_epochs
        self.total = total_epochs
        super().__init__(optimizer)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup:
            f = (ep + 1) / max(1, self.warmup)
        else:
            p = (ep - self.warmup) / max(1, self.total - self.warmup)
            f = 0.5 * (1 + math.cos(math.pi * p))
        return [base * f for base in self.base_lrs]


def mixup_batch(x, y, alpha=0.3, n_cls=20):
    if alpha > 0 and x.size(0) > 1:
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        ye = torch.zeros(y.size(0), n_cls, device=x.device)
        ye.scatter_(1, y.view(-1, 1), lam)
        ye.scatter_(1, y[idx].view(-1, 1), 1 - lam)
        return x, ye
    return x, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_kent(model, items, device):
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    per_class = {}
    with torch.no_grad():
        for path, kent_class in items:
            x = load_wav(path)
            mel = compute_mel(x)
            t = torch.from_numpy(mel).unsqueeze(0).to(device)  # (1, T, M)
            logits = model(t)
            pred = INTENTS[int(logits.argmax(dim=1).item())]
            ok = pred in KENT_TO_HF[kent_class]
            correct += int(ok)
            total += 1
            pc = per_class.setdefault(kent_class, [0, 0])
            pc[1] += 1
            pc[0] += int(ok)
    if was_training:
        model.train()
    return correct / max(1, total), per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-max", type=int, default=4000)
    ap.add_argument("--mark-frac", type=float, default=0.5)
    ap.add_argument("--out", default=CKPT_OUT)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load golden checkpoint (weights nested under "model") ---
    ckpt = torch.load(CKPT_GOLDEN, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = CRNN(n_mels=N_MELS, n_frames=N_FRAMES, n_intents=N_INTENTS)
    model.load_state_dict(state, strict=True)
    print(f"Loaded golden ckpt: {CKPT_GOLDEN} (strict=True, OK)")
    model = model.to(device)

    # --- Build Kent items ---
    meta = pd.read_csv(KENT_META)
    kent_items = []
    skipped_missing = 0
    for _, row in meta.iterrows():
        fn = row["file_name"]
        klass = str(row["intent"]).strip()
        if klass not in KENT_CANON:
            continue
        path = os.path.join(KENT_DIR, fn)
        if not os.path.isfile(path):
            alt = os.path.join(KENT_DIR, klass, fn)
            if os.path.isfile(alt):
                path = alt
            else:
                skipped_missing += 1
                continue
        kent_items.append((path, klass))
    print(f"Kent clips: {len(kent_items)} (skipped missing: {skipped_missing})")
    assert len(kent_items) >= 20, "Need at least 20 Kent clips"

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(kent_items))
    n_val = max(10, int(0.20 * len(kent_items)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    kent_train_items = [kent_items[i] for i in train_idx]
    kent_val_items = [kent_items[i] for i in val_idx]
    print(f"Kent train: {len(kent_train_items)}, Kent val: {len(kent_val_items)}")

    base_acc, base_per = evaluate_kent(model, kent_val_items, device)
    print(f"\n[baseline] Golden model on Kent val: {base_acc:.1%}")
    for k in sorted(base_per):
        c, t = base_per[k]
        print(f"  {k:18s} {c}/{t}")

    # --- Build train pool ---
    kent_train_ds = KentDataset(kent_train_items, augment=True, seed=args.seed)

    mark_ds_full = MarkDataset(augment=True, seed=args.seed)
    n_mark = int(len(mark_ds_full) * args.mark_frac)
    mark_idx = np.random.default_rng(args.seed + 7).choice(
        len(mark_ds_full), size=n_mark, replace=False)
    mark_ds = SubsetDS(mark_ds_full, mark_idx)
    print(f"Mark clips in pool: {n_mark} (of {len(mark_ds_full)})")

    hf_files = sorted(glob.glob(os.path.join(HF_DIR, "*.parquet")))
    hf_ds = HFPoolDataset(hf_files, max_items=args.hf_max, augment=True, seed=args.seed)
    print(f"HF clips in pool: {len(hf_ds)}")

    pool = MixedDataset(kent_train_ds, mark_ds, hf_ds)
    print(f"Total train pool: {len(pool)} "
          f"(Kent {len(kent_train_ds)}, Mark {len(mark_ds)}, HF {len(hf_ds)})")

    train_loader = DataLoader(pool, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val = base_acc
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'=' * 64}")
    print(f"  FINE-TUNE on Kent voice ({args.epochs} epochs, lr={args.lr})")
    print(f"{'=' * 64}")

    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch - 1)
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for x, y in train_loader:
            x, soft_y = mixup_batch(x, y, args.mixup_alpha, N_INTENTS)
            x = x.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if soft_y.dim() == 2:
                loss = -(soft_y.to(device) * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                loss = criterion(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            tgt = soft_y.argmax(dim=1) if soft_y.dim() == 2 else y.to(device)
            tgt = tgt.to(device)
            total_correct += int((preds == tgt).sum().item())
            total_samples += x.size(0)
        train_acc = total_correct / max(1, total_samples)

        val_acc, _ = evaluate_kent(model, kent_val_items, device)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total_loss/max(1,total_samples):.4f}  "
              f"pool_acc={train_acc:.1%}  KENT_VAL={val_acc:.1%}  lr={lr_now:.2e}")

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "intents": INTENTS,
                "n_intents": N_INTENTS,
                "n_mels": N_MELS,
                "n_frames": N_FRAMES,
                "sample_rate": SAMPLE_RATE,
                "kent_val_acc": best_val,
                "baseline_kent_val_acc": base_acc,
                "epoch": epoch,
                "fine_tuned_from": CKPT_GOLDEN,
            }, args.out)
            print(f"  -> saved best ({val_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  early stop at epoch {epoch}")
                break

    print(f"\nBest Kent val acc: {best_val:.1%} (epoch {best_epoch})  "
          f"[baseline was {base_acc:.1%}]")
    print(f"Saved: {args.out}")

    if os.path.isfile(args.out):
        ck = torch.load(args.out, map_location="cpu")
        model.load_state_dict(ck["model"])
        model.to(device)
        final_acc, final_per = evaluate_kent(model, kent_val_items, device)
        print(f"\nFinal per-class (Kent val, n={len(kent_val_items)}):")
        for k in sorted(final_per):
            c, t = final_per[k]
            print(f"  {k:18s} {c}/{t}  ({c/t:.0%})")


if __name__ == "__main__":
    main()
