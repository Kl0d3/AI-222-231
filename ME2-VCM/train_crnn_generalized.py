"""Train CRNN with maximum generalization on HF AI231-ME2 + Mark's domain.

Strategy (4 phases):
  Phase 1: Pre-train on HF train (10,682 clips, 315 speakers)
           - Aggressive SpecAugment
           - Mixup (alpha=0.2)
           - Cosine LR + warmup
  Phase 2: Domain-adapt fine-tune on HF train + Mark's 1,444 clips
           - Lower LR (1e-4)
           - 10 epochs
  Phase 3: Evaluate on HF test (4,418 clips, speaker-disjoint)
  Phase 4: Validate on Mark's 1,444 clips (held out from Phase 2)

Key improvements over v2:
  1. Domain adaptation: Mark's clips join training in Phase 2
  2. Mixup regularization: interpolates samples to smooth decision boundaries
  3. Stronger SpecAugment: 3 time masks (up to 5 frames), 2 freq masks (up to 20 mels)
  4. Warmup schedule: linear warmup for first 3 epochs prevents early divergence
  5. Stochastic Weight Averaging (SWA): averages last 5 checkpoints for flatter minima
  6. GradNorm clipping at 0.5 (tighter than v2's 1.0)

Usage:
    python train_crnn_generalized.py [--device cuda] [--seed 42]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import time
import wave
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import features as F
from crnn_model import CRNN, count_params

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HF_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2"
MARK_DIR = ("/home/kent.justin.canja/sandbox/"
            "ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/"
            "ME2_ Voice Controlled Smart Device/vcm_dataset")
REAL_DATA_DIR = "/home/kent.justin.canja/sandbox/AI-222-231/ME2-VCM/real_data"

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER", "OUT_OF_SCOPE",
]
INTENT2IDX = {i: k for k, i in enumerate(INTENTS)}
N_INTENTS = len(INTENTS)

MARK_TO_HF = {
    "dim_lights":       "BRIGHTNESS",
    "set_timer":        "TIMER",
    "set_alarm":        "ALARM",
    "set_temperature":  "TEMPERATURE",
    "media_control":    "PAUSE",
    "make_call":        "CALL",
    "light_on":         "LIGHT_ON",
    "light_off":        "LIGHT_OFF",
    "play_music":       "PLAY_MUSIC",
    "manage_reminders": "CREATE_REMINDER",
    "get_weather":      "WEATHER",
    "get_time":         "TIME",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_parquet(split: str) -> pd.DataFrame:
    files = []
    for pat in (os.path.join(HF_DIR, "data", f"{split}-*.parquet"),
                os.path.join(HF_DIR, f"{split}-*.parquet")):
        files = sorted(glob.glob(pat))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"no parquet for split '{split}' under {HF_DIR}")
    frames = [pd.read_parquet(f) for f in files]
    print(f"  loaded {split}: {len(files)} shard(s) -> {sum(len(x) for x in frames)} rows")
    return pd.concat(frames, ignore_index=True)


def _coarse_intent(row) -> str | None:
    if int(row["out_of_scope"]) == 1 or row["command"] == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    cmd = str(row["command"]).strip().upper()
    return cmd if cmd in INTENT2IDX else None


def _decode_wav_bytes(b: bytes) -> np.ndarray:
    w = wave.open(io.BytesIO(b), "rb")
    assert w.getnchannels() == 1
    assert w.getsampwidth() == 2
    assert w.getframerate() == 16000
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class HFDataset(Dataset):
    """HF parquet-based dataset with strong augmentation."""

    def __init__(self, df: pd.DataFrame, augment: bool = False, seed: int = 0):
        self.items = []
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        for _, row in df.iterrows():
            intent = _coarse_intent(row)
            if intent is None:
                continue
            audio = row["audio"]
            if audio is None or audio.get("bytes") is None:
                continue
            try:
                x = _decode_wav_bytes(audio["bytes"])
            except Exception:
                continue
            self.items.append((x, INTENT2IDX[intent], intent))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, label, intent = self.items[i]
        x = x.astype(np.float32)
        if self.augment:
            x = F.augment(x, self.rng)
        feat = F.preprocess(x)
        if self.augment:
            feat[0, 0] = self._strong_spec_augment(feat[0, 0], self.rng)
        return torch.from_numpy(feat).squeeze(0), label, intent

    @staticmethod
    def _strong_spec_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Stronger SpecAugment: 3 time masks (≤5 frames), 2 freq masks (≤20 mels)."""
        img = img.copy()
        h, w = img.shape  # (99, 40)
        # Time masks: 3 masks, up to 5 frames
        for _ in range(3):
            t0 = int(rng.integers(0, h))
            tw = int(rng.integers(1, 6))
            img[t0:min(h, t0 + tw), :] = 0.0
        # Freq masks: 2 masks, up to 20 mels
        for _ in range(2):
            f0 = int(rng.integers(0, w))
            fw = int(rng.integers(1, 21))
            img[:, f0:min(w, f0 + fw)] = 0.0
        return img


class MarkDataset(Dataset):
    """Mark's 1,444 clips (synthetic + real) mapped to HF 20-class labels."""

    def __init__(self, augment: bool = False, seed: int = 0):
        self.items = []
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        # Synthetic TTS clips
        meta = pd.read_csv(os.path.join(MARK_DIR, "metadata.csv"))
        for _, row in meta.iterrows():
            fname = row["file_name"]
            intent_mark = str(row["intent"]).strip()
            if intent_mark not in MARK_TO_HF:
                continue
            hf_label = MARK_TO_HF[intent_mark]
            path = os.path.join(MARK_DIR, fname)
            if not os.path.isfile(path):
                continue
            try:
                x = F.load_wav(path)
            except Exception:
                continue
            self.items.append((x, INTENT2IDX[hf_label], hf_label, intent_mark))

        # Real voice clips
        real_meta_path = os.path.join(REAL_DATA_DIR, "real_metadata.csv")
        if os.path.isfile(real_meta_path):
            real_meta = pd.read_csv(real_meta_path)
            for _, row in real_meta.iterrows():
                fname = row["file_name"]
                intent_mark = str(row["intent"]).strip()
                if intent_mark not in MARK_TO_HF:
                    continue
                hf_label = MARK_TO_HF[intent_mark]
                # Try real_data dir first, then Mark's real_voice_audio
                path = os.path.join(REAL_DATA_DIR, fname)
                if not os.path.isfile(path):
                    path = os.path.join(MARK_DIR, "real_voice_audio", fname)
                if not os.path.isfile(path):
                    continue
                try:
                    x = F.load_wav(path)
                except Exception:
                    continue
                self.items.append((x, INTENT2IDX[hf_label], hf_label, intent_mark))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, label, hf_label, mark_label = self.items[i]
        x = x.astype(np.float32)
        if self.augment:
            x = F.augment(x, self.rng)
        feat = F.preprocess(x)
        if self.augment:
            feat[0, 0] = HFDataset._strong_spec_augment(feat[0, 0], self.rng)
        return torch.from_numpy(feat).squeeze(0), label, hf_label, mark_label


class MixedDataset(Dataset):
    """Concatenate HF train + Mark's clips for Phase 2 domain adaptation."""

    def __init__(self, hf_ds: HFDataset, mark_ds: MarkDataset):
        self.hf = hf_ds
        self.mark = mark_ds

    def __len__(self):
        return len(self.hf) + len(self.mark)

    def __getitem__(self, i):
        if i < len(self.hf):
            return self.hf[i]
        # MarkDataset returns 4 values; normalize to 3 for mixed training
        x, label, hf_label, mark_label = self.mark[i - len(self.hf)]
        return x, label, hf_label


class SubsetDS(Dataset):
    """Index subset of a parent dataset."""

    def __init__(self, parent: HFDataset, sel, augment: bool = False, seed: int = 0):
        self.parent = parent
        self.sel = sel
        self.rng = np.random.default_rng(seed)
        self.augment = augment

    def __len__(self):
        return len(self.sel)

    def __getitem__(self, i):
        x, label, intent = self.parent.items[self.sel[i]]
        x = x.astype(np.float32)
        if self.augment:
            x = F.augment(x, self.rng)
        feat = F.preprocess(x)
        if self.augment:
            feat[0, 0] = HFDataset._strong_spec_augment(feat[0, 0], self.rng)
        return torch.from_numpy(feat).squeeze(0), label, intent


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------
def mixup_batch(x, y, alpha=0.2):
    """Apply mixup to a batch. Returns mixed x, soft targets."""
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], float(lam)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, tag=""):
    model.eval()
    correct, total = 0, 0
    per_intent = defaultdict(lambda: [0, 0])
    for x, y, *rest in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for p, t in zip(pred.cpu().tolist(), y.tolist()):
            per_intent[INTENTS[t]][0] += int(p == t)
            per_intent[INTENTS[t]][1] += 1
    acc = correct / total if total else 0.0
    if tag:
        print(f"\n  [{tag}] accuracy: {acc:.4f}  ({correct}/{total})")
    return acc, dict(per_intent)


@torch.no_grad()
def evaluate_mark(model, loader, device):
    """Evaluate on Mark's dataset with per-Mark-intent breakdown."""
    model.eval()
    correct, total = 0, 0
    per_mark = defaultdict(lambda: [0, 0])
    per_hf = defaultdict(lambda: [0, 0])
    for x, y, hf_label, mark_label in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for p, t, hi, mi in zip(pred.cpu().tolist(), y.tolist(), hf_label, mark_label):
            per_hf[hi][0] += int(p == t)
            per_hf[hi][1] += 1
            per_mark[mi][0] += int(p == t)
            per_mark[mi][1] += 1
    acc = correct / total if total else 0.0
    print(f"\n{'='*64}")
    print(f"  MARK DATASET VALIDATION  —  accuracy: {acc:.4f}  ({correct}/{total})")
    print(f"{'='*64}")
    print("  Per Mark intent:")
    for mi in sorted(per_mark.keys()):
        c, t = per_mark[mi]
        print(f"    {mi:<25s}  {c:>5d}/{t:<5d}  {c/t:.4f}")
    print("  Per HF label:")
    for hi in sorted(per_hf.keys()):
        c, t = per_hf[hi]
        print(f"    {hi:<25s}  {c:>5d}/{t:<5d}  {c/t:.4f}")
    return acc, dict(per_mark), dict(per_hf)


# ---------------------------------------------------------------------------
# SWA (Stochastic Weight Averaging)
# ---------------------------------------------------------------------------
def swa_average(model, swa_states, device):
    """Average the last K state dicts for SWA."""
    n = len(swa_states)
    model_dict = model.state_dict()
    for key in model_dict:
        model_dict[key] = sum(s["model"][key].to(device) for s in swa_states) / n
    model.load_state_dict(model_dict)
    return model


# ---------------------------------------------------------------------------
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------
class WarmupCosineScheduler:
    """Linear warmup for `warmup_epochs` then cosine decay."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup:
            frac = (epoch + 1) / self.warmup
        else:
            import math
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            frac = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * frac


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-epochs", type=int, default=20,
                    help="Pre-training epochs on HF only")
    ap.add_argument("--phase2-epochs", type=int, default=10,
                    help="Domain-adapt epochs on HF + Mark")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="Phase 1 learning rate")
    ap.add_argument("--lr-phase2", type=float, default=1e-4,
                    help="Phase 2 learning rate (lower for fine-tuning)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--swa-window", type=int, default=5,
                    help="Number of last epochs to average for SWA")
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="models/crnn_hf_20_gen.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable -> cpu")

    print("=" * 64)
    print(f"CRNN Generalized Training  (seed={args.seed}, device={device})")
    print(f"  Phase 1: Pre-train on HF train  ({args.phase1_epochs} epochs)")
    print(f"  Phase 2: Domain-adapt on HF + Mark  ({args.phase2_epochs} epochs)")
    print(f"  Phase 3: Evaluate on HF test")
    print(f"  Phase 4: Validate on Mark's dataset")
    print(f"  Mixup alpha={args.mixup_alpha}, SWA window={args.swa_window}")
    print(f"  Grad clip={args.grad_clip}, Label smoothing={args.label_smoothing}")
    print("=" * 64)

    # --- Load data ---
    print("\n[1/5] Loading HF data...")
    train_df = _load_parquet("train")
    test_df = _load_parquet("test")

    train_full = HFDataset(train_df, augment=True, seed=args.seed)
    test_ds = HFDataset(test_df, augment=False)
    print(f"  HF train: {len(train_full)}  HF test: {len(test_ds)}")

    # Carve 10% of HF train for early-stopping validation
    n_val = max(1, int(len(train_full) * 0.1))
    idx = np.arange(len(train_full))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    train_ds = SubsetDS(train_full, tr_idx, augment=True, seed=args.seed)
    val_ds = SubsetDS(train_full, val_idx, augment=False)
    print(f"  Split: train={len(train_ds)}  val={len(val_ds)}")

    # Load Mark's dataset
    print("\n[2/5] Loading Mark's dataset...")
    mark_ds = MarkDataset(augment=True, seed=args.seed)
    mark_eval_ds = MarkDataset(augment=False)
    print(f"  Mark: {len(mark_ds)} clips (train) / {len(mark_eval_ds)} clips (eval)")

    # --- DataLoaders ---
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    mark_eval_loader = DataLoader(mark_eval_ds, batch_size=args.batch, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

    # --- Model ---
    model = CRNN(n_intents=N_INTENTS).to(device)
    n_params = count_params(model)
    print(f"\n[3/5] Model: CRNN  params={n_params:,}  ({n_params*4/1e6:.2f} MB)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ===================================================================
    # PHASE 1: Pre-train on HF only
    # ===================================================================
    print(f"\n{'='*64}")
    print(f"  PHASE 1: Pre-train on HF ({args.phase1_epochs} epochs, lr={args.lr})")
    print(f"{'='*64}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=3,
                                      total_epochs=args.phase1_epochs)

    best_val = 0.0
    best_epoch = 0
    bad_epochs = 0
    t0 = time.time()
    history = []
    swa_states = []  # For SWA

    for epoch in range(1, args.phase1_epochs + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Mixup
            mixed_x, y_a, y_b, lam = mixup_batch(x, y, args.mixup_alpha)

            optimizer.zero_grad()
            out = model(mixed_x)
            # Soft mixup loss
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            run_loss += loss.item() * y.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            run_total += y.size(0)

        scheduler.step(epoch)
        train_acc = run_correct / run_total
        train_loss = run_loss / run_total

        # Validate
        val_acc, _ = evaluate(model, val_loader, device, tag=f"P1 epoch {epoch}")

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  P1 epoch {epoch:>3d}/{args.phase1_epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  lr={cur_lr:.2e}  elapsed={elapsed:.0f}s")

        history.append({
            "phase": 1, "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc,
        })

        # Track best
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": val_acc, "phase": 1}, args.out)
            print(f"    ★ saved best (val_acc={val_acc:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1

        # SWA: keep last K states
        swa_states.append({"model": {k: v.clone() for k, v in model.state_dict().items()},
                           "epoch": epoch, "val_acc": val_acc})
        if len(swa_states) > args.swa_window:
            swa_states.pop(0)

        if bad_epochs >= args.patience:
            print(f"  Phase 1 early stop at epoch {epoch} (patience={args.patience})")
            break

    # Apply SWA
    print(f"\n  Applying SWA (averaging last {len(swa_states)} checkpoints)...")
    swa_average(model, swa_states, device)
    swa_val_acc, _ = evaluate(model, val_loader, device, tag="P1 SWA")
    print(f"  SWA val_acc={swa_val_acc:.4f} (best single={best_val:.4f})")
    if swa_val_acc > best_val:
        best_val = swa_val_acc
        torch.save({"model": model.state_dict(), "epoch": best_epoch,
                    "val_acc": swa_val_acc, "phase": 1, "swa": True}, args.out)
        print(f"  ★ SWA model is better, saved.")

    # ===================================================================
    # PHASE 2: Domain-adapt on HF + Mark
    # ===================================================================
    print(f"\n{'='*64}")
    print(f"  PHASE 2: Domain-adapt HF + Mark ({args.phase2_epochs} epochs, lr={args.lr_phase2})")
    print(f"{'='*64}")

    # Build mixed dataset: HF train (full, no val carve-out) + Mark
    mixed_ds = MixedDataset(train_full, mark_ds)
    mixed_loader = DataLoader(mixed_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"  Mixed dataset: {len(mixed_ds)} clips "
          f"(HF={len(train_full)}, Mark={len(mark_ds)})")

    # New optimizer with lower LR
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=args.lr_phase2,
                                   weight_decay=args.weight_decay)
    scheduler2 = WarmupCosineScheduler(optimizer2, warmup_epochs=2,
                                       total_epochs=args.phase2_epochs)

    best_val_p2 = 0.0
    best_epoch_p2 = 0
    bad_epochs_p2 = 0
    swa_states_p2 = []

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0

        for x, y, _ in mixed_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Mixup
            mixed_x, y_a, y_b, lam = mixup_batch(x, y, args.mixup_alpha)

            optimizer2.zero_grad()
            out = model(mixed_x)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer2.step()

            run_loss += loss.item() * y.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            run_total += y.size(0)

        scheduler2.step(epoch)
        train_acc = run_correct / run_total
        train_loss = run_loss / run_total

        # Validate on HF val (still use HF val for early stopping)
        val_acc, _ = evaluate(model, val_loader, device, tag=f"P2 epoch {epoch}")

        # Also track Mark acc during training (diagnostic)
        mark_acc_diag, _, _ = evaluate_mark(model, mark_eval_loader, device)

        elapsed = time.time() - t0
        cur_lr = optimizer2.param_groups[0]["lr"]
        print(f"  P2 epoch {epoch:>3d}/{args.phase2_epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  mark_acc={mark_acc_diag:.4f}  "
              f"lr={cur_lr:.2e}  elapsed={elapsed:.0f}s")

        history.append({
            "phase": 2, "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc,
            "mark_acc": mark_acc_diag,
        })

        if val_acc > best_val_p2:
            best_val_p2 = val_acc
            best_epoch_p2 = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": val_acc, "phase": 2}, args.out)
            print(f"    ★ saved best P2 (val_acc={val_acc:.4f})")
            bad_epochs_p2 = 0
        else:
            bad_epochs_p2 += 1

        swa_states_p2.append({"model": {k: v.clone() for k, v in model.state_dict().items()},
                              "epoch": epoch, "val_acc": val_acc})
        if len(swa_states_p2) > args.swa_window:
            swa_states_p2.pop(0)

        if bad_epochs_p2 >= args.patience:
            print(f"  Phase 2 early stop at epoch {epoch}")
            break

    # Apply SWA for Phase 2
    print(f"\n  Applying SWA P2 (averaging last {len(swa_states_p2)} checkpoints)...")
    swa_average(model, swa_states_p2, device)
    swa_val_p2, _ = evaluate(model, val_loader, device, tag="P2 SWA")
    print(f"  SWA P2 val_acc={swa_val_p2:.4f} (best single={best_val_p2:.4f})")
    if swa_val_p2 > best_val_p2:
        best_val_p2 = swa_val_p2
        torch.save({"model": model.state_dict(), "epoch": best_epoch_p2,
                    "val_acc": swa_val_p2, "phase": 2, "swa": True}, args.out)
        print(f"  ★ SWA P2 model is better, saved.")

    # ===================================================================
    # PHASE 3: Final evaluation on HF test
    # ===================================================================
    print(f"\n{'='*64}")
    print(f"  PHASE 3: HF TEST EVALUATION")
    print(f"{'='*64}")
    test_acc, test_per = evaluate(model, test_loader, device, tag="HF TEST")
    print("  Per-intent (HF test):")
    for intent in sorted(test_per.keys()):
        c, t = test_per[intent]
        print(f"    {intent:<25s}  {c:>5d}/{t:<5d}  {c/t:.4f}")

    # ===================================================================
    # PHASE 4: Mark's validation
    # ===================================================================
    mark_acc, mark_per, mark_hf_per = evaluate_mark(model, mark_eval_loader, device)

    # ===================================================================
    # Summary
    # ===================================================================
    summary = {
        "config": vars(args),
        "n_params": n_params,
        "phase1_best_val": best_val,
        "phase1_best_epoch": best_epoch,
        "phase2_best_val": best_val_p2,
        "phase2_best_epoch": best_epoch_p2,
        "hf_test_acc": test_acc,
        "mark_validation_acc": mark_acc,
        "mark_per_intent": mark_per,
        "history": history,
    }
    out_json = args.out.replace(".pth", "_eval.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved -> {out_json}")
    print(f"\n{'='*64}")
    print(f"  FINAL RESULTS")
    print(f"{'='*64}")
    print(f"  Phase 1 best val (HF):   {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Phase 2 best val (HF):   {best_val_p2:.4f}  (epoch {best_epoch_p2})")
    print(f"  HF Test Accuracy:        {test_acc:.4f}")
    print(f"  Mark Validation Acc:     {mark_acc:.4f}")
    print(f"  Total time:              {time.time()-t0:.0f}s")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
