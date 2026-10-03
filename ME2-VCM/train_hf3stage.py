"""Train CRNN on HF AI231-ME2 with 3-stage validation.

Strategy:
  1. TRAIN  on HF train split  (10,682 clips, 315 speakers)
  2. TEST   on HF test split   (4,418 clips, 121 speakers, speaker-disjoint)
  3. VALIDATE on Mark's real dataset (1,368 synthetic + 76 real = 1,444 clips,
     12 intents — a completely different domain)

The model learns 20 classes (19 intents + OUT_OF_SCOPE) from the HF data,
then we check generalisation on Mark's smaller, differently-phrased set.

Usage:
    python train_hf3stage.py [--epochs 25] [--batch 64] [--lr 5e-4]
                             [--device cuda] [--seed 42] [--workers 8]
                             [--patience 6] [--out models/crnn_hf_20_v2.pth]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import time
import wave
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import features as F
from crnn_model import CRNN, count_params

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HF_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2"
MARK_DIR = ("/home/kent.justin.canja/sandbox/"
            "ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/"
            "ME2_ Voice Controlled Smart Device/vcm_dataset")

# ---------------------------------------------------------------------------
# 20 classes: 19 coarse intents + OUT_OF_SCOPE
# ---------------------------------------------------------------------------
INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER", "OUT_OF_SCOPE",
]
INTENT2IDX = {i: k for k, i in enumerate(INTENTS)}

# Map Mark's 12 intents -> HF 20-class labels
MARK_TO_HF = {
    "dim_lights":        "BRIGHTNESS",
    "set_timer":         "TIMER",
    "set_alarm":         "ALARM",
    "set_temperature":   "TEMPERATURE",
    "media_control":     "PAUSE",
    "make_call":         "CALL",
    "light_on":          "LIGHT_ON",
    "light_off":         "LIGHT_OFF",
    "play_music":        "PLAY_MUSIC",
    "manage_reminders":  "CREATE_REMINDER",
    "get_weather":       "WEATHER",
    "get_time":          "TIME",
}


# ---------------------------------------------------------------------------
# HF data loading
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


class HFDataset(Dataset):
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
            feat[0, 0] = F.spec_augment(feat[0, 0], self.rng)
        return torch.from_numpy(feat).squeeze(0), label, intent


class SubsetDS(Dataset):
    def __init__(self, parent: HFDataset, sel, augment: bool = False, seed: int = 0):
        self.parent, self.sel = parent, sel
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
            feat[0, 0] = F.spec_augment(feat[0, 0], self.rng)
        return torch.from_numpy(feat).squeeze(0), label, intent


# ---------------------------------------------------------------------------
# Mark's dataset (WAV files on disk)
# ---------------------------------------------------------------------------
class MarkDataset(Dataset):
    """Load Mark's 1,444 clips (synthetic + real) mapped to HF 20-class labels."""

    def __init__(self):
        self.items = []
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

        # Also add real voice clips
        real_meta = pd.read_csv("/home/kent.justin.canja/sandbox/AI-222-231/ME2-VCM/real_data/real_metadata.csv")
        for _, row in real_meta.iterrows():
            fname = row["file_name"]
            intent_mark = str(row["intent"]).strip()
            if intent_mark not in MARK_TO_HF:
                continue
            hf_label = MARK_TO_HF[intent_mark]
            path = os.path.join("/home/kent.justin.canja/sandbox/AI-222-231/ME2-VCM/real_data", fname)
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
        x, label, hf_intent, mark_intent = self.items[i]
        feat = F.preprocess(x.astype(np.float32))
        return torch.from_numpy(feat).squeeze(0), label, hf_intent, mark_intent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, tag: str = ""):
    model.eval()
    correct = total = 0
    per = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for p, t in zip(pred.cpu().tolist(), y.tolist()):
                a = INTENTS[t]
                per[a][0] += int(p == t)
                per[a][1] += 1
    acc = correct / total if total else 0.0
    print(f"\n{'='*64}")
    print(f"  {tag.upper()}  —  accuracy: {acc:.4f}  ({correct}/{total})")
    print(f"{'='*64}")
    for intent in INTENTS:
        c, t = per.get(intent, [0, 0])
        if t > 0:
            print(f"  {intent:<20s}  {c:>5d}/{t:<5d}  {c/t:.4f}")
    return acc, dict(per)


def evaluate_mark(model, loader, device):
    """Evaluate on Mark's dataset with intent-level reporting."""
    model.eval()
    correct = total = 0
    per_hf = defaultdict(lambda: [0, 0])
    per_mark = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch in loader:
            x, y, hf_intent, mark_intent = batch
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for p, t, hi, mi in zip(pred.cpu().tolist(), y.tolist(), hf_intent, mark_intent):
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
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--out", type=str, default="models/crnn_hf_20_v2.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable -> cpu")

    print("=" * 64)
    print(f"CRNN HF 3-stage training  (seed={args.seed}, device={device})")
    print(f"  Stage 1: TRAIN  on HF train  (10,682 clips)")
    print(f"  Stage 2: TEST   on HF test   (4,418 clips)")
    print(f"  Stage 3: VALIDATE on Mark's  (1,444 clips)")
    print("=" * 64)

    # --- Load data ---
    print("\n[1/4] Loading HF data...")
    train_df = _load_parquet("train")
    test_df = _load_parquet("test")

    train_full = HFDataset(train_df, augment=True, seed=args.seed)
    test_ds = HFDataset(test_df, augment=False)
    print(f"  train: {len(train_full)}  test: {len(test_ds)}")

    # Carve 10% of train for early-stopping validation
    n_val = max(1, int(len(train_full) * 0.1))
    idx = np.arange(len(train_full))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    train_ds = SubsetDS(train_full, tr_idx, augment=True, seed=args.seed)
    val_ds = SubsetDS(train_full, val_idx, augment=False)
    print(f"  split: train={len(train_ds)}  val={len(val_ds)}")

    # Load Mark's dataset
    print("\n[2/4] Loading Mark's dataset...")
    mark_ds = MarkDataset()
    print(f"  mark: {len(mark_ds)} clips")

    # --- DataLoaders ---
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    mark_loader = DataLoader(mark_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- Model ---
    model = CRNN(n_intents=len(INTENTS)).to(device)
    n_params = count_params(model)
    print(f"\n[3/4] Model: CRNN  params={n_params:,}  ({n_params*4/1e6:.2f} MB)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    print(f"\n[4/4] Training {args.epochs} epochs...")
    best_val = 0.0
    best_epoch = 0
    bad_epochs = 0
    t0 = time.time()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            run_correct += (out.argmax(1) == y).sum().item()
            run_total += y.size(0)

        scheduler.step()
        train_acc = run_correct / run_total
        train_loss = run_loss / run_total

        # Validate on HF val split
        val_acc, _ = evaluate(model, val_loader, device, tag=f"epoch {epoch} val")

        elapsed = time.time() - t0
        print(f"  epoch {epoch:>3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  elapsed={elapsed:.0f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "val_acc": val_acc,
        })

        # Early stopping
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "intents": INTENTS,
                "intent2idx": INTENT2IDX,
                "mark_to_hf": MARK_TO_HF,
                "epoch": epoch,
                "val_acc": val_acc,
                "n_params": n_params,
            }, args.out)
            print(f"  ★ saved best (val_acc={val_acc:.4f}) -> {args.out}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop at epoch {epoch} (patience={args.patience})")
                break

    # --- Restore best ---
    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"\nRestored best model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    # --- Stage 2: HF Test ---
    test_acc, test_per = evaluate(model, test_loader, device, tag="HF TEST")

    # --- Stage 3: Mark's Validation ---
    mark_acc, mark_per, mark_hf_per = evaluate_mark(model, mark_loader, device)

    # --- Summary ---
    summary = {
        "config": vars(args),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
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
    print(f"  HF Test Accuracy:      {test_acc:.4f}")
    print(f"  Mark Validation Acc:   {mark_acc:.4f}")
    print(f"  Best Val (early stop): {best_val:.4f}  (epoch {best_epoch})")
    print(f"  Total time:            {time.time()-t0:.0f}s")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
