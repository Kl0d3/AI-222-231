"""Train the CRNN on the HF AI231-ME2 multi-speaker dataset (20 classes).

Same architecture / regularisation as train_crnn_ai231.py, but:
  - data source = HF parquet (audio embedded in the `audio` column)
  - 20 classes = the 19 coarse intents + OUT_OF_SCOPE (rejection)
  - speaker-disjoint splits come straight from the dataset (train/test)

The coarse `intent` is derived from the `command` column:
  command == "OUT_OF_SCOPE"  -> OUT_OF_SCOPE
  otherwise                  -> the command itself (already the coarse intent)

Usage:
    python train_crnn_hf.py [--epochs 20] [--batch 64] [--lr 5e-4]
                            [--device cuda] [--seed 42] [--workers 8]
                            [--patience 6] [--out models/crnn_hf_20.pth]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import features as F
from crnn_model import CRNN, count_params

HF_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2"

# 19 coarse intents (alphabetical) + OUT_OF_SCOPE appended as the 20th class.
INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER", "OUT_OF_SCOPE",
]
INTENT2IDX = {i: k for k, i in enumerate(INTENTS)}


def _load_parquet(split: str) -> pd.DataFrame:
    """Load ALL parquet shards for a split (they live under data/)."""
    files = []
    for pat in (os.path.join(HF_DIR, "data", f"{split}-*.parquet"),
                os.path.join(HF_DIR, f"{split}-*.parquet")):
        files = sorted(glob.glob(pat))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"no parquet for split '{split}' under {HF_DIR} (checked data/)")
    frames = [pd.read_parquet(f) for f in files]
    print(f"  loaded {split}: {len(files)} shard(s) -> {sum(len(x) for x in frames)} rows")
    return pd.concat(frames, ignore_index=True)


def _coarse_intent(row) -> str:
    if int(row["out_of_scope"]) == 1 or row["command"] == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    cmd = str(row["command"]).strip().upper()
    return cmd if cmd in INTENT2IDX else None



def _decode_wav_bytes(b: bytes) -> np.ndarray:
    """Decode raw WAV bytes (16-bit mono @ 16 kHz) -> float32 PCM in [-1, 1]."""
    import io, wave
    w = wave.open(io.BytesIO(b), "rb")
    assert w.getnchannels() == 1, f"expected mono, got {w.getnchannels()} ch"
    assert w.getsampwidth() == 2, f"expected 16-bit, got {w.getsampwidth()*8}-bit"
    assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()} Hz"
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    pcm = np.frombuffer(raw, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0

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
        feat = F.preprocess(x)                       # (1, 1, N_FRAMES, N_MELS)
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


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per = {}
    with torch.no_grad():
        for x, y, intent in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for p, t in zip(pred.cpu().tolist(), y.tolist()):
                a = INTENTS[t]
                per.setdefault(a, [0, 0])
                per[a][0] += int(p == t)
                per[a][1] += 1
    acc = correct / total if total else 0.0
    return acc, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--out", type=str, default="models/crnn_hf_20.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable -> cpu")

    print("=" * 64)
    print(f"CRNN on HF AI231-ME2 20-class  (seed={args.seed}, device={device})")
    print("=" * 64)

    train_df = _load_parquet("train")
    test_df = _load_parquet("test")
    print(f"  raw rows: train={len(train_df)}  test={len(test_df)}")

    train_ds = HFDataset(train_df, augment=True, seed=args.seed)
    test_ds = HFDataset(test_df, augment=False)
    n_val = max(1, int(len(train_ds) * 0.1))
    idx = np.arange(len(train_ds))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tr_ds = SubsetDS(train_ds, tr_idx, augment=True, seed=args.seed)
    val_ds = SubsetDS(train_ds, val_idx, augment=False)

    kw = dict(num_workers=args.workers, pin_memory=True,
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, **kw)

    model = CRNN(n_mels=F.N_MELS, n_frames=F.N_FRAMES,
                 n_intents=len(INTENTS)).to(device)
    npar = count_params(model)
    print(f"[train] CRNN params={npar:,}  ({npar*4/1e6:.2f} MB fp32)")

    cnt = Counter(y for _, y, _ in train_ds.items)
    total_n = sum(cnt.values())
    weights = torch.tensor([total_n / max(1, cnt.get(INTENTS[i], 1))
                            for i in range(len(INTENTS))],
                           dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val, bad = 0.0, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, run_acc, n = 0.0, 0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            run_acc += (out.argmax(1) == y).sum().item()
            n += y.numel()
        sched.step()
        val_acc, _ = evaluate(model, val_dl, device)
        dt = time.time() - t0
        print(f"ep {ep:02d}/{args.epochs}  train_acc={run_acc/n:.4f} "
              f"loss={run_loss/n:.4f}  val_acc={val_acc:.4f}  ({dt:.0f}s)")
        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            torch.save({"model": model.state_dict(),
                        "intents": INTENTS,
                        "n_mels": F.N_MELS, "n_frames": F.N_FRAMES,
                        "val_acc": best_val, "epoch": ep}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train] early stop at ep {ep} (best val {best_val:.4f})")
                break

    ck = torch.load(args.out, map_location=device)
    model.load_state_dict(ck["model"])
    test_acc, per = evaluate(model, test_dl, device)
    print("\n=== TEST (speaker-disjoint) ===")
    print(f"overall acc = {test_acc:.4f}  ({len(test_ds)} clips)")
    for it in INTENTS:
        c, t = per.get(it, [0, 0])
        print(f"  {it:16s} {c:5d}/{t:<5d} {c/t if t else 0:.3f}")
    with open(os.path.join(os.path.dirname(args.out) or ".", "hf_test_eval.json"),
              "w") as fh:
        json.dump({"overall_acc": test_acc, "n_test": len(test_ds),
                   "per_intent": per, "best_val_acc": best_val}, fh, indent=2)
    print(f"\n[done] best_val={best_val:.4f} test={test_acc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
