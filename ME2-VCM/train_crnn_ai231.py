"""Train the CRNN on the AI231/MEX2 19-intent dataset.

Uses the manifest's speaker-disjoint splits directly (train/val/test), predicts
the COARSE `intent` (19 classes, slot abstracted away), and applies the same
regularisation recipe as the R12 generalisation retrain:
  - waveform augment (time-shift / gain)
  - SpecAugment (time + freq masks) at train time
  - class-balanced CE with label smoothing
  - AdamW + cosine schedule + early stopping on val acc

Usage:
    python train_crnn_ai231.py [--epochs 40] [--batch 64] [--lr 5e-4]
                               [--device cuda] [--seed 42] [--workers 8]
                               [--patience 8] [--out models/crnn_ai231.pth]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import features as F
from crnn_model import CRNN, count_params

DATA_DIR = ("/home/kent.justin.canja/sandbox/"
            "ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/"
            "AI231/MEX2/Data")

# Canonical 19-intent ordering (matches labels.json, alphabetical).
INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER",
]
INTENT2IDX = {i: k for k, i in enumerate(INTENTS)}


def load_manifest(data_dir: str = DATA_DIR):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "manifest.csv"))))
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        intent = r["intent"].strip()
        if intent not in INTENT2IDX:
            continue
        p = os.path.join(data_dir, r["path"])
        if not os.path.exists(p):
            continue
        splits[r["split"]].append((p, INTENT2IDX[intent], intent,
                                   r["speaker"]))
    return splits


class AIDataset(Dataset):
    def __init__(self, items, augment: bool = False, seed: int = 0):
        self.items = items
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, intent, spk = self.items[i]
        x = F.load_wav(path)
        if self.augment:
            x = F.augment(x, self.rng)
        feat = F.preprocess(x)                       # (1, 1, N_FRAMES, N_MELS)
        if self.augment:
            img = F.spec_augment(feat[0, 0], self.rng)
            feat[0, 0] = img
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
                a = INTENTS[p]
                per.setdefault(a, [0, 0])
                per[a][0] += int(p == t)
                per[a][1] += 1
    acc = correct / total if total else 0.0
    return acc, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--label-smoothing", type=float, default=0.10)
    ap.add_argument("--out", type=str, default="models/crnn_ai231.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable -> cpu")

    print("=" * 64)
    print(f"CRNN on AI231/MEX2 19-intent  (seed={args.seed}, device={device})")
    print("=" * 64)

    splits = load_manifest()
    for k in ("train", "val", "test"):
        print(f"  {k:5s}: {len(splits[k]):6d} clips")

    train_ds = AIDataset(splits["train"], augment=True, seed=args.seed)
    val_ds = AIDataset(splits["val"], augment=False)
    test_ds = AIDataset(splits["test"], augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True,
                         persistent_workers=args.workers > 0)

    model = CRNN(n_mels=F.N_MELS, n_frames=F.N_FRAMES,
                 n_intents=len(INTENTS)).to(device)
    npar = count_params(model)
    print(f"[train] CRNN params={npar:,}  ({npar*4/1e6:.2f} MB fp32)")
    print(f"[train] conv out: tf={model.tf} tm={model.tm} "
          f"feat={model.cnn[4].bn.num_features * model.tm}")

    # class-balanced loss
    cnt = Counter(y for _, y, _, _ in splits["train"])
    weights = torch.ones(len(INTENTS), device=device)
    for i in range(len(INTENTS)):
        c = cnt.get(INTENTS[i], 0)
        if c > 0:
            weights[i] = len(splits["train"]) / (len(INTENTS) * c)
    weights = weights / weights.sum() * len(INTENTS)
    crit = nn.CrossEntropyLoss(weight=weights,
                               label_smoothing=args.label_smoothing)
    print("[train] class weights: " + ", ".join(
        f"{INTENTS[i][:5]}={weights[i]:.2f}" for i in range(len(INTENTS))))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    history = []
    best_val = 0.0
    best_epoch = -1
    bad = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        nb = 0
        for x, y, _ in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            nb += 1
        sched.step()
        train_loss = run_loss / max(nb, 1)

        val_acc, _ = evaluate(model, val_dl, device)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_acc": round(val_acc, 4),
                        "lr": opt.param_groups[0]["lr"],
                        "elapsed_s": round(elapsed, 1)})
        print(f"ep {epoch:3d}  loss {train_loss:.4f}  val {val_acc*100:5.2f}%"
              f"  lr {opt.param_groups[0]['lr']:.2e}  {elapsed:6.0f}s")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "model": "CRNN",
                "n_intents": len(INTENTS),
                "intents": INTENTS,
                "n_mels": F.N_MELS,
                "n_frames": F.N_FRAMES,
                "seed": args.seed,
                "best_val_acc": round(best_val, 4),
                "best_epoch": epoch,
                "data": "AI231/MEX2",
            }, args.out)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train] early stop @ ep {epoch} "
                      f"(best val {best_val*100:.2f}% @ ep {best_epoch})")
                break

    # reload best, report val + test
    ck = torch.load(args.out, map_location=device)
    model.load_state_dict(ck["state_dict"])
    val_acc, val_per = evaluate(model, val_dl, device)
    test_acc, test_per = evaluate(model, test_dl, device)

    print("\n" + "=" * 64)
    print(f"BEST val {best_val*100:.2f}% @ ep {best_epoch}")
    print(f"FINAL val {val_acc*100:.2f}%   TEST {test_acc*100:.2f}%  "
          f"({len(splits['test'])} held-out clips, 10 unseen speakers)")
    print("-" * 64)
    print(f"{'intent':<18}{'test':>10}")
    for a in INTENTS:
        c, t = test_per.get(a, [0, 0])
        print(f"{a:<18}{c}/{t:>4}  {c/t*100 if t else 0:6.1f}%")
    print("=" * 64)

    with open(os.path.join(os.path.dirname(args.out) or ".",
                           "crnn_ai231_history.json"), "w") as fh:
        json.dump({"args": vars(args), "history": history,
                   "best_val": best_val, "final_val": val_acc,
                   "final_test": test_acc, "test_per_intent": test_per},
                  fh, indent=2)
    print(f"[train] saved -> {args.out}")


if __name__ == "__main__":
    main()
