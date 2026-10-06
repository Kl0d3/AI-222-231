"""Train CRNN-HF-20 with HF + Mark + Kent real voice.

Phase 1: Pre-train on HF train (10,682 clips, 315 speakers)
Phase 2: Domain-adapt on HF + Mark + Kent 300 real clips
Phase 3: Evaluate on HF test + Mark + Kent real voice

Output: models/crnn_hf_20_real.pth
"""
from __future__ import annotations
import argparse, glob, io, json, math, os, random, time, wave
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import features as F
from crnn_model import CRNN, count_params

HF_DIR = "/home/kent.justin.canja/sandbox/hf_ai231_me2"
MARK_DIR = (
    "/home/kent.justin.canja/sandbox/"
    "ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/"
    "ME2_ Voice Controlled Smart Device/vcm_dataset"
)
REAL_DATA_DIR = "/home/kent.justin.canja/sandbox/AI-222-231/ME2-VCM/real_data"
HERE = os.path.dirname(os.path.abspath(__file__))

INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER", "OUT_OF_SCOPE",
]
INTENT2IDX = {i: k for k, i in enumerate(INTENTS)}
N_INTENTS = len(INTENTS)

MARK_TO_HF = {
    "dim_lights": "BRIGHTNESS", "set_timer": "TIMER", "set_alarm": "ALARM",
    "set_temperature": "TEMPERATURE", "media_control": "PAUSE",
    "make_call": "CALL", "light_on": "LIGHT_ON", "light_off": "LIGHT_OFF",
    "play_music": "PLAY_MUSIC", "manage_reminders": "CREATE_REMINDER",
    "get_weather": "WEATHER", "get_time": "TIME",
}
REAL_TO_HF = MARK_TO_HF


def _load_parquet(split):
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


def _coarse_intent(row):
    if int(row["out_of_scope"]) == 1 or row["command"] == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    cmd = str(row["command"]).strip().upper()
    return cmd if cmd in INTENT2IDX else None


def _decode_wav_bytes(b):
    w = wave.open(io.BytesIO(b), "rb")
    assert w.getnchannels() == 1
    a = w.readframes(w.getnframes())
    sr = w.getframerate()
    w.close()
    x = np.frombuffer(a, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


def _load_wav_16k(path):
    w = wave.open(path, "rb")
    assert w.getnchannels() == 1, f"{path}: expected mono"
    a = w.readframes(w.getnframes())
    sr = w.getframerate()
    w.close()
    x = np.frombuffer(a, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


def _augment(x, rng):
    shift = int(rng.integers(-480, 480))
    if shift > 0:
        x = np.concatenate([np.zeros(shift, np.float32), x[:-shift]])
    elif shift < 0:
        x = x[-shift:]
    gain_db = rng.uniform(-8, 8)
    x = x * (10 ** (gain_db / 20.0))
    if rng.random() < 0.7:
        snr_db = rng.uniform(15, 35)
        noise = rng.standard_normal(len(x)).astype(np.float32)
        sig_power = np.mean(x ** 2) + 1e-10
        noise_power = np.mean(noise ** 2) + 1e-10
        noise_scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10.0)))
        x = x + noise * noise_scale
    if rng.random() < 0.3:
        rate = rng.uniform(0.9, 1.1)
        new_len = int(len(x) / rate)
        indices = np.linspace(0, len(x) - 1, new_len).astype(int)
        indices = np.clip(indices, 0, len(x) - 1)
        x = x[indices]
    return np.clip(x, -1.0, 1.0)


class HFDataset(Dataset):
    def __init__(self, df, augment=False, seed=0):
        self.items = []
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        for idx, row in df.iterrows():
            label = _coarse_intent(row)
            if label is None:
                continue
            self.items.append((row["audio"]["bytes"], INTENT2IDX[label]))
        print(f"  HF dataset: {len(self.items)} clips (augment={augment})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        audio_bytes, label = self.items[i]
        x = _decode_wav_bytes(audio_bytes)
        if self.augment:
            x = _augment(x, self.rng)
        feat = F.preprocess(x)  # (1, 1, 99, 40)
        feat = feat.squeeze(0)  # (1, 99, 40)
        return torch.from_numpy(feat), label


class MarkDataset(Dataset):
    def __init__(self, augment=False, seed=0):
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
            if not os.path.exists(path):
                continue
            self.items.append((path, INTENT2IDX[hf_label]))
        print(f"  Mark dataset: {len(self.items)} clips (augment={augment})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        x = _load_wav_16k(path)
        if self.augment:
            x = _augment(x, self.rng)
        feat = F.preprocess(x)  # (1, 1, 99, 40)
        feat = feat.squeeze(0)  # (1, 99, 40)
        return torch.from_numpy(feat), label


class RealVoiceDataset(Dataset):
    def __init__(self, items, augment=True, seed=0):
        self.items = items
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        print(f"  RealVoice dataset: {len(self.items)} clips (augment={augment})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        x = _load_wav_16k(path)
        if self.augment:
            x = _augment(x, self.rng)
        feat = F.preprocess(x)  # (1, 1, 99, 40)
        feat = feat.squeeze(0)  # (1, 99, 40)
        return torch.from_numpy(feat), label


class MixedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.offsets = []
        total = 0
        for ds in datasets:
            self.offsets.append(total)
            total += len(ds)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        for ds, off in zip(self.datasets, self.offsets):
            if i < off + len(ds):
                return ds[i - off]
        raise IndexError(i)


class SubsetDS(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


def mixup_batch(x, y, alpha, rng):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(rng.beta(alpha, alpha))
    n = x.size(0)
    idx = torch.randperm(n, device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def evaluate(model, loader, device, tag=""):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total if total else 0.0
    if tag:
        print(f"  [{tag}] acc={acc:.4f} ({correct}/{total})")
    return acc, (correct, total)


def evaluate_per_intent(model, loader, device, label):
    model.eval()
    per_intent = defaultdict(lambda: {"n": 0, "ok": 0})
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            for yi, pi in zip(y.tolist(), pred.tolist()):
                intent_name = INTENTS[yi]
                per_intent[intent_name]["n"] += 1
                per_intent[intent_name]["ok"] += (yi == pi)
                correct += (yi == pi)
                total += 1
    acc = correct / total if total else 0.0
    print(f"  {label}: acc={acc:.4f} ({correct}/{total})")
    for intent in sorted(per_intent):
        d = per_intent[intent]
        a = d["ok"] / d["n"] * 100 if d["n"] else 0
        print(f"    {intent:<20} {d['ok']:>4}/{d['n']:<4} ({a:.0f}%)")
    return acc, dict(per_intent)


def swa_average(model, swa_states, device):
    n = len(swa_states)
    model_dict = model.state_dict()
    for key in model_dict:
        model_dict[key] = sum(s["model"][key].to(device) for s in swa_states) / n
    model.load_state_dict(model_dict)


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            frac = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            frac = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * frac


def stratified_split(items, val_frac, seed=42):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for item in items:
        by_label[item[1]].append(item)
    train_items = []
    val_items = []
    for label, group in sorted(by_label.items()):
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_frac))
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-epochs", type=int, default=20)
    ap.add_argument("--phase2-epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-phase2", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--swa-window", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "models", "crnn_hf_20_real.pth"))
    ap.add_argument("--eval-out", default=os.path.join(HERE, "models", "crnn_hf_20_real_eval.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = CRNN(n_intents=N_INTENTS)
    n_params = count_params(model)
    print(f"Model: CRNN | params={n_params:,}")
    model.to(device)

    print("\n" + "=" * 64)
    print("  LOADING DATA")
    print("=" * 64)

    train_df = _load_parquet("train")
    test_df = _load_parquet("test")

    mark_meta = pd.read_csv(os.path.join(MARK_DIR, "metadata.csv"))
    mark_items = []
    for _, row in mark_meta.iterrows():
        fname = row["file_name"]
        intent_mark = str(row["intent"]).strip()
        if intent_mark not in MARK_TO_HF:
            continue
        hf_label = MARK_TO_HF[intent_mark]
        path = os.path.join(MARK_DIR, fname)
        if os.path.exists(path):
            mark_items.append((path, INTENT2IDX[hf_label]))
    print(f"  Mark clips: {len(mark_items)}")

    real_meta = pd.read_csv(os.path.join(REAL_DATA_DIR, "real_metadata.csv"))
    real_items_all = []
    for _, row in real_meta.iterrows():
        fname = row["file_name"]
        intent_real = str(row["intent"]).strip()
        if intent_real not in REAL_TO_HF:
            continue
        hf_label = REAL_TO_HF[intent_real]
        path = os.path.join(REAL_DATA_DIR, fname)
        if os.path.exists(path):
            real_items_all.append((path, INTENT2IDX[hf_label]))
    print(f"  Kent real clips (total): {len(real_items_all)}")

    real_train_items, real_val_items = stratified_split(
        real_items_all, val_frac=args.val_frac, seed=args.seed
    )
    print(f"  Kent real train: {len(real_train_items)}, val: {len(real_val_items)}")

    val_counts = defaultdict(int)
    for _, label in real_val_items:
        val_counts[INTENTS[label]] += 1
    print(f"  Val per-intent: {dict(sorted(val_counts.items()))}")

    # Phase 1
    print(f"\n{'=' * 64}")
    print(f"  PHASE 1: Pre-train on HF ({args.phase1_epochs} epochs, lr={args.lr})")
    print(f"{'=' * 64}")

    train_full = HFDataset(train_df, augment=True, seed=args.seed)
    test_ds = HFDataset(test_df, augment=False, seed=args.seed)

    n_val_hf = max(1, int(len(train_full) * 0.05))
    rng = random.Random(args.seed)
    all_indices = list(range(len(train_full)))
    rng.shuffle(all_indices)
    val_hf_indices = all_indices[:n_val_hf]
    train_hf_indices = all_indices[n_val_hf:]

    train_ds_p1 = SubsetDS(train_full, train_hf_indices)
    val_ds_p1 = SubsetDS(train_full, val_hf_indices)

    train_loader = DataLoader(train_ds_p1, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds_p1, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, args.warmup, args.phase1_epochs)

    best_val = 0.0
    best_epoch = 0
    patience_counter = 0
    swa_states = []
    history = []

    for epoch in range(1, args.phase1_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        rng_np = np.random.default_rng(args.seed + epoch)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_mix, y_a, y_b, lam = mixup_batch(x, y, args.mixup_alpha, rng_np)
            optimizer.zero_grad()
            out = model(x_mix)
            loss_a = F_torch.cross_entropy(out, y_a, label_smoothing=args.label_smoothing)
            loss_b = F_torch.cross_entropy(out, y_b, label_smoothing=args.label_smoothing)
            loss = lam * loss_a + (1 - lam) * loss_b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            total_correct += (lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()).sum().item()
            total_samples += x.size(0)

        scheduler.step(epoch)
        train_acc = total_correct / total_samples
        val_acc, _ = evaluate(model, val_loader, device, tag=f"P1 ep{epoch}")
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  ep{epoch:>2}/{args.phase1_epochs} | loss={total_loss / total_samples:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")
        history.append({"phase": 1, "epoch": epoch,
                        "train_loss": total_loss / total_samples,
                        "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": val_acc, "phase": 1}, args.out)
            print(f"    * new best val_acc={val_acc:.4f}, saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch} (patience={args.patience})")
                break
        swa_states.append({"model": model.state_dict().copy(), "epoch": epoch})
        if len(swa_states) > args.swa_window:
            swa_states.pop(0)

    print(f"\n  Applying SWA (averaging last {len(swa_states)} checkpoints)...")
    swa_average(model, swa_states, device)
    swa_val_acc, _ = evaluate(model, val_loader, device, tag="P1 SWA")
    print(f"  SWA val_acc={swa_val_acc:.4f} (best single={best_val:.4f})")
    if swa_val_acc > best_val:
        best_val = swa_val_acc
        torch.save({"model": model.state_dict(), "epoch": best_epoch,
                    "val_acc": swa_val_acc, "phase": 1, "swa": True}, args.out)
        print(f"  * SWA model is better, saved.")

    # Phase 2
    print(f"\n{'=' * 64}")
    print(f"  PHASE 2: Domain-adapt HF + Mark + Real ({args.phase2_epochs} epochs, lr={args.lr_phase2})")
    print(f"{'=' * 64}")

    hf_train_p2 = HFDataset(train_df, augment=True, seed=args.seed + 100)
    mark_ds = MarkDataset(augment=True, seed=args.seed + 200)
    real_train_ds = RealVoiceDataset(real_train_items, augment=True, seed=args.seed + 300)

    mixed_ds = MixedDataset(hf_train_p2, mark_ds, real_train_ds)
    mixed_loader = DataLoader(mixed_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    real_val_ds = RealVoiceDataset(real_val_items, augment=False, seed=args.seed + 400)
    real_val_loader = DataLoader(real_val_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

    hf_val_ds = HFDataset(test_df, augment=False, seed=args.seed + 500)
    hf_val_loader = DataLoader(hf_val_ds, batch_size=args.batch, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_phase2,
                                  weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, 1, args.phase2_epochs)

    best_real_val = 0.0
    best_epoch_p2 = 0
    patience_counter = 0
    swa_states_p2 = []

    for epoch in range(1, args.phase2_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        rng_np = np.random.default_rng(args.seed + 1000 + epoch)

        for x, y in mixed_loader:
            x, y = x.to(device), y.to(device)
            x_mix, y_a, y_b, lam = mixup_batch(x, y, args.mixup_alpha, rng_np)
            optimizer.zero_grad()
            out = model(x_mix)
            loss_a = F_torch.cross_entropy(out, y_a, label_smoothing=args.label_smoothing)
            loss_b = F_torch.cross_entropy(out, y_b, label_smoothing=args.label_smoothing)
            loss = lam * loss_a + (1 - lam) * loss_b
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            total_correct += (lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()).sum().item()
            total_samples += x.size(0)

        scheduler.step(epoch)
        train_acc = total_correct / total_samples
        real_val_acc, _ = evaluate(model, real_val_loader, device, tag=f"P2 ep{epoch} REAL")
        hf_val_acc, _ = evaluate(model, hf_val_loader, device, tag=f"P2 ep{epoch} HF")
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  ep{epoch:>2}/{args.phase2_epochs} | loss={total_loss / total_samples:.4f} "
              f"train_acc={train_acc:.4f} real_val={real_val_acc:.4f} hf_val={hf_val_acc:.4f} "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")
        history.append({"phase": 2, "epoch": epoch,
                        "train_loss": total_loss / total_samples,
                        "train_acc": train_acc,
                        "val_acc": real_val_acc,
                        "hf_val_acc": hf_val_acc})
        if real_val_acc > best_real_val:
            best_real_val = real_val_acc
            best_epoch_p2 = epoch
            patience_counter = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": real_val_acc, "phase": 2}, args.out)
            print(f"    * new best real_val={real_val_acc:.4f}, saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch} (patience={args.patience})")
                break
        swa_states_p2.append({"model": model.state_dict().copy(), "epoch": epoch})
        if len(swa_states_p2) > args.swa_window:
            swa_states_p2.pop(0)

    print(f"\n  Applying SWA (averaging last {len(swa_states_p2)} checkpoints)...")
    swa_average(model, swa_states_p2, device)
    swa_real_acc, _ = evaluate(model, real_val_loader, device, tag="P2 SWA REAL")
    swa_hf_acc, _ = evaluate(model, hf_val_loader, device, tag="P2 SWA HF")
    print(f"  SWA real_val={swa_real_acc:.4f} (best single={best_real_val:.4f})")
    if swa_real_acc > best_real_val:
        best_real_val = swa_real_acc
        torch.save({"model": model.state_dict(), "epoch": best_epoch_p2,
                    "val_acc": swa_real_acc, "phase": 2, "swa": True}, args.out)
        print(f"  * SWA model is better, saved.")

    # Final evaluation
    print(f"\n{'=' * 64}")
    print(f"  FINAL EVALUATION")
    print(f"{'=' * 64}")

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    hf_test_acc, (hf_correct, hf_total) = evaluate(model, test_loader, device, tag="HF TEST")

    mark_val_ds = MarkDataset(augment=False, seed=args.seed + 600)
    mark_val_loader = DataLoader(mark_val_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    mark_acc, mark_per_intent = evaluate_per_intent(model, mark_val_loader, device, "Mark validation")

    real_full_ds = RealVoiceDataset(real_items_all, augment=False, seed=args.seed + 700)
    real_full_loader = DataLoader(real_full_ds, batch_size=args.batch, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    real_acc, real_per_intent = evaluate_per_intent(model, real_full_loader, device, "Kent real voice (all)")

    payload = {
        "config": vars(args),
        "n_params": n_params,
        "phase1_best_val": best_val,
        "phase1_best_epoch": best_epoch,
        "phase2_best_real_val": best_real_val,
        "phase2_best_epoch": best_epoch_p2,
        "hf_test_acc": hf_test_acc,
        "mark_validation_acc": mark_acc,
        "mark_per_intent": {k: [v["ok"], v["n"]] for k, v in sorted(mark_per_intent.items())},
        "real_voice_acc": real_acc,
        "real_voice_per_intent": {k: [v["ok"], v["n"]] for k, v in sorted(real_per_intent.items())},
        "real_val_split_size": len(real_val_items),
        "real_train_split_size": len(real_train_items),
        "history": history,
    }
    os.makedirs(os.path.dirname(args.eval_out) or ".", exist_ok=True)
    json.dump(payload, open(args.eval_out, "w"), indent=2)
    print(f"\n  Eval report saved to: {args.eval_out}")
    print(f"  Model saved to: {args.out}")
    print(f"\n  SUMMARY:")
    print(f"    HF test accuracy:      {hf_test_acc * 100:.1f}%")
    print(f"    Mark validation:       {mark_acc * 100:.1f}%")
    print(f"    Kent real voice (all): {real_acc * 100:.1f}%")
    print(f"    Kent real voice (val): {best_real_val * 100:.1f}%")


if __name__ == "__main__":
    main()
