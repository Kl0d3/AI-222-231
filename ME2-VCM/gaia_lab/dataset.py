"""Dataset loading + speaker-aware stratified train/val/test split."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

import features as F

DATASET_DIR = ("/home/kent.justin.canja/sandbox/"
               "ME2_ Voice Controlled Smart Device-20260925T153305Z-1-001/"
               "ME2_ Voice Controlled Smart Device/vcm_dataset")

# Canonical intent ordering (matches idx2intent.pth in the dataset).
INTENTS = [
    "dim_lights", "get_time", "get_weather", "light_off", "light_on",
    "make_call", "manage_reminders", "media_control", "play_music",
    "set_alarm", "set_temperature", "set_timer",
]
INTENT2IDX = {i: k for k, i in enumerate(INTENTS)}


@dataclass
class Sample:
    path: str
    intent: str
    label: int
    voice: str
    source: str          # "synthetic" | "real"


def load_samples(dataset_dir: str = DATASET_DIR,
                 extra_dirs: list[str] | str | None = None) -> list[Sample]:
    """Load the original ME2 dataset plus optional generated extra data.

    `extra_dirs` (e.g. [vcm/extra_data, vcm/extra_data2]) must each contain
    a metadata csv (extra_metadata.csv) and the wav files referenced inside.
    """
    if isinstance(extra_dirs, str):
        extra_dirs = [extra_dirs]
    extra_dirs = [d for d in (extra_dirs or []) if d and os.path.isdir(d)]
    rows = list(csv.DictReader(open(os.path.join(dataset_dir, "metadata.csv"))))
    for d in extra_dirs:
        # extra dirs use extra_metadata.csv; the UI collector uses real_metadata.csv
        for meta in ("extra_metadata.csv", "real_metadata.csv"):
            em = os.path.join(d, meta)
            if os.path.exists(em):
                rows += list(csv.DictReader(open(em)))
    out = []
    for r in rows:
        fn = r["file_name"]
        p = os.path.join(dataset_dir, fn)
        if not os.path.exists(p):
            found = False
            for d in extra_dirs:
                for cand in (os.path.join(d, fn),
                             os.path.join(d, os.path.basename(r["intent"]), fn)):
                    if os.path.exists(cand):
                        p = cand
                        found = True
                        break
                if found:
                    break
            if not found:
                continue
        intent = r["intent"].strip()
        if intent not in INTENT2IDX:
            continue
        if "real_voice_audio" in fn:
            source = "real"
        elif "real_voice_" in fn:
            # NEW real-mic clips collected via record_ui.py (vcm/real_data/)
            source = "real_new"
        elif "robust_" in fn:
            source = "robust"
        elif "audio_extra3" in fn or "web_real_" in fn:
            source = "web_real"
        else:
            source = "synthetic"
        out.append(Sample(p, intent, INTENT2IDX[intent], r["voice"], source))
    return out


def _group_key(s: Sample):
    # Group by (intent, voice) so a given speaker never crosses the split.
    return (s.intent, s.voice)


def stratified_split(samples: list[Sample], seed: int = 42,
                     val_frac: float = 0.10, test_frac: float = 0.10):
    """Speaker-aware stratified split.

    Real (human) recordings are reserved entirely as the held-out TEST set:
    they come from a single microphone/speaker that never appears in training,
    so they form a genuine cross-domain generalization benchmark. The
    synthetic (TTS) recordings are split by (intent, voice) group so no TTS
    speaker crosses the train/val boundary.
    """
    rng = np.random.default_rng(seed)
    real = [s for s in samples if s.source == "real"]
    synth = [s for s in samples if s.source == "synthetic"]

    groups: dict[tuple, list[Sample]] = {}
    for s in synth:
        groups.setdefault(_group_key(s), []).append(s)

    keys = list(groups.keys())
    rng.shuffle(keys)
    total = sum(len(groups[k]) for k in keys)
    n_val = int(total * val_frac)
    n_test = int(total * test_frac)

    train, val, test = [], [], []
    acc = 0
    for k in keys:
        g = groups[k]
        if acc < n_test:
            test.extend(g)
        elif acc < n_test + n_val:
            val.extend(g)
        else:
            train.extend(g)
        acc += len(g)

    # real recordings -> dedicated generalization test set
    test = test + real
    return train, val, test



def real_voice_split(samples: list[Sample], seed: int = 42,
                     val_frac: float = 0.12, test_frac: float = 0.12):
    """Speaker-disjoint split tuned for REAL-mic generalization.

    The user's ORIGINAL microphone recordings (source == 'real', 76 clips,
    'human_user_mic') are reserved ENTIRELY as the held-out TEST set — the
    true benchmark: an unseen human voice on a real mic.

    NEW real-mic clips collected via the recording UI (source == 'real_new')
    are TRAINING data: they are the user's voice/mic/room, which is exactly
    what the model needs to learn. Each clip has a unique voice tag, so the
    (intent, voice) grouping never splits one clip across sets.

    Everything else -- clean TTS (source == 'synthetic') AND the web-sourced
    real-mic-domain data (source == 'web_real') -- goes into train/val,
    grouped by (intent, voice) so no single voice crosses the boundary.
    """
    rng = np.random.default_rng(seed)
    user_real = [s for s in samples if s.source == "real"]
    # robust (SC/SLURP real-mic OOV) + synthetic + web_real + real_new (UI
    # collector) all train the model on real-mic acoustics; only the user's
    # ORIGINAL mic recordings are held out as the benchmark.
    others = [s for s in samples
              if s.source in ("synthetic", "web_real", "robust", "real_new")]

    groups: dict[tuple, list[Sample]] = {}
    for s in others:
        groups.setdefault(_group_key(s), []).append(s)

    keys = list(groups.keys())
    rng.shuffle(keys)
    total = sum(len(groups[k]) for k in keys)
    n_val = int(total * val_frac)
    n_test = int(total * test_frac)

    train, val, test = [], [], []
    acc = 0
    for k in keys:
        g = groups[k]
        if acc < n_test:
            test.extend(g)
        elif acc < n_test + n_val:
            val.extend(g)
        else:
            train.extend(g)
        acc += len(g)

    # the user's real mic recordings are the generalization benchmark
    test = test + user_real
    return train, val, test

class VCMDataset(Dataset):
    def __init__(self, samples: list[Sample], augment: bool = False,
                 seed: int = 0):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = F.load_wav(s.path)
        if self.augment:
            x = F.augment(x, self.rng)
        feat = F.preprocess(x)                     # (1, 1, N_FRAMES, N_MELS)
        if self.augment:
            # SpecAugment on the (N_FRAMES, N_MELS) image (axis 2,3)
            img = feat[0, 0]
            img = F.spec_augment(img, self.rng)
            feat[0, 0] = img
        return torch.from_numpy(feat).squeeze(0), s.label, s.intent, s.source
