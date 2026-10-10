"""Standalone inference for the VCM.

Loads the trained checkpoint and exposes predict() on raw 16 kHz PCM.
Runs on CPU (no CUDA needed) so it works on a Raspberry Pi 5.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

import features as F
from model import TinyVCM
from crnn_model import CRNN

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "models", "tiny_vcm.pth")
CRNN_CKPT = os.path.join(HERE, "models", "crnn_ai231_42.pth")


def _resolve_ckpt(candidates):
    """Return the first existing checkpoint path, else raise with a clear hint."""
    for p in candidates:
        if os.path.isfile(p):
            return p
    tried = "\n  ".join(os.path.normpath(p) for p in candidates)
    raise FileNotFoundError(
        "Model checkpoint not found. Looked for:\n  " + tried +
        "\nCopy crnn_hf_20_gen.pth next to the app (see README)."
    )


# The HF 20-class checkpoint lives in ../models/ in the repo layout but in
# ./models/ when the gaia_lab folder is copied flat onto the Pi. Accept both.
HF_CKPT = _resolve_ckpt([
    os.path.join(HERE, "models", "crnn_hf_20_gen.pth"),        # flat (Pi)
    os.path.join(HERE, "..", "models", "crnn_hf_20_gen.pth"),  # repo
])


class VCMPredictor:
    def __init__(self, ckpt_path: str = DEFAULT_CKPT, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.intents = ckpt.get("intents") or [
            "dim_lights", "get_time", "get_weather", "light_off", "light_on",
            "make_call", "manage_reminders", "media_control", "play_music",
            "set_alarm", "set_temperature", "set_timer"]
        self.model = TinyVCM(n_mels=ckpt.get("n_mels", F.N_MELS),
                             n_frames=ckpt.get("n_frames", F.N_FRAMES),
                             n_intents=len(self.intents)).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.meta = ckpt

    @torch.no_grad()
    def predict(self, pcm: np.ndarray, sr: int = 16000):
        """pcm: float32/int16 mono audio. Returns dict(intent, confidence, probs)."""
        t0 = time.perf_counter()
        x = pcm.astype(np.float32)
        if x.dtype == np.int16 or (x.max() > 2.0):
            x = x / 32768.0
        if sr != F.SAMPLE_RATE:
            x = F.resample(x, sr, F.SAMPLE_RATE)
        feat = torch.from_numpy(F.preprocess(x)).to(self.device)
        logits = self.model(feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return {
            "intent": self.intents[idx],
            "confidence": float(probs[idx]),
            "probs": {self.intents[i]: float(probs[i]) for i in range(len(self.intents))},
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


def load_wav_predictor(ckpt_path=DEFAULT_CKPT):
    return VCMPredictor(ckpt_path)


class CRNNPredictor:
    """Predictor for the CRNN-AI231 19-intent model (A/B vs the R12 CNN).

    Same feature pipeline as VCMPredictor (F.preprocess -> 40-mel, 99-frame,
    onset-cropped, normalised) so the two models receive identical inputs and
    the A/B comparison is apples-to-apples. Only the classifier head and the
    intent vocabulary differ.
    """

    def __init__(self, ckpt_path: str = CRNN_CKPT, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.intents = ckpt.get("intents") or []
        sd = ckpt.get("state_dict") or ckpt.get("model")
        if sd is None:
            raise KeyError(f"No state_dict or model key in checkpoint. Keys: {list(ckpt.keys())}")
        # Two CRNN layouts exist: the flat Sequential (cnn.0.weight) and the
        # ConvBlock wrapper (cnn.0.conv.weight). Pick whichever matches the
        # saved keys so both golden and *_gen checkpoints load.
        use_blocks = any(".conv." in k for k in sd.keys())
        from crnn_model import CRNNBlock
        cls = CRNNBlock if use_blocks else CRNN
        self.arch = "CRNNBlock" if use_blocks else "CRNN"
        self.model = cls(n_mels=ckpt.get("n_mels", F.N_MELS),
                         n_frames=ckpt.get("n_frames", F.N_FRAMES),
                         n_intents=len(self.intents)).to(device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.meta = ckpt

    @torch.no_grad()
    def predict(self, pcm: np.ndarray, sr: int = 16000):
        """pcm: float32/int16 mono audio. Returns dict(intent, confidence, probs)."""
        t0 = time.perf_counter()
        x = pcm.astype(np.float32)
        if x.dtype == np.int16 or (x.max() > 2.0):
            x = x / 32768.0
        if sr != F.SAMPLE_RATE:
            x = F.resample(x, sr, F.SAMPLE_RATE)
        feat = torch.from_numpy(F.preprocess(x)).to(self.device)
        logits = self.model(feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return {
            "intent": self.intents[idx],
            "confidence": float(probs[idx]),
            "probs": {self.intents[i]: float(probs[i]) for i in range(len(self.intents))},
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
