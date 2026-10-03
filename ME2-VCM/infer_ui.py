#!/usr/bin/env python3
"""Gradio inference UI for the AI231-ME2 20-class CRNN voice-command model.

Usage:
    python infer_ui.py --checkpoint models/crnn_hf_20.pth --port 7860

Features:
    - Mic recording (browser)
    - File upload (wav/mp3/flac/ogg)
    - Top-5 predictions with confidence bars
    - Out-of-scope detection
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# --- project imports ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import SAMPLE_RATE, N_MELS, N_FRAMES, HOP, preprocess
from crnn_model import CRNN

# 20 classes (must match train_crnn_hf.py)
INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER", "OUT_OF_SCOPE",
]

INTENT_EMOJI = {
    "ALARM": "⏰", "BRIGHTNESS": "💡", "CALL": "📞", "COLOR": "🎨",
    "CREATE_REMINDER": "📝", "LIGHT_OFF": "🌑", "LIGHT_ON": "☀️",
    "LIST_REMINDERS": "📋", "MESSAGE": "💬", "NEXT": "⏭️", "PAUSE": "⏸️",
    "PLAY_MUSIC": "🎵", "STOP": "⏹️", "TEMPERATURE": "🌡️", "TIME": "🕐",
    "TIMER": "⏱️", "VOLUME_DOWN": "🔉", "VOLUME_UP": "🔊",
    "WEATHER": "🌤️", "OUT_OF_SCOPE": "🚫",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_model: CRNN | None = None
_device: torch.device | None = None


def load_model(checkpoint_path: str, device: str = "cpu"):
    global _model, _device
    _device = torch.device(device)
    _model = CRNN(n_mels=N_MELS, n_frames=N_FRAMES, n_intents=len(INTENTS))
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    _model.load_state_dict(state)
    _model.to(_device).eval()
    print(f"[model] loaded {checkpoint_path} on {_device}")
    return _model


# ---------------------------------------------------------------------------
# Audio loading helpers
# ---------------------------------------------------------------------------
def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    from scipy.signal import resample_poly
    import math
    g = math.gcd(src_sr, dst_sr)
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def _load_audio(data: np.ndarray, sr: int) -> np.ndarray:
    """Ensure mono float32 @ 16 kHz."""
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SAMPLE_RATE:
        data = _resample(data, sr, SAMPLE_RATE)
    return data


def _gradio_load(audio_input):
    """Handle Gradio audio input: (sr, ndarray) tuple or file path string."""
    if audio_input is None:
        return None
    if isinstance(audio_input, tuple):
        sr, data = audio_input
        return _load_audio(data, sr)
    if isinstance(audio_input, str):
        data, sr = sf.read(audio_input, dtype="float32")
        return _load_audio(data, sr)
    return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(audio_input):
    """Take audio (mic or file) -> top-5 prediction table + result text."""
    if _model is None:
        return "⚠️ Model not loaded yet.", ""

    x = _gradio_load(audio_input)
    if x is None or x.size == 0:
        return "⚠️ No audio received. Please record or upload a clip.", ""

    # Preprocess -> (1, N_MELS, N_FRAMES)
    feat = preprocess(x)  # (1, 40, 99)
    tensor = torch.from_numpy(feat).unsqueeze(0).to(_device)  # (1,1,40,99)

    logits = _model(tensor)
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    top5_idx = np.argsort(probs)[::-1][:5]

    best_idx = int(top5_idx[0])
    best_intent = INTENTS[best_idx]
    best_conf = float(probs[best_idx])

    # Build result text
    emoji = INTENT_EMOJI.get(best_intent, "")
    if best_intent == "OUT_OF_SCOPE":
        result = f"🚫 **Out of Scope**\nConfidence: {best_conf:.1%}"
    else:
        result = f"{emoji} **{best_intent.replace('_', ' ').title()}**\nConfidence: {best_conf:.1%}"

    # Duration info
    dur = len(x) / SAMPLE_RATE
    result += f"\nDuration: {dur:.2f}s"

    # Build top-5 table
    rows = []
    for rank, idx in enumerate(top5_idx, 1):
        intent = INTENTS[idx]
        conf = float(probs[idx])
        em = INTENT_EMOJI.get(intent, "")
        bar = "█" * int(conf * 30)
        rows.append([rank, f"{em} {intent.replace('_', ' ').title()}", f"{conf:.1%}", bar])

    header = ["Rank", "Intent", "Confidence", ""]
    return result, [header] + rows


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
import gradio as gr


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI231-ME2 Voice Command Recognizer") as demo:
        gr.Markdown(
            "# 🎙️ AI231-ME2 Voice Command Recognizer\n"
            "**20-class CRNN** — 19 smart-home intents + out-of-scope rejection\n"
            "Trained on 10,682 clips / 315 speakers · Test accuracy: **86.5%** (speaker-disjoint)"
        )

        with gr.Row():
            # --- Left column: input ---
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Record or Upload")
                mic_input = gr.Audio(
                    label="Microphone Recording",
                    type="numpy",
                    
                )
                file_input = gr.Audio(
                    label="Upload Audio File (wav / mp3 / flac / ogg)",
                    type="filepath",
                    sources=["upload"],
                )
                predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")

            # --- Right column: output ---
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Result")
                result_md = gr.Markdown("*Waiting for input...*")
                top5_table = gr.Dataframe(
                    headers=["Rank", "Intent", "Confidence", ""],
                    label="Top-5 Predictions",
                    interactive=False,
                    wrap=True,
                )

        # Wire up: predict from mic OR file
        def _predict_from_mic(audio):
            if audio is not None:
                return predict(audio)
            return "⚠️ Please record something first.", ""

        def _predict_from_file(filepath):
            if filepath:
                return predict(filepath)
            return "⚠️ Please upload a file first.", ""

        mic_input.change(_predict_from_mic, mic_input, [result_md, top5_table])
        file_input.change(_predict_from_file, file_input, [result_md, top5_table])
        predict_btn.click(_predict_from_mic, mic_input, [result_md, top5_table])
        predict_btn.click(_predict_from_file, file_input, [result_md, top5_table])

        gr.Markdown(
            "---\n"
            "**Intents:** " + " · ".join(
                f"{INTENT_EMOJI.get(i, '')} {i.replace('_', ' ').title()}"
                for i in INTENTS if i != "OUT_OF_SCOPE"
            ) + " · 🚫 Out of Scope"
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/crnn_hf_20.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create public gradio share link")
    args = ap.parse_args()

    load_model(args.checkpoint, args.device)
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
