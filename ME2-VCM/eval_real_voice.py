"""Evaluate the CRNN on REAL user voice (Kent's mic recordings).

Loads real_metadata.csv (12 device classes, ~269 clips), maps each real class
to the CRNN's 19 coarse intents, scores every clip, and reports:
  - overall accuracy (predicted intent == expected intent)
  - per-class accuracy + confusion detail
  - the full per-clip dump (transcript, expected, predicted, confidence)

Usage:
    python eval_real_voice.py --ckpt models/crnn_ai231_42.pth --device cuda
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict

import numpy as np
import torch

import features as F
from crnn_model import CRNN, count_params

# The CRNN's canonical 19-intent ordering (must match training).
INTENTS = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC",
    "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
    "WEATHER",
]
INTENT2IDX = {i: k for k, i in enumerate(INTENTS)}

# Map each REAL-VOICE class (Kent's 12 device categories) to the CRNN intent(s)
# that count as "correct". A real clip is right if its predicted intent is in
# the acceptable set. Some real classes are ambiguous (e.g. media_control covers
# pause/stop/next/volume), so we accept the whole family.
REAL_TO_EXPECTED = {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/crnn_ai231_42.pth")
    ap.add_argument("--meta", default="real_data/real_metadata.csv")
    ap.add_argument("--data", default="real_data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="logs/real_voice_eval.json")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = CRNN(n_intents=len(INTENTS))
    sd = torch.load(args.ckpt, map_location=device)
    # support raw state_dict, {'state_dict':...}, and {'model': {...}} wrappers
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            inner = sd["model"]
            if any(k.startswith(("cnn.", "rnn.", "head.")) for k in inner):
                sd = inner
            elif "state_dict" in inner:
                sd = inner["state_dict"]
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"[eval] loaded {args.ckpt} | params={count_params(model):,} | device={device}")

    rows = list(csv.DictReader(open(args.meta)))
    results = []
    per_class = defaultdict(lambda: {"n": 0, "ok": 0, "preds": defaultdict(int)})

    with torch.no_grad():
        for r in rows:
            fname = r["file_name"]
            cls = r["intent"].strip()
            expected = REAL_TO_EXPECTED.get(cls, {cls})
            path = os.path.join(args.data, fname)
            if not os.path.exists(path):
                continue
            x = F.load_wav(path)
            feat = F.preprocess(x)              # (1, 1, N_FRAMES, N_MELS)
            xt = torch.from_numpy(feat).unsqueeze(0).to(device)
            if xt.dim() == 5:  # (1,1,1,99,40) -> drop extra leading dim
                xt = xt.squeeze(1)
            out = model(xt)[0]
            probs = torch.softmax(out, dim=0)
            idx = int(probs.argmax())
            pred = INTENTS[idx]
            conf = float(probs[idx])
            ok = pred in expected
            results.append({
                "file": fname, "class": cls, "transcript": r["transcript"],
                "expected": sorted(expected), "predicted": pred,
                "confidence": round(conf, 4), "correct": ok,
            })
            pc = per_class[cls]
            pc["n"] += 1
            pc["ok"] += ok
            pc["preds"][pred] += 1

    total = len(results)
    correct = sum(r["correct"] for r in results)
    acc = correct / total * 100 if total else 0.0

    print("\n" + "=" * 70)
    print(f"REAL-VOICE EVALUATION  (n={total} clips)")
    print("=" * 70)
    print(f"OVERALL ACCURACY: {correct}/{total} = {acc:.1f}%\n")
    print(f"{'CLASS':<18}{'ACC':>8}   PREDICTIONS")
    print("-" * 70)
    for cls in sorted(per_class):
        pc = per_class[cls]
        acc_c = pc["ok"] / pc["n"] * 100
        preds = ", ".join(f"{k}:{v}" for k, v in sorted(pc["preds"].items()))
        flag = "" if acc_c >= 80 else ("  <-- LOW" if acc_c < 50 else "  <-- weak")
        print(f"{cls:<18}{acc_c:>7.0f}%   {preds}{flag}")
    print("-" * 70)

    # save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "checkpoint": os.path.abspath(args.ckpt),
        "n_clips": total, "correct": correct, "accuracy_pct": round(acc, 2),
        "per_class": {k: {
            "n": v["n"], "ok": v["ok"],
            "acc_pct": round(v["ok"] / v["n"] * 100, 1),
            "predictions": dict(v["preds"]),
        } for k, v in per_class.items()},
        "clips": results,
    }
    json.dump(payload, open(args.out, "w"), indent=2)
    print(f"\n[eval] saved -> {args.out}")

    # print the wrong ones for inspection
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n--- {len(wrong)} WRONG PREDICTIONS (sample) ---")
        for r in wrong[:40]:
            print(f"  [{r['class']}] '{r['transcript']}' -> {r['predicted']} ({r['confidence']:.2f})  exp={r['expected']}")


if __name__ == "__main__":
    main()
