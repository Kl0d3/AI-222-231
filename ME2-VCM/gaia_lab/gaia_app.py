"""Gaia's Chamber — 3D smart-lab voice-controlled environment.

Flask backend serving a Three.js 3D scene. Reuses the same CRNN model
and SmartHome simulator as the main Inference UI, but renders state
as a full 3D laboratory with interactive objects.

Port: 8901 (distinct from the main dashboard on 8000)
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import torch
torch.set_num_threads(4)

from flask import Flask, jsonify, request, send_from_directory

from inference import CRNNPredictor, HF_CKPT
from smarthome import SmartHome

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=HERE,
            static_url_path="/static")

# Shared state — one instance, protected by its own lock
HOME = SmartHome()
_PREDICTOR = CRNNPredictor(HF_CKPT, device="cpu")
_INTENTS = _PREDICTOR.intents

# Intent translation: HF 20-class model -> 12 device intents
HF_TO_DEVICE = {
    "PLAY_MUSIC":     "play_music",
    "PAUSE":          "media_control",
    "NEXT":           "media_control",
    "STOP":           "media_control",
    "VOLUME_UP":      "media_control",
    "VOLUME_DOWN":    "media_control",
    "LIGHT_ON":       "light_on",
    "LIGHT_OFF":      "light_off",
    "BRIGHTNESS":     "dim_lights",
    "COLOR":          "dim_lights",
    "TEMPERATURE":    "set_temperature",
    "ALARM":          "set_alarm",
    "TIMER":          "set_timer",
    "TIME":           "get_time",
    "WEATHER":        "get_weather",
    "CALL":           "make_call",
    "MESSAGE":        "make_call",
    "CREATE_REMINDER":"manage_reminders",
    "LIST_REMINDERS": "manage_reminders",
    "OUT_OF_SCOPE":   None,
}

def _translate(intent: str) -> str | None:
    """Map model intent to device vocabulary. Returns None to reject."""
    up = intent.upper()
    if up in HF_TO_DEVICE:
        return HF_TO_DEVICE[up]
    # Already device vocabulary (lowercase) — pass through
    return intent


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/state")
def api_state():
    snap = HOME.snapshot()
    snap["active_model"] = "CRNN-HF-20-GEN"
    snap["intents"] = _INTENTS
    return jsonify(snap)


# Confidence threshold: below this, the model is guessing.
# Don't execute the intent — tell the user to repeat.
CONF_THRESHOLD = 0.30


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept raw PCM (16 kHz mono int16) as application/octet-stream."""
    raw = request.get_data()
    if len(raw) < 1600:  # < 50 ms
        return jsonify({"error": "audio too short"}), 400
    pcm = np.frombuffer(raw, dtype=np.int16)
    pred = _PREDICTOR.predict(pcm, sr=16000)
    intent = pred["intent"]
    conf = pred["confidence"]

    # Low-confidence guard: if the model is unsure, don't execute.
    # This prevents the "works once then breaks" perception — instead of
    # silently executing a wrong intent, we ask the user to repeat.
    if conf < CONF_THRESHOLD:
        snap = HOME.snapshot()
        return jsonify({
            "prediction": pred,
            "state": snap,
            "active_model": "CRNN-HF-20-GEN",
            "device_intent": None,
            "low_confidence": True,
            "speak": "Sorry, I didn't catch that. Please try again.",
        })

    # OUT_OF_SCOPE guard: the model is confident, but it heard something that
    # isn't a known command. Treat it the same as low-confidence — ask to
    # repeat instead of silently doing nothing. (Without this, a confident
    # OOS prediction bypassed the threshold and the lab just... sat there.)
    if intent.upper() == "OUT_OF_SCOPE":
        snap = HOME.snapshot()
        return jsonify({
            "prediction": pred,
            "state": snap,
            "active_model": "CRNN-HF-20-GEN",
            "device_intent": None,
            "low_confidence": True,
            "out_of_scope": True,
            "speak": "Sorry, I didn't catch that. Please try again.",
        })

    # Translate to device vocabulary
    device_intent = _translate(intent)
    if device_intent is not None:
        HOME.execute(device_intent)
    snap = HOME.snapshot()
    return jsonify({
        "prediction": pred,
        "state": snap,
        "active_model": "CRNN-HF-20-GEN",
        "device_intent": device_intent,
        "speak": HOME.speak_for(device_intent, intent),
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Restore the lab to its pristine state (lights off, media stopped, etc.)."""
    HOME.reset()
    snap = HOME.snapshot()
    return jsonify({"state": snap, "reset": True})


@app.route("/api/command", methods=["POST"])
def api_command():
    """Direct text command (for testing / accessibility)."""
    data = request.get_json(force=True)
    intent = data.get("intent", "")
    arg = data.get("arg", "")
    device_intent = _translate(intent)
    if device_intent is not None:
        HOME.execute(device_intent, arg)
    snap = HOME.snapshot()
    return jsonify({"state": snap, "executed": device_intent,
                    "speak": HOME.speak_for(device_intent, intent)})


@app.route("/api/weather")
def api_weather():
    """Test endpoint: fetch and return current weather for Cebu City."""
    from smarthome import _fetch_weather, OWM_CITY, OWM_API_KEY
    if not OWM_API_KEY:
        return jsonify({"error": "OWM_API_KEY not set", "city": OWM_CITY}), 503
    data = _fetch_weather()
    if data is None:
        return jsonify({"error": "Failed to fetch weather", "city": OWM_CITY}), 502
    return jsonify({"city": OWM_CITY, "source": "openweathermap", **data})


if __name__ == "__main__":
    from smarthome import OWM_API_KEY, OWM_CITY
    print(f"[Gaia's Chamber] Serving on http://0.0.0.0:8901")
    print(f"[Gaia's Chamber] Model: {HF_CKPT}")
    print(f"[Gaia's Chamber] Intents ({len(_INTENTS)}): {_INTENTS}")
    if OWM_API_KEY:
        print(f"[Gaia's Chamber] Weather: OpenWeather API enabled for {OWM_CITY}")
    else:
        print(f"[Gaia's Chamber] Weather: OWM_API_KEY not set — using fallback")
    app.run(host="0.0.0.0", port=8901, debug=False, threaded=True)
