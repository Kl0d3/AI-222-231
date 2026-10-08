"""Gaia's Chamber — 3D smart-lab voice-controlled environment.

Flask backend serving a Three.js 3D scene. Reuses the same CRNN model
and SmartHome simulator as the main Inference UI, but renders state
as a full 3D laboratory with interactive objects.

Supports two audio sources (toggleable from the UI):
  • laptop  — browser captures mic, sends PCM to /api/predict
  • pi      — Pi captures USB mic, runs wake word + VAD + CRNN locally,
              pushes state to browser via WebSocket

Port: 8901 (distinct from the main dashboard on 8000)
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import struct
import threading
import time

import numpy as np
import torch
torch.set_num_threads(4)

from flask import Flask, jsonify, request, send_from_directory

from inference import CRNNPredictor, HF_CKPT
from smarthome import SmartHome

log = logging.getLogger("gaia")
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=HERE,
            static_url_path="/static")

# ── Shared state ──────────────────────────────────────────────────────────
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
    return intent


# ── WebSocket hub (manual RFC 6455 — no extra deps) ───────────────────────
_WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

class WSClient:
    """One connected WebSocket client."""
    def __init__(self, ws):
        self.ws = ws
        self.lock = threading.Lock()
        self.alive = True

    def send(self, obj: dict):
        if not self.alive:
            return
        payload = json.dumps(obj).encode("utf-8")
        header = bytearray([0x81])  # FIN + text
        n = len(payload)
        if n < 126:
            header.append(n)
        elif n < 65536:
            header.append(126)
            header += struct.pack(">H", n)
        else:
            header.append(127)
            header += struct.pack(">Q", n)
        frame = bytes(header) + payload
        with self.lock:
            try:
                self.ws.sendall(frame)
            except OSError:
                self.alive = False

    def close(self):
        self.alive = False
        try:
            self.ws.close()
        except OSError:
            pass


class WSHub:
    """Tracks all connected WebSocket clients."""
    def __init__(self):
        self.clients: list[WSClient] = []
        self.lock = threading.Lock()

    def add(self, client: WSClient):
        with self.lock:
            self.clients.append(client)

    def remove(self, client: WSClient):
        with self.lock:
            if client in self.clients:
                self.clients.remove(client)

    def broadcast(self, obj: dict):
        dead = []
        with self.lock:
            clients = list(self.clients)
        for c in clients:
            c.send(obj)
            if not c.alive:
                dead.append(c)
        for c in dead:
            self.remove(c)

    @property
    def count(self) -> int:
        with self.lock:
            return len(self.clients)


HUB = WSHub()


def _ws_handshake(ws) -> bool:
    """Perform the WebSocket upgrade handshake. Returns True on success."""
    try:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = ws.recv(4096)
            if not chunk:
                return False
            data += chunk
        headers = {}
        lines = data.decode("latin-1").split("\r\n")
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        key = headers.get("sec-websocket-key", "")
        if not key:
            return False
        accept = base64.b64encode(
            hashlib.sha1((key + _WS_MAGIC).encode()).digest()
        ).decode()
        resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        ws.sendall(resp.encode())
        return True
    except Exception:
        return False


def _ws_read_frame(ws):
    """Read one WebSocket frame. Returns (opcode, payload) or None on close."""
    try:
        hdr = ws.recv(2)
        if len(hdr) < 2:
            return None
        b1, b2 = hdr[0], hdr[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack(">H", ws.recv(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", ws.recv(8))[0]
        mask_key = ws.recv(4) if masked else b"\x00\x00\x00\x00"
        payload = bytearray()
        remaining = length
        while remaining > 0:
            chunk = ws.recv(min(remaining, 4096))
            if not chunk:
                return None
            payload += chunk
            remaining -= len(chunk)
        if masked:
            payload = bytes(b ^ mask_key[i % 4]
                            for i, b in enumerate(payload))
        return opcode, bytes(payload)
    except OSError:
        return None


def _ws_client_loop(ws):
    """Background thread: read frames from one client until disconnect."""
    client = WSClient(ws)
    HUB.add(client)
    log.info("WS client connected (%d total)", HUB.count)
    snap = HOME.snapshot()
    snap["mic_source"] = MIC_SOURCE.value
    snap["ws_connected"] = True
    client.send({"type": "init", **snap})
    try:
        while client.alive:
            frame = _ws_read_frame(ws)
            if frame is None:
                break
            opcode, payload = frame
            if opcode == 0x8:  # close
                break
    finally:
        client.close()
        HUB.remove(client)
        log.info("WS client disconnected (%d total)", HUB.count)


# ── Pi-side audio capture (USB mic) ───────────────────────────────────────

SAMPLE_RATE = 16000
BLOCK_SIZE = 4096  # ~256 ms per block

WAKE_RMS_THRESHOLD = 0.015
WAKE_MIN_BLOCKS = 3
WAKE_COOLDOWN_SEC = 2.0

SPEECH_END_SILENCE = 0.8
MAX_CAPTURE_SEC = 5.0


class PiMicListener:
    """Background thread that listens on the Pi's USB mic.

    Pipeline:
      1. Continuously capture audio blocks
      2. Detect wake word via RMS energy
      3. After wake: capture command until silence
      4. Run CRNN on captured command
      5. Execute intent, broadcast state via WebSocket
    """

    def __init__(self, predictor, home, hub):
        self.predictor = predictor
        self.home = home
        self.hub = hub
        self._stop = threading.Event()
        self._thread = None
        self._active = False
        self._lock = threading.Lock()
        self._last_wake = 0.0
        self._capturing = False
        self._cmd_buffer = []
        self._silence_start = None
        self._wake_count = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="pi-mic-listener")
        self._thread.start()
        log.info("Pi mic listener started")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        log.info("Pi mic listener stopped")

    def set_active(self, active: bool):
        with self._lock:
            self._active = active
        log.info("Pi mic %s", "ENABLED" if active else "disabled")

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def _broadcast(self, extra: dict):
        snap = self.home.snapshot()
        snap["mic_source"] = "pi"
        snap["active_model"] = "CRNN-HF-20-GEN"
        self.hub.broadcast({"type": "update", **snap, **extra})

    def _run(self):
        reader = None
        try:
            import sounddevice as sd
            default_in = sd.query_devices(kind='input')
            dev_idx = default_in['index']
            log.info("Using sounddevice, input device: %s (idx=%d)",
                     default_in['name'], dev_idx)
            reader = _SoundDeviceReader(dev_idx)
        except ImportError:
            log.info("sounddevice not installed — falling back to arecord")
            dev = self._find_arecord_device()
            if dev is None:
                log.warning("No USB mic found — Pi mic mode unavailable")
                return
            log.info("Using arecord, device: %s", dev)
            reader = _ArecordReader(dev)
        except Exception as e:
            log.warning("sounddevice init failed (%s) — trying arecord", e)
            dev = self._find_arecord_device()
            if dev is None:
                log.warning("No USB mic found — Pi mic mode unavailable")
                return
            reader = _ArecordReader(dev)

        if reader is None:
            return

        try:
            while not self._stop.is_set():
                block = reader.read(BLOCK_SIZE)
                if block is None:
                    continue
                if not self.active:
                    continue
                self._process_block(block)
        except Exception as e:
            log.error("Pi mic listener error: %s", e, exc_info=True)
        finally:
            reader.close()

    def _process_block(self, block: np.ndarray):
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        now = time.time()

        if not self._capturing:
            if rms > WAKE_RMS_THRESHOLD:
                self._wake_count += 1
            else:
                self._wake_count = 0

            if self._wake_count >= WAKE_MIN_BLOCKS:
                if now - self._last_wake < WAKE_COOLDOWN_SEC:
                    self._wake_count = 0
                    return
                self._last_wake = now
                self._wake_count = 0
                self._capturing = True
                self._cmd_buffer = []
                self._silence_start = None
                log.info("Wake word detected (RMS=%.4f) — capturing", rms)
                self._broadcast({"status": "capturing",
                                 "speak": "Yes?"})
                return
            return

        # Capturing command
        self._cmd_buffer.append(block)
        elapsed = (len(self._cmd_buffer) * BLOCK_SIZE) / SAMPLE_RATE

        if rms > 0.005:
            self._silence_start = None
        else:
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start > SPEECH_END_SILENCE:
                self._finish_capture()
                return

        if elapsed >= MAX_CAPTURE_SEC:
            self._finish_capture()

    def _finish_capture(self):
        self._capturing = False
        if not self._cmd_buffer:
            return
        cmd = np.concatenate(self._cmd_buffer).astype(np.int16)
        self._cmd_buffer = []
        dur = len(cmd) / SAMPLE_RATE
        log.info("Command captured: %.1fs (%d samples)", dur, len(cmd))

        if len(cmd) < 16000:
            self._broadcast({"status": "idle",
                             "speak": "Sorry, I didn't catch that."})
            return

        t0 = time.perf_counter()
        pred = self.predictor.predict(cmd, sr=SAMPLE_RATE)
        latency = (time.perf_counter() - t0) * 1000
        intent = pred["intent"]
        conf = pred["confidence"]
        log.info("CRNN: %s (conf=%.3f, %.0fms)", intent, conf, latency)

        if conf < CONF_THRESHOLD or intent.upper() == "OUT_OF_SCOPE":
            self._broadcast({
                "status": "idle",
                "prediction": pred,
                "low_confidence": True,
                "speak": "Sorry, I didn't catch that. Please try again.",
            })
            return

        device_intent = _translate(intent)
        if device_intent:
            self.home.execute(device_intent)
        self._broadcast({
            "status": "idle",
            "prediction": pred,
            "device_intent": device_intent,
            "speak": self.home.speak_for(device_intent, intent),
        })

    @staticmethod
    def _find_arecord_device():
        import re
        import subprocess
        try:
            out = subprocess.check_output(
                ["arecord", "-l"], text=True, timeout=5
            )
            for line in out.splitlines():
                if "USB" in line.upper() or "MIC" in line.upper():
                    m = re.search(r"card\s+(\d+)", line)
                    if m:
                        return f"hw:{m.group(1)},0"
            return "default"
        except Exception as e:
            log.warning("arecord -l failed: %s", e)
            return None


class _SoundDeviceReader:
    def __init__(self, device=None):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='int16',
            device=device, blocksize=BLOCK_SIZE,
        )
        self._stream.start()

    def read(self, num_frames: int):
        try:
            data, _ = self._stream.read(num_frames)
            return data.flatten()
        except Exception:
            return None

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class _ArecordReader:
    def __init__(self, device: str):
        import subprocess
        self.proc = subprocess.Popen(
            ["arecord", "-q", "-D", device, "-f", "S16_LE",
             "-r", str(SAMPLE_RATE), "-c", "1", "--buffer-time", "256000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def read(self, num_frames: int):
        try:
            raw = self.proc.stdout.read(num_frames * 2)
            if not raw:
                return None
            return np.frombuffer(raw, dtype=np.int16)
        except Exception:
            return None

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            pass


# ── Mic source toggle ─────────────────────────────────────────────────────
class MicSource:
    LAPTOP = "laptop"
    PI = "pi"

    def __init__(self):
        self._value = self.LAPTOP
        self._lock = threading.Lock()

    @property
    def value(self) -> str:
        with self._lock:
            return self._value

    def set(self, source: str):
        with self._lock:
            self._value = source if source in (self.LAPTOP, self.PI) else self.LAPTOP


MIC_SOURCE = MicSource()
PI_MIC = PiMicListener(_PREDICTOR, HOME, HUB)


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/state")
def api_state():
    snap = HOME.snapshot()
    snap["active_model"] = "CRNN-HF-20-GEN"
    snap["intents"] = _INTENTS
    snap["mic_source"] = MIC_SOURCE.value
    snap["pi_mic_available"] = PI_MIC.running
    return jsonify(snap)


@app.route("/api/mic-source")
def get_mic_source():
    return jsonify({"source": MIC_SOURCE.value,
                    "pi_available": PI_MIC.running})


@app.route("/api/mic-source", methods=["POST"])
def set_mic_source():
    data = request.get_json(force=True)
    source = data.get("source", "laptop")
    if source not in (MicSource.LAPTOP, MicSource.PI):
        return jsonify({"error": "source must be 'laptop' or 'pi'"}), 400

    if source == MicSource.PI and not PI_MIC.running:
        return jsonify({"error": "Pi mic not available (no USB mic detected?)",
                        "source": MIC_SOURCE.value}), 503

    MIC_SOURCE.set(source)
    PI_MIC.set_active(source == MicSource.PI)
    log.info("Mic source switched to: %s", source)
    HUB.broadcast({"type": "mic_source", "source": source})
    return jsonify({"source": source, "ok": True})


CONF_THRESHOLD = 0.30


@app.route("/api/predict", methods=["POST"])
def api_predict():
    raw = request.get_data()
    if len(raw) < 1600:
        return jsonify({"error": "audio too short"}), 400
    pcm = np.frombuffer(raw, dtype=np.int16)
    pred = _PREDICTOR.predict(pcm, sr=16000)
    intent = pred["intent"]
    conf = pred["confidence"]

    if conf < CONF_THRESHOLD:
        snap = HOME.snapshot()
        return jsonify({
            "prediction": pred, "state": snap,
            "active_model": "CRNN-HF-20-GEN",
            "device_intent": None, "low_confidence": True,
            "speak": "Sorry, I didn't catch that. Please try again.",
        })

    if intent.upper() == "OUT_OF_SCOPE":
        snap = HOME.snapshot()
        return jsonify({
            "prediction": pred, "state": snap,
            "active_model": "CRNN-HF-20-GEN",
            "device_intent": None, "low_confidence": True,
            "out_of_scope": True,
            "speak": "Sorry, I didn't catch that. Please try again.",
        })

    device_intent = _translate(intent)
    if device_intent is not None:
        HOME.execute(device_intent)
    snap = HOME.snapshot()
    result = {
        "prediction": pred, "state": snap,
        "active_model": "CRNN-HF-20-GEN",
        "device_intent": device_intent,
        "speak": HOME.speak_for(device_intent, intent),
    }
    HUB.broadcast({"type": "update", **result})
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    HOME.reset()
    snap = HOME.snapshot()
    HUB.broadcast({"type": "reset", "state": snap})
    return jsonify({"state": snap, "reset": True})


@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json(force=True)
    intent = data.get("intent", "")
    arg = data.get("arg", "")
    device_intent = _translate(intent)
    if device_intent is not None:
        HOME.execute(device_intent, arg)
    snap = HOME.snapshot()
    result = {"state": snap, "executed": device_intent,
              "speak": HOME.speak_for(device_intent, intent)}
    HUB.broadcast({"type": "update", **result})
    return jsonify(result)


@app.route("/api/weather")
def api_weather():
    from smarthome import _fetch_weather, OWM_CITY, OWM_API_KEY
    if not OWM_API_KEY:
        return jsonify({"error": "OWM_API_KEY not set", "city": OWM_CITY}), 503
    data = _fetch_weather()
    if data is None:
        return jsonify({"error": "Failed to fetch weather", "city": OWM_CITY}), 502
    return jsonify({"city": OWM_CITY, "source": "openweathermap", **data})


# ── WebSocket endpoint ────────────────────────────────────────────────────
@app.route("/ws")
def websocket():
    environ = request.environ
    try:
        import socket as _socket
        handler = environ.get("werkzeug.server_handler")
        if handler:
            sock = handler.connection
        else:
            sock = environ.get("werkzeug.socket")
        if sock is None:
            return jsonify({"error": "WebSocket not supported"}), 501
        if _ws_handshake(sock):
            _ws_client_loop(sock)
    except Exception as e:
        log.debug("WS error: %s", e)
    return "", 200


# ── Startup ───────────────────────────────────────────────────────────────
def _startup():
    try:
        PI_MIC.start()
    except Exception as e:
        log.warning("Pi mic listener failed to start: %s", e)


if __name__ == "__main__":
    from smarthome import OWM_API_KEY, OWM_CITY
    print(f"[Gaia's Chamber] Serving on http://0.0.0.0:8901")
    print(f"[Gaia's Chamber] Model: {HF_CKPT}")
    print(f"[Gaia's Chamber] Intents ({len(_INTENTS)}): {_INTENTS}")
    if OWM_API_KEY:
        print(f"[Gaia's Chamber] Weather: OpenWeather API enabled for {OWM_CITY}")
    else:
        print(f"[Gaia's Chamber] Weather: OWM_API_KEY not set — using fallback")
    print(f"[Gaia's Chamber] WebSocket: ws://0.0.0.0:8901/ws")
    print(f"[Gaia's Chamber] Mic toggle: /api/mic-source (laptop | pi)")

    _startup()
    app.run(host="0.0.0.0", port=8901, debug=False, threaded=True)
