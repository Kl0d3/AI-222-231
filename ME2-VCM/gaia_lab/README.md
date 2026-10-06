# Gaia's Chamber — Voice-Controlled 3D Smart Lab

A fully on-device, voice-controlled 3D laboratory. Say *"Hey Gaia, turn on the
lights"* and the 3D scene responds — no cloud, no LLM, just a tiny CRNN
classifier running on CPU.

Built for **ME2 · Voice Controlled Smart Device** (AI 222-231, UPD).

---

## Architecture

```
Browser Mic (Web Speech API)
  │
  ├─→ Wake word: "Hey Gaia" (fuzzy Levenshtein match, dist ≤ 1)
  │     └─→ 3-second command window opens
  │
  ├─→ PATH 1: Text regex match (textToIntent)
  │     └─→ POST /api/command → SmartHome.execute()
  │     Reliability: ~100% when regex hits
  │
  └─→ PATH 2: Audio → CRNN model
        └─→ POST /api/predict → CRNNPredictor.predict()
              └─→ Confidence guard (conf < 0.30 → "didn't catch that")
              └─→ OOS guard (OUT_OF_SCOPE → "didn't catch that")
              └─→ SmartHome.execute()
        Reliability: ~65-90% depending on intent

SmartHome (in-memory simulator)
  └─→ 12 device intents: lights, brightness, color, temperature,
      alarm, timer, time, weather, music, media control,
      call, reminders
  └─→ 3D scene (Three.js r164) renders state live
```

## The Model: CRNN-HF-20-GEN

| Property | Value |
|----------|-------|
| Architecture | CRNNBlock (Conv2d+BN+ReLU ×4, MaxPool ×2) → BiGRU(128, 2-layer) → FC head |
| Parameters | 988,980 (3.8 MB fp32) |
| Input | 16 kHz mono → log-mel spectrogram (40 mels × 99 frames, ~990 ms) |
| Intents | 20 classes (12 device + 8 auxiliary + OUT_OF_SCOPE) |
| Training | 2-phase (20+10 epochs), AdamW, cosine LR, label smoothing 0.1, mixup α=0.2, SWA window 5 |
| HF test accuracy | **86.5%** |
| Mark validation (real voice) | **71.7%** |
| Best per-intent | dim_lights 90.3%, set_alarm 86.0%, set_timer 78.6% |
| Weakest per-intent | make_call 40.5%, manage_reminders 31.2% |
| Inference latency | ~10-50 ms on CPU (4 threads) |

### 20 Intent Classes

`PLAY_MUSIC`, `PAUSE`, `NEXT`, `STOP`, `VOLUME_UP`, `VOLUME_DOWN`,
`LIGHT_ON`, `LIGHT_OFF`, `BRIGHTNESS`, `COLOR`, `TEMPERATURE`, `ALARM`,
`TIMER`, `TIME`, `WEATHER`, `CALL`, `MESSAGE`, `CREATE_REMINDER`,
`LIST_REMINDERS`, `OUT_OF_SCOPE`

### 12 Device Intents (SmartHome)

`play_music`, `media_control`, `light_on`, `light_off`, `dim_lights`,
`set_temperature`, `set_alarm`, `set_timer`, `get_time`, `get_weather`,
`make_call`, `manage_reminders`

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server (from ME2-VCM/gaia_lab/)
python gaia_app.py
# → http://localhost:8901
```

Open the URL in a browser, click the mic, say **"Hey Gaia"**, then a command:
- *"Turn on the lights"*
- *"Set temperature to 22 degrees"*
- *"Play some music"*
- *"What's the weather?"*

## Project Structure

```
ME2-VCM/
├── gaia_lab/                  ← this directory (Gaia's Chamber app)
│   ├── README.md              ← this file
│   ├── requirements.txt       ← Python dependencies
│   ├── gaia_app.py            ← Flask server (port 8901)
│   ├── index.html             ← 3D lab UI (Three.js r164, ~1500 lines)
│   ├── music.wav              ← ambient lab music
│   ├── inference.py           ← CRNNPredictor + VCMPredictor
│   ├── crnn_model.py          ← CRNN + CRNNBlock architectures
│   ├── features.py            ← log-mel extraction (40 mels × 99 frames)
│   ├── smarthome.py           ← smart-home simulator (12 intents)
│   ├── dataset.py             ← dataset loading + speaker-aware splits
│   ├── model.py               ← TinyVCM CNN (legacy 12-intent)
│   ├── AUDIT.md               ← end-to-end audit report
│   └── deploy_pi.sh           ← Raspberry Pi 5 deployment
├── models/
│   ├── crnn_hf_20_gen.pth     ← CRNN-HF-20-GEN checkpoint (3.8 MB)
│   └── crnn_hf_20_gen_eval.json
├── train_crnn_hf.py           ← training script
├── train_crnn_generalized.py  ← generalized training
├── train_crnn_golden.py       ← golden training
├── train_crnn_maxgen.py       ← max-generalization training
├── train_crnn_with_real.py    ← real-voice fine-tuning
├── finetune_kent.py           ← speaker-specific fine-tuning
├── infer_ui.py                ← Gradio inference UI
├── eval_real_voice.py         ← real-voice evaluation
├── features.py                ← feature extraction (shared)
├── crnn_model.py              ← CRNN architecture (shared)
└── ...              ← Ambient lab music (1.4 MB)
│   ├── inference.py           ← CRNNPredictor + VCMPredictor
│   ├── crnn_model.py          ← CRNN + CRNNBlock architectures
│   ├── model.py               ← TinyVCM CNN (legacy 12-intent model)
│   ├── features.py            ← Log-mel feature extraction (numpy/scipy)
│   ├── smarthome.py           ← Smart-home simulator (12 intents)
│   └── dataset.py             ← Dataset loading + speaker-aware splits
├── models/
│   ├── crnn_hf_20_gen.pth     ← Trained CRNN-HF-20-GEN checkpoint (3.8 MB)
│   └── crnn_hf_20_gen_eval.json ← Full eval metrics + training history
├── docs/
│   └── AUDIT.md               ← End-to-end audit report (Oct 6, 2026)
├── scripts/
│   └── deploy_pi.sh           ← Raspberry Pi 5 deployment
└── data/
    └── README.md              ← Dataset structure (parquet files)
```

## Wake Word System

| Parameter | Value |
|-----------|-------|
| Targets | gaia, gaiya, gaiaa, gaja, gayah, heygaia, heigaia |
| Fuzzy match | Levenshtein distance ≤ 1 |
| Min word length | 4 characters (filters "the", "and", "nah") |
| Cooldown | 1,500 ms |
| Command window | 3,000 ms |
| Silence timeout | 6,000 ms (auto-return to idle) |
| Listen window | 2,500 ms |
| Gain boost | +6 dB (2.0×) — lifts distant speakers |
| Priming guard | 800 ms (skips post-respawn ASR noise) |
| Dead zone | 2,300 ms (priming + cooldown) |
| Watchdog | `ensureWakeRecognition()` every 4s |
| TTS gate | Waits for `ttsPlaying=false` + 500ms grace before respawn |
| Self-healing | Generation counter, transcript buffer clear, visibilitychange, pageshow |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | 3D lab UI |
| `/api/state` | GET | Current smart-home state + active model |
| `/api/predict` | POST | Raw PCM (16 kHz int16) → intent prediction |
| `/api/command` | POST | Direct text command (JSON: `{intent, arg}`) |
| `/api/reset` | POST | Reset lab to pristine state |
| `/api/weather` | GET | Fetch current weather (OpenWeather API) |

## Deployment

### Local / Server

```bash
cd src && python gaia_app.py
```

### Raspberry Pi 5 (4 GB)

```bash
./scripts/deploy_pi.sh
PORT=8901 python3 src/gaia_app.py
```

### Cloudflare Tunnel (remote access)

```bash
cloudflared tunnel --url http://localhost:8901
```

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0 (CPU is fine)
- NumPy, SciPy, SoundFile, Flask
- Browser with Web Speech API (Chrome recommended)

## Limitations

- **Model accuracy varies by intent**: `dim_lights` 90.3% vs `make_call` 40.5%.
  The confidence guard (0.30) rejects low-confidence predictions, which means
  ~40% of real clips get "didn't catch that" for weaker intents.
- **Single-speaker bias**: The Mark validation set is one person's voice.
  Multi-speaker accuracy is untested.
- **`createScriptProcessor` is deprecated**: Works in all current browsers
  but will eventually be removed in favor of `AudioWorklet`.
- **Flask dev server**: Single-threaded. Fine for one user; would bottleneck
  under concurrent load.

## License

Course project — AI 222-231, UP Diliman.
