"""Smart-home simulator controlled by the VCM.

The model emits one of 12 intents. This module maps each intent to a concrete
action on an in-memory "house" (lights, thermostat, timers, alarms, reminders,
media, phone, clock, weather). It is a simulation: nothing physical happens,
but the state changes are observable through the web dashboard and the
/activity feed.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request


# ── OpenWeather API ────────────────────────────────────────────────────────
OWM_API_KEY = os.environ.get("OWM_API_KEY", "614a4f8a28a99c725b71e7cdf342f042")
OWM_CITY = "Cebu City"
OWM_URL = (
    "https://api.openweathermap.org/data/2.5/weather"
    "?q={city}&appid={key}&units=metric"
)
_WEATHER_CACHE: dict = {"ts": 0.0, "data": None}
_WEATHER_TTL = 300  # cache for 5 minutes


def _fetch_weather() -> dict | None:
    """Fetch current weather for OWM_CITY from OpenWeather API.

    Returns a dict with keys: temp, feels_like, humidity, wind_speed,
    condition, icon, city — or None on failure.
    Results are cached for _WEATHER_TTL seconds.
    """
    now = time.time()
    if _WEATHER_CACHE["data"] and (now - _WEATHER_CACHE["ts"]) < _WEATHER_TTL:
        return _WEATHER_CACHE["data"]

    if not OWM_API_KEY:
        return None

    url = OWM_URL.format(city=urllib.parse.quote(OWM_CITY), key=OWM_API_KEY)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GaiaLab/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = json.loads(resp.read().decode())
        data = {
            "temp": round(raw["main"]["temp"], 1),
            "feels_like": round(raw["main"]["feels_like"], 1),
            "humidity": raw["main"]["humidity"],
            "wind_speed": round(raw["wind"]["speed"] * 3.6, 1),  # m/s → km/h
            "condition": raw["weather"][0]["description"],
            "icon": raw["weather"][0]["icon"],
            "city": raw.get("name", OWM_CITY),
        }
        _WEATHER_CACHE["ts"] = now
        _WEATHER_CACHE["data"] = data
        return data
    except Exception as e:
        print(f"[weather] OpenWeather fetch failed: {e}")
        return None


def _weather_speech(data: dict) -> str:
    """Convert weather data to a natural spoken sentence."""
    temp = data["temp"]
    cond = data["condition"].replace("_", " ")
    wind = data["wind_speed"]
    hum = data["humidity"]
    feels = data["feels_like"]

    temp_str = f"{round(temp)} degrees"
    parts = [f"It's currently {temp_str} and {cond.lower()} in {OWM_CITY}."]
    if abs(feels - temp) >= 2:
        parts.append(f"It feels like {round(feels)} degrees.")
    parts.append(f"Humidity is {hum} percent.")
    if wind > 15:
        parts.append(f"Wind is at {round(wind)} kilometers per hour.")
    elif wind > 5:
        parts.append(f"Light breeze at {round(wind)} kilometers per hour.")
    return " ".join(parts)


def _weather_log(data: dict) -> str:
    """Convert weather data to a log-line string."""
    return (f"\u2600 {OWM_CITY}: {data['temp']}\u00b0C, {data['condition']}, "
            f"humidity {data['humidity']}%, wind {data['wind_speed']} km/h")


class SmartHome:
    def __init__(self):
        self.lock = threading.Lock()
        self.lights_on = False
        self.light_brightness = 100          # percent
        self.temperature = 22.0             # degrees C (current)
        self.thermostat_setpoint = 22.0     # target
        self.media_playing = False
        self.media_track = "Ambient Lab — Gaia"
        self.media_idx = 0
        self.volume = 50
        self.timers: dict[str, dict] = {}
        self.alarms: list[dict] = []
        self.reminders: list[dict] = []
        self.last_call = None
        self._last_temp_was_explicit = False
        self.activity: list[dict] = []
        self.started = time.time()
        self._tick()

    # ---- helpers ---------------------------------------------------------
    def _now(self):
        return _dt.datetime.now()

    def _log(self, msg):
        """Append an activity entry. Caller MUST already hold self.lock —
        _tick/execute call this while holding it, and threading.Lock is not
        reentrant, so re-acquiring here deadlocks the whole worker."""
        entry = {"t": self._now().strftime("%H:%M:%S"), "msg": msg}
        self.activity.append(entry)
        self.activity = self.activity[-40:]

    def _tick(self):
        # expire finished timers/alarms
        now = time.time()
        with self.lock:
            for tid in list(self.timers):
                tm = self.timers[tid]
                if tm["active"] and now >= tm["ends_at"]:
                    tm["active"] = False
                    self._log(f"⏱ Timer '{tid}' finished.")
            self.alarms = [a for a in self.alarms
                           if not (a["triggered"])]

    def snapshot(self):
        self._tick()
        with self.lock:
            return {
                "lights_on": self.lights_on,
                "brightness": self.light_brightness,
                "temperature": self.temperature,
                "setpoint": self.thermostat_setpoint,
                "_temp_confirmed": self._last_temp_was_explicit,
                "media_playing": self.media_playing,
                "media_track": self.media_track,
                "volume": self.volume,
                "timers": {k: {**v, "remaining": max(0, int(v["ends_at"] - time.time()))}
                           for k, v in self.timers.items()},
                "alarms": self.alarms,
                "reminders": self.reminders,
                "last_call": self.last_call,
                "activity": list(reversed(self.activity)),
                "clock": self._now().strftime("%H:%M:%S"),
                "uptime_s": int(time.time() - self.started),
            }

    # ---- command dispatch ------------------------------------------------
    def reset(self):
        """Restore pristine state — used by the 3D Lab on every page load."""
        with self.lock:
            self.lights_on = False
            self.light_brightness = 100
            self.temperature = 22.0
            self.thermostat_setpoint = 22.0
            self.media_playing = False
            self.media_track = "Ambient Lab — Gaia"
            self.media_idx = 0
            self.volume = 50
            self.timers.clear()
            self.alarms.clear()
            self.reminders.clear()
            self.last_call = None
            self._last_temp_was_explicit = False
            self._log("🔄 Lab reset to pristine state.")

    def execute(self, intent: str, arg: str | None = None):
        """Apply an intent to the house. `arg` carries any parsed value."""
        a = (arg or "").strip()
        if intent == "light_on":
            if not self.lights_on:
                self.lights_on = True
                if self.light_brightness < 10:
                    self.light_brightness = 100
                self._log("💡 Lights turned ON.")
        elif intent == "light_off":
            if self.lights_on:
                self.lights_on = False
                self._log("🌑 Lights turned OFF.")
        elif intent == "dim_lights":
            pct = self._parse_percent(a)
            if not self.lights_on:
                self.lights_on = True
            if self.light_brightness != pct:
                self.light_brightness = pct
                self._log(f"🔅 Lights dimmed to {pct}%.")
        elif intent == "set_temperature":
            deg = self._parse_degrees(a)
            self._last_temp_was_explicit = deg is not None
            if deg is None:
                # No number in the utterance — do NOT default to 22°.
                self._log("\U0001f321\ufe0f Which temperature? Try “set temperature to 24”.")
            else:
                self.thermostat_setpoint = deg
                self._log(f"🌡 Thermostat set to {deg}°C.")
        elif intent == "set_timer":
            mins = self._parse_minutes(a)
            tid = f"timer_{len(self.timers)+1}"
            self.timers[tid] = {"label": a or f"{mins} min",
                                "ends_at": time.time() + mins * 60,
                                "active": True}
            self._log(f"⏱ Timer set for {mins} minute(s).")
        elif intent == "set_alarm":
            hhmm = self._parse_time(a)
            self.alarms.append({"time": hhmm, "label": a or hhmm,
                                "triggered": False})
            self._log(f"⏰ Alarm set for {hhmm}.")
        elif intent == "manage_reminders":
            text = a or "check email"
            self.reminders.append({"text": text,
                                   "added": self._now().strftime("%H:%M")})
            self._log(f"📝 Reminder added: {text}")
        elif intent == "play_music":
            self.media_playing = True
            self.media_idx = (self.media_idx + 1) % len(self.TRACKS)
            self.media_track = self.TRACKS[self.media_idx]
            self._log("▶ Music started: " + self.media_track)
        elif intent == "media_control":
            self.media_playing = not self.media_playing
            self._log("⏯ Media " + ("resumed." if self.media_playing else "paused."))
        elif intent == "make_call":
            contact = self._extract_contact(a)
            self.last_call = {"contact": contact,
                              "at": self._now().strftime("%H:%M:%S")}
            self._log(f"📞 Calling {contact}…")
        elif intent == "get_time":
            self._log("🕐 It is " + self._now().strftime("%I:%M %p") + ".")
        elif intent == "get_weather":
            wdata = _fetch_weather()
            if wdata:
                self._log(_weather_log(wdata))
            else:
                self._log("☀ Weather: 24°C, partly cloudy, light breeze.")
        else:
            self._log(f"❓ Unknown intent: {intent}")
        return self.snapshot()

    # ---- spoken confirmation (sent to the browser for TTS) -------------
    TRACKS = [
        "Ambient Lab \u2014 Gaia",
        "Deep Focus \u2014 Slow Pulse",
        "Morning Light \u2014 Soft Keys",
        "Night Cycle \u2014 Low Drift",
    ]

    def speak_for(self, device_intent, raw_intent=""):
        """Return a short sentence describing what just happened (or None)."""
        if not device_intent:
            return None
        try:
            if device_intent == "light_on":
                return "Lights on."
            if device_intent == "light_off":
                return "Lights off."
            if device_intent == "dim_lights":
                return f"Lights set to {self.light_brightness} percent."
            if device_intent == "set_temperature":
                # Only confirm when a number was actually heard — never invent one.
                if self._last_temp_was_explicit:
                    return f"Temperature set to {self.thermostat_setpoint:.0f} degrees."
                return "Which temperature? Please say a number, like “set temperature to 24”."
            if device_intent == "play_music":
                return f"Playing {self.media_track}."
            if device_intent == "media_control":
                if self.media_playing:
                    return "Resumed."
                return "Paused."
            if device_intent == "get_time":
                return "It is " + self._now().strftime("%I:%M %p") + "."
            if device_intent == "get_weather":
                wdata = _fetch_weather()
                if wdata:
                    return _weather_speech(wdata)
                return "Twenty four degrees, partly cloudy, light breeze."
            if device_intent == "make_call":
                return "Calling " + (self.last_call or {}).get("contact", "Mom") + "."
            if device_intent == "set_timer":
                return "Timer set."
            if device_intent == "set_alarm":
                return "Alarm set."
            if device_intent == "manage_reminders":
                return "Reminder updated."
        except Exception:
            return None
        return None

    # ---- parsers ---------------------------------------------------------
    @staticmethod
    def _parse_percent(a):
        m = re.search(r"(\d{1,3})", a)
        return max(5, min(100, int(m.group(1)))) if m else 50

    @staticmethod
    def _parse_degrees(a):
        """Parse a degree value from the utterance. Returns None when no
        number is present — callers must NOT invent a default (e.g. 22°)."""
        m = re.search(r"(\d{1,2}(?:\.\d)?)", a)
        return max(10, min(30, float(m.group(1)))) if m else None

    @staticmethod
    def _parse_minutes(a):
        m = re.search(r"(\d+)", a)
        return max(1, int(m.group(1))) if m else 5

    @staticmethod
    def _extract_contact(text: str | None) -> str:
        """Pull the contact name out of a command like 'call mom' or 'phone dad'."""
        if not text:
            return "Mom"
        t = text.lower().strip()
        # Strip leading wake word + verbs
        t = re.sub(r"^(hey\s+gaia|heigh\s+ayah|hey\s+aiya|hi\s+gaia)[\s,.!]*", "", t)
        t = re.sub(r"^(please\s+)?(can\s+you\s+|could\s+you\s+)?(call|dial|ring|phone|put\s+me\s+through\s+to\s+|connect\s+me\s+to\s+|reach\s+)", "", t)
        t = t.strip(" ,.!")
        if not t:
            return "Mom"
        # Capitalize first letter of each word
        return " ".join(w.capitalize() for w in t.split())

    @staticmethod
    def _parse_time(a):
        m = re.search(r"(\d{1,2})[:h](\d{2})", a)
        if m:
            return f"{int(m.group(1)):02d}:{m.group(2)}"
        m = re.search(r"(\d{1,2})\s*(am|pm)", a, re.I)
        if m:
            h = int(m.group(1)) % 12
            if m.group(2).lower() == "pm":
                h += 12
            return f"{h:02d}:00"
        return "07:00"


HOME = SmartHome()


# ---- text -> intent (fallback when mic is unavailable) -------------------
_RULES = [
    (r"\b(dim|lower|reduce|fade|soften|brightness)\b.*\b(light|lights)\b", "dim_lights"),
    (r"\b(dim|lower|reduce|fade|soften)\b", "dim_lights"),
    (r"\b(turn on|switch on|lights? on|light it up|brighten|wake up the lights|activate the lights|make it bright)\b", "light_on"),
    (r"\b(turn off|switch off|lights? off|lights? out|kill the lights|deactivate the lights|make it dark|darken)\b", "light_off"),
    (r"\b(temperature|thermostat|\bac\b|cool it|heat it|degrees?)\b", "set_temperature"),
    (r"\btimer\b|\bcount ?down\b|\btime me for\b", "set_timer"),
    (r"\balarm\b|\bwake me up\b|\bring me at\b|\bbuzz me\b", "set_alarm"),
    (r"\bremind\b|\breminder\b|\bto ?do list\b|\bcoming up\b", "manage_reminders"),
    (r"\b(call|dial|ring|phone|put me through)\b", "make_call"),
    (r"\b(volume|louder|quieter|mute|shut it up|next|previous|skip|pause|resume|hold on|fast forward|rewind|stop)\b", "media_control"),
    (r"\b(play|start|put on|music|song|playlist|tunes)\b", "play_music"),
    (r"\b(weather|forecast|rain|umbrella|outside)\b", "get_weather"),
    (r"\b(time|hour|how late)\b", "get_time"),
]


def parse_text(text: str) -> str:
    """Map a typed command to an intent (regex rules, most specific first)."""
    t = (text or "").lower().strip()
    for pat, intent in _RULES:
        if re.search(pat, t):
            return intent
    return "get_time"
