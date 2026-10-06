"""Audio feature extraction for the Voice Command Model (VCM).

Pure numpy/scipy so it runs identically on the training server and on a
Raspberry Pi 5 (no torch needed to compute a feature). The model consumes a
fixed-size log-mel spectrogram: (1, N_MELS, N_FRAMES).
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 400          # 25 ms @ 16 kHz
HOP = 160            # 10 ms @ 16 kHz
N_FRAMES = 99        # target frame count (~990 ms window)


def load_wav(path: str) -> np.ndarray:
    """Load any wav file as float32 mono @ 16 kHz."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)                      # mono
    if sr != SAMPLE_RATE:
        x = resample(x, sr, SAMPLE_RATE)
    return x


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    from scipy.signal import resample_poly
    import math
    g = math.gcd(src_sr, dst_sr)
    return resample_poly(x, dst_sr // g, src_sr // g).astype(np.float32)


def _mel_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filterbank (no external deps)."""
    def hz_to_mel(h):
        return 2595.0 * np.log10(1.0 + h / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low, high = 0.0, sr / 2.0
    mel_low, mel_high = hz_to_mel(low), hz_to_mel(high)
    pts = mel_to_hz(np.linspace(mel_low, mel_high, n_filters + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft)
    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_filters):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_MEL_FB = None


def get_mel_filterbank() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE)
    return _MEL_FB


def mel_spectrogram(x: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram of shape (N_MELS, T)."""
    if x.size == 0:
        x = np.zeros(SAMPLE_RATE, dtype=np.float32)
    win = np.hanning(N_FFT).astype(np.float32)
    n_frames = 1 + (len(x) - N_FFT) // HOP
    if n_frames <= 0:
        x = np.pad(x, (0, N_FFT))
        n_frames = 1
    shape = (n_frames, N_FFT)
    strides = (x.strides[0] * HOP, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    frames = frames * win
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2) + 1e-10
    mel = get_mel_filterbank() @ power.T
    return np.log(mel + 1e-5).astype(np.float32)


def _find_onset(x: np.ndarray, sr: int = SAMPLE_RATE) -> int:
    """Return sample index where speech onset occurs.

    Uses RMS energy in 10 ms hops with an adaptive threshold:
    threshold = max(absolute_floor, relative_fraction * peak_energy).
    
    KEY FIXES:
    1. Requires sustained energy (>= 50 ms / 5 consecutive hops) before
       accepting an onset. Filters out keyboard/mouse clicks.
    2. Detects and skips initial transients (clicks) that are significantly
       louder than the subsequent speech. If the first 100ms is >2x louder
       than the rest, we assume it's a click and start looking after it.
    
    Returns 0 if no clear sustained onset is found.
    """
    hop = 160  # 10 ms
    if len(x) < hop:
        return 0
    n_hops = 1 + (len(x) - hop) // hop
    # Compute per-hop RMS efficiently
    shape = (n_hops, hop)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = rms.max()
    if peak < 1e-6:
        return 0
    
    # NEW: Detect and skip initial transient (click)
    # If first 100ms is significantly louder than the rest, assume it's a click
    first_100ms = rms[:10]  # first 10 hops = 100ms
    rest = rms[10:]
    if len(rest) > 0:
        first_rms = np.mean(first_100ms)
        rest_rms = np.mean(rest)
        if rest_rms > 1e-6 and first_rms > 2.0 * rest_rms:
            # Likely a click at the start - skip the first 100ms
            start_search = 10
        else:
            start_search = 0
    else:
        start_search = 0
    
    # Adaptive threshold: 12% of peak, but at least an absolute floor
    threshold = max(peak * 0.12, 0.003)
    above = rms >= threshold  # boolean array, same length as rms
    
    # Find the first SUSTAINED region: >= 5 consecutive hops above threshold
    # Start searching from start_search to skip initial clicks
    min_sustained = 5  # 50 ms of continuous energy = speech, not a click
    onset_hop = -1
    run = 0
    for i in range(start_search, len(above)):
        if above[i]:
            run += 1
            if run >= min_sustained:
                onset_hop = i - min_sustained + 1  # start of the run
                break
        else:
            run = 0
    
    if onset_hop < 0:
        # No sustained region found - fall back to first above-threshold hop
        first_above = np.where(above[start_search:])[0]
        if len(first_above) == 0:
            return 0
        onset_hop = start_search + first_above[0]
    
    # Go back one hop to catch the attack transient
    onset_sample = max(0, (onset_hop - 1) * hop)
    return onset_sample


def preprocess(x: np.ndarray) -> np.ndarray:
    """Full pipeline -> (1, N_MELS, N_FRAMES) float32, mean/std normalized.

    Uses onset-aligned cropping: detects where speech begins and crops
    from there, so the model always sees the command rather than the
    middle of a long recording. Falls back to center-crop if no onset
    is detected (pure silence / noise).
    """
    # --- Onset detection on raw waveform (before spectrogram) ---
    onset = 0
    if len(x) > N_FRAMES * HOP:  # only worth detecting if longer than target
        onset = _find_onset(x)
        # Trim leading silence (keep 40 ms pre-roll for natural attack)
        pre_roll = min(40 * (SAMPLE_RATE // 1000), onset)  # 40 ms
        x = x[max(0, onset - pre_roll):]

    mel = mel_spectrogram(x)                       # (N_MELS, T)
    t = mel.shape[1]
    if t > N_FRAMES:
        # Onset-aligned crop: start from beginning of (trimmed) audio.
        # If we already trimmed to onset, start=0 is correct.
        # If onset wasn't found (fallback), use center-crop.
        start = 0 if onset > 0 else (t - N_FRAMES) // 2
        mel = mel[:, start:start + N_FRAMES]
    elif t < N_FRAMES:                              # zero-pad on the right
        mel = np.pad(mel, ((0, 0), (0, N_FRAMES - t)), mode="constant")
    img = mel.T.astype(np.float32)                  # (N_FRAMES, N_MELS)
    mu = img.mean()
    sd = img.std() + 1e-6
    img = (img - mu) / sd
    # (1, 1, N_FRAMES, N_MELS) = (N, C, H, W) for Conv2d
    return img[None, None, :, :]


def augment(x: np.ndarray, rng: np.random.Generator,
            noise: np.ndarray | None = None) -> np.ndarray:
    """Conservative augmentation: small time-shift + mild gain.

    Kept deliberately light so the model memorizes neither shift nor noise;
    heavy augmentation caused severe overfitting on this small dataset.
    """
    n = len(x)
    shift = int(rng.integers(-int(0.02 * SAMPLE_RATE), int(0.02 * SAMPLE_RATE)))
    if shift > 0:
        x = np.concatenate([np.zeros(shift, np.float32), x[:-shift]])
    elif shift < 0:
        x = np.concatenate([x[-shift:], np.zeros(-shift, np.float32)])
    x = x * float(10 ** (rng.uniform(-0.25, 0.25) / 20.0))
    return x


def spec_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time/frequency masking on the (N_FRAMES, N_MELS) log-mel image.

    Applied at TRAIN time only (after preprocess), so the model learns to
    ignore brief spectral gaps -- a standard, well-proven regularizer for
    small speech datasets. Mask widths are kept modest so we never erase the
    whole command.
    """
    img = img.copy()
    h, w = img.shape  # (N_FRAMES, N_MELS) = (99, 40)
    # time masks: 2 masks, up to 3 frames wide
    for _ in range(2):
        t0 = int(rng.integers(0, h))
        tw = int(rng.integers(1, 4))
        img[t0:min(h, t0 + tw), :] = 0.0
    # freq masks: 1 mask, up to 24 mels wide
    f0 = int(rng.integers(0, w))
    fw = int(rng.integers(1, 25))
    img[:, f0:min(w, f0 + fw)] = 0.0
    return img


def spec_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time/frequency masking on the (N_FRAMES, N_MELS) log-mel image.

    Applied at TRAIN time only (after preprocess), so the model learns to
    ignore brief spectral gaps -- a standard, well-proven regularizer for
    small speech datasets. Mask widths are kept modest so we never erase the
    whole command.
    """
    img = img.copy()
    h, w = img.shape  # (N_FRAMES, N_MELS) = (99, 40)
    # time masks: 2 masks, up to 3 frames wide
    for _ in range(2):
        t0 = int(rng.integers(0, h))
        tw = int(rng.integers(1, 4))
        img[t0:min(h, t0 + tw), :] = 0.0
    # freq masks: 1 mask, up to 24 mels wide
    f0 = int(rng.integers(0, w))
    fw = int(rng.integers(1, 25))
    img[:, f0:min(w, f0 + fw)] = 0.0
    return img


def spec_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time/frequency masking on the (N_FRAMES, N_MELS) log-mel image.

    Applied at TRAIN time only (after preprocess), so the model learns to
    ignore brief spectral gaps -- a standard, well-proven regularizer for
    small speech datasets. Mask widths are kept modest so we never erase the
    whole command.
    """
    img = img.copy()
    h, w = img.shape  # (N_FRAMES, N_MELS) = (99, 40)
    # time masks: 2 masks, up to 3 frames wide
    for _ in range(2):
        t0 = int(rng.integers(0, h))
        tw = int(rng.integers(1, 4))
        img[t0:min(h, t0 + tw), :] = 0.0
    # freq masks: 1 mask, up to 24 mels wide
    f0 = int(rng.integers(0, w))
    fw = int(rng.integers(1, 25))
    img[:, f0:min(w, f0 + fw)] = 0.0
    return img
