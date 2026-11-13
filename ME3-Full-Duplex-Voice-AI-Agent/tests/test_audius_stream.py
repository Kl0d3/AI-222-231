
import os
import subprocess
import requests
import numpy as np
import sounddevice as sd

from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

AUDIUS_API_KEY = os.getenv("AUDIUS_API_KEY")

BASE_URL = "https://api.audius.co/v1"

TRACK_ID = "gWd2p"

SAMPLE_RATE = 24000
CHANNELS = 2

MUSIC_VOLUME = 0.30


# ============================================================
# AUDIUS STREAM
# ============================================================

def get_track_stream(track_id):

    if not AUDIUS_API_KEY:
        raise RuntimeError("AUDIUS_API_KEY is missing from .env")

    url = f"{BASE_URL}/tracks/{track_id}/stream"

    params = {
        "api_key": AUDIUS_API_KEY
    }

    print("\nRequesting Audius stream...")

    response = requests.get(
        url,
        params=params,
        timeout=30,
        allow_redirects=True
    )

    response.raise_for_status()

    print("Audius stream retrieved successfully.")
    print(f"Final URL host: {response.url.split('/')[2]}")

    return response.content


# ============================================================
# MP3 DECODING
# ============================================================

def decode_audio(mp3_data):

    print("\nDecoding MP3 using FFmpeg...")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "pipe:1"
    ]

    result = subprocess.run(
        command,
        input=mp3_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32
    )

    audio = audio.reshape(-1, CHANNELS)

    print("MP3 decoded successfully.")
    print(f"Audio samples: {len(audio)}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Duration: {len(audio) / SAMPLE_RATE:.2f} seconds")

    return audio


# ============================================================
# MUSIC PLAYBACK
# ============================================================

def play_music(audio, volume=1.0):

    print("\nStarting music playback...")

    if not 0.0 <= volume <= 1.0:
        raise ValueError("Volume must be between 0.0 and 1.0")

    # Apply music-only volume control.
    audio = audio * volume

    # Prevent samples from exceeding the valid audio range.
    audio = np.clip(audio, -1.0, 1.0)

    sd.play(
        audio,
        samplerate=SAMPLE_RATE,
        blocking=True
    )

    print("Playback completed.")


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 55)
    print("AUDIUS MUSIC PLAYBACK TEST")
    print("=" * 55)

    try:

        mp3_data = get_track_stream(TRACK_ID)

        print(f"Downloaded audio: {len(mp3_data):,} bytes")

        audio = decode_audio(mp3_data)

        play_music(
            audio,
            volume=MUSIC_VOLUME
        )

        print("\nAudius playback test completed successfully.")

    except requests.RequestException as error:

        print(f"\nAudius API error: {error}")

    except subprocess.CalledProcessError as error:

        print("\nFFmpeg decoding failed.")
        print(error.stderr.decode(errors="replace"))

    except Exception as error:

        print(f"\nPlayback test failed: {error}")

    finally:

        sd.stop()


if __name__ == "__main__":
    main()
