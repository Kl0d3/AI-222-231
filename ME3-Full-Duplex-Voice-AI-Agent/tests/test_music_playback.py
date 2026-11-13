
import time

from src.mcp_servers.music.audius_client import (
    get_track,
    get_track_stream
)

from src.mcp_servers.music.audio_decoder import AudioDecoder

from src.mcp_servers.music.playback_engine import PlaybackEngine


# ============================================================
# CONFIGURATION
# ============================================================

TRACK_ID = "gWd2p"

SAMPLE_RATE = 24000
CHANNELS = 2

INITIAL_VOLUME = 0.30
DUCK_FACTOR = 0.20

TEST_PLAYBACK_SECONDS = 8


# ============================================================
# MAIN INTEGRATION TEST
# ============================================================

def main():

    print("=" * 60)
    print("AUDIUS MUSIC PLAYBACK INTEGRATION TEST")
    print("=" * 60)

    playback_engine = PlaybackEngine(
        fade_duration=0.25
    )

    try:

        # ----------------------------------------------------
        # 1. Retrieve track metadata
        # ----------------------------------------------------

        print("\n1. Retrieving track information...")

        track = get_track(TRACK_ID)

        print(f"Title: {track['title']}")
        print(f"Artist: {track['artist']}")
        print(f"Duration: {track['duration']} seconds")
        print(f"Streamable: {track['is_streamable']}")

        if not track["is_streamable"]:
            raise RuntimeError(
                "Selected Audius track is not streamable."
            )

        # ----------------------------------------------------
        # 2. Retrieve MP3
        # ----------------------------------------------------

        print("\n2. Retrieving Audius audio stream...")

        response = get_track_stream(TRACK_ID)

        try:
            mp3_data = response.content
        finally:
            response.close()

        print("MP3 retrieved successfully.")
        print(f"Downloaded bytes: {len(mp3_data):,}")

        # ----------------------------------------------------
        # 3. Decode MP3
        # ----------------------------------------------------

        print("\n3. Decoding MP3...")

        decoder = AudioDecoder(
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS
        )

        audio, sample_rate = decoder.decode_mp3(
            mp3_data
        )

        duration = len(audio) / sample_rate

        print("Decoding successful.")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Channels: {audio.shape[1]}")
        print(f"Duration: {duration:.2f} seconds")

        # ----------------------------------------------------
        # 4. Start playback
        # ----------------------------------------------------

        print("\n4. Starting music playback...")

        playback_engine.set_volume(INITIAL_VOLUME)

        playback_engine.start(
            audio=audio,
            sample_rate=sample_rate
        )

        time.sleep(2)

        print("Playback status:")
        print(playback_engine.get_status())

        # ----------------------------------------------------
        # 5. Test pause
        # ----------------------------------------------------

        print("\n5. Testing pause...")

        playback_engine.pause()

        time.sleep(1)

        print("Playback status:")
        print(playback_engine.get_status())

        # ----------------------------------------------------
        # 6. Test resume
        # ----------------------------------------------------

        print("\n6. Testing resume...")

        playback_engine.resume()

        time.sleep(2)

        print("Playback status:")
        print(playback_engine.get_status())

        # ----------------------------------------------------
        # 7. Test ducking
        # ----------------------------------------------------

        print("\n7. Testing music ducking...")

        playback_engine.duck(
            factor=DUCK_FACTOR
        )

        time.sleep(1)

        print("Playback status:")
        print(playback_engine.get_status())

        # ----------------------------------------------------
        # 8. Restore volume
        # ----------------------------------------------------

        print("\n8. Restoring music volume...")

        playback_engine.unduck()

        time.sleep(1)

        print("Playback status:")
        print(playback_engine.get_status())

        # ----------------------------------------------------
        # 9. Continue playback
        # ----------------------------------------------------

        print(
            f"\n9. Continuing playback for "
            f"{TEST_PLAYBACK_SECONDS} seconds..."
        )

        time.sleep(TEST_PLAYBACK_SECONDS)

        print("Playback status:")
        print(playback_engine.get_status())

        # ----------------------------------------------------
        # 10. Stop playback
        # ----------------------------------------------------

        print("\n10. Stopping music...")

        playback_engine.stop()

        print("Playback status:")
        print(playback_engine.get_status())

        print("\nIntegration test completed successfully.")

    except Exception as error:

        print("\nIntegration test failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")

    finally:

        playback_engine.stop()


if __name__ == "__main__":
    main()
