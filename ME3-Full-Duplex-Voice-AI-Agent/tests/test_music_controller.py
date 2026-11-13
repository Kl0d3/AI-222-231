
import time

from src.mcp_servers.music.music_controller import MusicController


def main():

    print("=" * 60)
    print("MUSIC CONTROLLER TEST")
    print("=" * 60)

    music = MusicController(
        initial_volume=30,
        duck_factor=0.20
    )

    try:

        # ----------------------------------------------------
        # 1. Search and play
        # ----------------------------------------------------

        print("\n1. Searching for and playing lofi music...")

        status = music.search_and_play(
            query="lofi",
            limit=5
        )

        print("Playback started.")
        print(status)

        time.sleep(2)

        # ----------------------------------------------------
        # 2. Pause
        # ----------------------------------------------------

        print("\n2. Testing pause...")

        print(music.pause())

        time.sleep(1)

        # ----------------------------------------------------
        # 3. Resume
        # ----------------------------------------------------

        print("\n3. Testing resume...")

        print(music.resume())

        time.sleep(2)

        # ----------------------------------------------------
        # 4. Test assistant speech ducking
        # ----------------------------------------------------

        print("\n4. Simulating Gemini speech started...")

        print(music.assistant_speech_started())

        time.sleep(2)

        print("\n5. Simulating Gemini speech finished...")

        print(music.assistant_speech_finished())

        time.sleep(1)

        # ----------------------------------------------------
        # 5. Test volume controls
        # ----------------------------------------------------

        print("\n6. Setting music volume to 50%...")

        print(music.set_volume(50))

        time.sleep(1)

        print("\n7. Increasing music volume by 10%...")

        print(music.adjust_volume(10))

        time.sleep(1)

        # ----------------------------------------------------
        # 6. Test mute and unmute
        # ----------------------------------------------------

        print("\n8. Testing mute...")

        print(music.mute())

        time.sleep(1)

        print("\n9. Testing unmute...")

        print(music.unmute())

        time.sleep(2)

        # ----------------------------------------------------
        # 7. Retrieve current track
        # ----------------------------------------------------

        print("\n10. Retrieving current track...")

        print(music.get_current_track())

        # ----------------------------------------------------
        # 8. Retrieve complete status
        # ----------------------------------------------------

        print("\n11. Retrieving complete music status...")

        print(music.get_status())

        # ----------------------------------------------------
        # 9. Stop playback
        # ----------------------------------------------------

        print("\n12. Stopping music...")

        print(music.stop())

        print("\nMusic controller test completed.")

    except Exception as error:

        print("\nMusic controller test failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")

    finally:

        music.close()


if __name__ == "__main__":
    main()
