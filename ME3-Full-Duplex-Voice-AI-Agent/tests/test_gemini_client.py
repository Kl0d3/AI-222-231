from src.llm.gemini_client import GeminiClient


def main():

    gemini = GeminiClient()

    try:
        prompt = "Explain what a Raspberry Pi is in one sentence."

        print("\nSending prompt to Gemini...")

        response = gemini.generate_response(prompt)

        print("\n========== GEMINI RESPONSE ==========")
        print(response)
        print("=====================================")

    except Exception as error:
        print(f"\nGemini request failed: {error}")

    finally:
        gemini.close()


if __name__ == "__main__":
    main()
