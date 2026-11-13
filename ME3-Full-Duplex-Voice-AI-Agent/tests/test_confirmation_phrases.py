
from src.agent.coordinator import ConversationCoordinator


def main():

    coordinator = ConversationCoordinator.__new__(
        ConversationCoordinator
    )

    test_cases = [
        # Affirmative confirmations
        ("Yes", "yes"),
        ("Yes, please.", "yes"),
        ("Yes, go ahead!", "yes"),
        ("Please proceed.", "yes"),

        # Negative confirmations
        ("No", "no"),
        ("No, thanks.", "no"),
        ("No, thank you.", "no"),
        ("Cancel it.", "no"),
        ("Don't do that.", "no"),
        ("Do not do that.", "no"),
        ("I changed my mind.", "no"),

        # Combined negative confirmations
        ("No, cancel it.", "no"),
        ("No, please cancel.", "no"),
        ("No, don't do that.", "no"),
        ("No, do not do that.", "no"),
        ("No, never mind.", "no"),

        # Ambiguous or unrelated responses
        ("Yes, but wait.", None),
        ("Yes, but change the date.", None),
        ("No, actually yes.", None),
        ("Maybe.", None),
        ("I need to check first.", None),
        ("Okay.", None),
    ]

    print("=" * 60)
    print("CONFIRMATION PHRASE CLASSIFICATION TESTS")
    print("=" * 60)

    passed = 0

    for transcript, expected in test_cases:

        actual = coordinator.classify_confirmation(
            transcript
        )

        if actual == expected:

            print(
                f"PASS: {transcript!r} -> {actual!r}"
            )

            passed += 1

        else:

            print(
                f"FAIL: {transcript!r} "
                f"Expected={expected!r}, Actual={actual!r}"
            )

    print("\n" + "=" * 60)
    print(
        f"RESULT: {passed}/{len(test_cases)} tests passed"
    )
    print("=" * 60)

    if passed != len(test_cases):

        raise AssertionError(
            "Some confirmation phrase tests failed."
        )

    print("ALL CONFIRMATION PHRASE TESTS PASSED")


if __name__ == "__main__":
    main()
