import time

from src.agent.speech_feedback import SpeechFeedbackManager


manager = SpeechFeedbackManager(
    minimum_feedback_interval=1.0,
    backchannel_cooldown=3.0
)

# Start a conversation turn.
manager.begin_turn()

# Simulate a user asking a general question.
manager.complete_user_turn()

print("\nStarting processing...")

start = time.monotonic()

# Start processing and queue the acknowledgment.
manager.begin_processing(
    "What is machine learning?"
)

# Collect feedback for approximately 6 seconds.
while time.monotonic() - start < 6.0:

    message = manager.get_next_feedback(timeout=0.1)

    if message:
        elapsed = time.monotonic() - start

        print(
            f"[{elapsed:.2f}s] "
            f"{message.category}: {message.text}"
        )

# Simulate processing completion.
manager.finish_processing()

print("\nProcessing finished.")

print("\nFinal status:")
print(manager.get_status())

manager.end_turn()