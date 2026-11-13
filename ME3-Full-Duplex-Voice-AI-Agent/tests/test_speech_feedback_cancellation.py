import time

from src.agent.speech_feedback import SpeechFeedbackManager


manager = SpeechFeedbackManager(
    minimum_feedback_interval=1.0,
    backchannel_cooldown=3.0
)

manager.begin_turn()
manager.complete_user_turn()

print("\nStarting processing...")

manager.begin_processing(
    "What is machine learning?"
)

# Retrieve immediate acknowledgment.
message = manager.get_next_feedback(timeout=0.2)

if message:
    print(f"Feedback: {message.text}")

# Simulate processing that finishes after 2 seconds.
time.sleep(2.0)

manager.finish_processing()

print("\nProcessing finished after 2 seconds.")

# Wait beyond the 3-second progress threshold.
time.sleep(3.5)

message = manager.get_next_feedback(timeout=0.2)

if message:
    print(f"UNEXPECTED FEEDBACK: {message.text}")
else:
    print("PASS: No outdated feedback was played.")

print("\nFinal status:")
print(manager.get_status())

manager.end_turn()