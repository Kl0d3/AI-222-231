
import heapq
import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class FeedbackMessage:
    text: str
    category: str
    priority: int
    turn_id: int
    timestamp: float


class SpeechFeedbackManager:

    PRIORITY = {
        "acknowledgment": 1,
        "progress": 2,
        "backchannel": 3,
    }

    BACKCHANNELS = [
        "Mm-hmm.",
        "Yeah.",
        "Aha.",
        "Right.",
    ]

    TOOL_MESSAGES = {
        "web_search": "Let me look that up.",
        "get_weather": "I'm checking the weather.",
        "get_news": "Let me check the latest news.",
        "get_thermostat_status": "I'm checking the thermostat.",
        "get_light_status": "I'm checking the lights.",
    }

    PROGRESS_SCHEDULE = [
        (3.0, "progress_stage_1"),
        (5.0, "progress_stage_2"),
    ]

    ACKNOWLEDGMENT_RULES = {
        "research": "Hmm, let me check.",
        "search": "Hmm, let me check.",
        "weather": "Sure, let me check.",
        "news": "Hmm, let me check.",
        "smart_home": "Sure, one moment.",
        "analysis": "Absolutely, let's take a look.",
        "general": "Yeah, sure!",
    }

    def __init__(
        self,
        minimum_feedback_interval=1.0,
        backchannel_cooldown=3.0
    ):

        self.feedback_queue = queue.PriorityQueue()
        self.lock = threading.RLock()

        self.minimum_feedback_interval = (
            minimum_feedback_interval
        )

        self.backchannel_cooldown = backchannel_cooldown

        self.turn_id = 0

        self.turn_active = False
        self.user_turn_complete = False

        self.processing_active = False
        self.processing_started_at = None

        self.active_tool = None

        self.last_feedback_time = 0.0
        self.last_backchannel_time = 0.0

        self.last_message = None
        self.feedback_counter = 0

        self.initial_acknowledgment_sent = False
        self.processing_feedback_sent = False

        # Heap entries:
        # (due_time, priority, counter, item_type, payload)
        #
        # item_type:
        #   progress_stage_1
        #   progress_stage_2
        #   deferred_message
        self.scheduled_feedback = []

        print("[FEEDBACK] Speech Feedback Manager initialized.")

    # ====================================================
    # Acknowledgment selection
    # ====================================================

    def select_acknowledgment(self, user_text=None):

        if not user_text:
            return "Yeah, sure!"

        text = str(user_text).lower()

        research_keywords = [
            "latest",
            "current",
            "recent",
            "search",
            "look up",
            "find information",
            "research",
            "sources",
            "news",
            "weather",
        ]

        smart_home_keywords = [
            "turn on",
            "turn off",
            "switch on",
            "switch off",
            "thermostat",
            "temperature",
            "lights",
            "light",
        ]

        analysis_keywords = [
            "analyze",
            "analyse",
            "evaluate",
            "compare",
            "calculate",
            "data",
            "dataset",
            "report",
        ]

        if any(word in text for word in smart_home_keywords):
            return self.ACKNOWLEDGMENT_RULES["smart_home"]

        if any(word in text for word in analysis_keywords):
            return self.ACKNOWLEDGMENT_RULES["analysis"]

        if any(word in text for word in research_keywords):
            return self.ACKNOWLEDGMENT_RULES["research"]

        return self.ACKNOWLEDGMENT_RULES["general"]

    # ====================================================
    # Internal queue management
    # ====================================================

    def _create_message(self, text, category):

        if not text or not self.turn_active:
            return None

        if (
            category == "backchannel"
            and self.user_turn_complete
        ):
            return None

        priority = self.PRIORITY.get(category, 2)

        self.feedback_counter += 1

        return FeedbackMessage(
            text=text,
            category=category,
            priority=priority,
            turn_id=self.turn_id,
            timestamp=time.monotonic()
        )

    def _enqueue(self, text, category):

        with self.lock:

            if text == self.last_message:
                return False

            message = self._create_message(
                text,
                category
            )

            if message is None:
                return False

            self.feedback_queue.put(
                (
                    message.priority,
                    self.feedback_counter,
                    message
                )
            )

            self.last_message = text

            return True

    def _clear_queue(self):

        while True:

            try:
                self.feedback_queue.get_nowait()
                self.feedback_queue.task_done()

            except queue.Empty:
                break

    def _clear_scheduled_feedback(self):
        self.scheduled_feedback.clear()

    def _remove_pending_backchannels(self):

        retained_messages = []

        while True:

            try:
                item = self.feedback_queue.get_nowait()

            except queue.Empty:
                break

            self.feedback_queue.task_done()

            _, _, message = item

            if message.category != "backchannel":
                retained_messages.append(item)

        for item in retained_messages:
            self.feedback_queue.put(item)

        # Also remove deferred backchannels.
        retained_schedules = [
            item
            for item in self.scheduled_feedback
            if not (
                item[3] == "deferred_message"
                and item[4].category == "backchannel"
            )
        ]

        self.scheduled_feedback = retained_schedules
        heapq.heapify(self.scheduled_feedback)

    def _schedule_progress_feedback(self):

        now = time.monotonic()

        for delay, category in self.PROGRESS_SCHEDULE:

            self.feedback_counter += 1

            heapq.heappush(
                self.scheduled_feedback,
                (
                    now + delay,
                    self.PRIORITY["progress"],
                    self.feedback_counter,
                    category,
                    None
                )
            )

    # ====================================================
    # Scheduled feedback processing
    # ====================================================

    def _release_due_feedback(self):

        now = time.monotonic()

        while self.scheduled_feedback:

            due_time, priority, counter, item_type, payload = (
                self.scheduled_feedback[0]
            )

            if due_time > now:
                break

            heapq.heappop(self.scheduled_feedback)

            if not self.turn_active:
                continue

            if item_type == "deferred_message":

                message = payload

                if message.turn_id != self.turn_id:
                    continue

                if (
                    message.category == "backchannel"
                    and self.user_turn_complete
                ):
                    continue

                if (
                    message.category == "progress"
                    and not self.processing_active
                ):
                    continue

                self.feedback_queue.put(
                    (
                        message.priority,
                        counter,
                        message
                    )
                )

                continue

            if not self.processing_active:
                continue

            if item_type == "progress_stage_1":

                if self.active_tool:
                    message_text = self.TOOL_MESSAGES.get(
                        self.active_tool,
                        "Let me check that."
                    )
                else:
                    message_text = "Hmm..."

            elif item_type == "progress_stage_2":

                if self.active_tool:
                    message_text = "I'm still checking that."
                else:
                    message_text = "Ahh, hmm..."

            else:
                continue

            self._enqueue(
                message_text,
                "progress"
            )

    def _schedule_deferred_message(
        self,
        message,
        delay
    ):

        self.feedback_counter += 1

        heapq.heappush(
            self.scheduled_feedback,
            (
                time.monotonic() + delay,
                message.priority,
                self.feedback_counter,
                "deferred_message",
                message
            )
        )

    # ====================================================
    # Conversation lifecycle
    # ====================================================

    def begin_turn(self):

        with self.lock:

            self.turn_id += 1

            self.turn_active = True
            self.user_turn_complete = False

            self.processing_active = False
            self.processing_started_at = None
            self.active_tool = None

            self.initial_acknowledgment_sent = False
            self.processing_feedback_sent = False

            self.last_message = None
            self.last_feedback_time = 0.0
            self.last_backchannel_time = 0.0

            self._clear_queue()
            self._clear_scheduled_feedback()

            print(
                f"[FEEDBACK] Turn {self.turn_id} started."
            )

            return self.turn_id

    def complete_user_turn(self):

        with self.lock:

            self.user_turn_complete = True

            self._remove_pending_backchannels()

            print("[FEEDBACK] User turn completed.")

    def end_turn(self):

        with self.lock:

            self.turn_active = False
            self.user_turn_complete = True

            self.processing_active = False
            self.processing_started_at = None
            self.active_tool = None

            self._clear_queue()
            self._clear_scheduled_feedback()

            print(
                f"[FEEDBACK] Turn {self.turn_id} ended."
            )

    def cancel_pending_feedback(self):

        with self.lock:

            self._clear_queue()
            self._clear_scheduled_feedback()

            self.processing_active = False
            self.processing_started_at = None
            self.active_tool = None

            print("[FEEDBACK] Pending feedback cancelled.")

    # ====================================================
    # Think-aloud processing
    # ====================================================

    def notify_processing_started(self, user_text=None):

        with self.lock:

            if not self.turn_active:
                return False

            if self.initial_acknowledgment_sent:
                return False

            self.processing_active = True
            self.processing_started_at = time.monotonic()
            self.active_tool = None

            acknowledgment = self.select_acknowledgment(
                user_text
            )

            accepted = self._enqueue(
                acknowledgment,
                "acknowledgment"
            )

            if accepted:

                self.initial_acknowledgment_sent = True

                self._schedule_progress_feedback()

                print(
                    "[FEEDBACK] Processing feedback scheduled."
                )

            return accepted

    def begin_processing(self, user_text=None):
        return self.notify_processing_started(user_text)

    def notify_tool_started(self, tool_name):

        with self.lock:

            if not self.turn_active or not self.processing_active:
                return False

            self.active_tool = str(tool_name).lower()

            print(
                f"[FEEDBACK] Tool processing started: "
                f"{self.active_tool}"
            )

            return True

    def notify_processing_slow(self):

        with self.lock:

            if not self.processing_active:
                return False

            return self._enqueue(
                "Just a moment.",
                "progress"
            )

    def finish_processing(self):

        with self.lock:

            self.processing_active = False
            self.processing_started_at = None
            self.active_tool = None

            self._clear_scheduled_feedback()

            retained_messages = []

            while True:

                try:
                    item = self.feedback_queue.get_nowait()

                except queue.Empty:
                    break

                self.feedback_queue.task_done()

                _, _, message = item

                if message.category != "progress":
                    retained_messages.append(item)

            for item in retained_messages:
                self.feedback_queue.put(item)

            print("[FEEDBACK] Processing completed.")

    # ====================================================
    # Conversational backchanneling
    # ====================================================

    def request_backchannel(
        self,
        user_turn_active,
        user_speech_active
    ):

        with self.lock:

            if not self.turn_active:
                return False

            if self.user_turn_complete:
                return False

            if not user_turn_active or user_speech_active:
                return False

            now = time.monotonic()

            if (
                now - self.last_backchannel_time
                < self.backchannel_cooldown
            ):
                return False

            if not self.BACKCHANNELS:
                return False

            message = self.BACKCHANNELS[
                self.feedback_counter % len(self.BACKCHANNELS)
            ]

            accepted = self._enqueue(
                message,
                "backchannel"
            )

            if accepted:
                self.last_backchannel_time = now

            return accepted

    # ====================================================
    # Feedback retrieval
    # ====================================================

    def get_next_feedback(self, timeout=0.1):

        deadline = (
            time.monotonic()
            + max(0.0, timeout)
        )

        while True:

            with self.lock:
                self._release_due_feedback()

            try:

                priority, counter, message = (
                    self.feedback_queue.get_nowait()
                )

            except queue.Empty:

                message = None

            if message is not None:

                with self.lock:

                    invalid_message = (
                        message.turn_id != self.turn_id
                        or not self.turn_active
                        or (
                            message.category == "backchannel"
                            and self.user_turn_complete
                        )
                        or (
                            message.category == "progress"
                            and not self.processing_active
                        )
                    )

                    if invalid_message:

                        self.feedback_queue.task_done()
                        continue

                    now = time.monotonic()

                    # Acknowledgments have immediate priority.
                    # Other feedback respects the minimum interval.
                    if (
                        message.category != "acknowledgment"
                        and self.last_feedback_time > 0
                        and now - self.last_feedback_time
                        < self.minimum_feedback_interval
                    ):

                        remaining = (
                            self.minimum_feedback_interval
                            - (now - self.last_feedback_time)
                        )

                        self._schedule_deferred_message(
                            message,
                            remaining
                        )

                        self.feedback_queue.task_done()

                        continue

                    self.last_feedback_time = now

                    self.feedback_queue.task_done()

                    if message.category == "progress":
                        self.processing_feedback_sent = True

                    return message

            remaining_timeout = (
                deadline - time.monotonic()
            )

            if remaining_timeout <= 0:
                return None

            with self.lock:

                if self.scheduled_feedback:

                    next_due = self.scheduled_feedback[0][0]

                    wait_time = min(
                        remaining_timeout,
                        max(
                            0.01,
                            next_due - time.monotonic()
                        )
                    )

                else:

                    wait_time = min(
                        remaining_timeout,
                        0.05
                    )

            time.sleep(wait_time)

    # ====================================================
    # Diagnostics
    # ====================================================

    def get_status(self):

        with self.lock:

            elapsed = None

            if self.processing_started_at is not None:

                elapsed = (
                    time.monotonic()
                    - self.processing_started_at
                )

            return {
                "turn_id": self.turn_id,
                "turn_active": self.turn_active,
                "user_turn_complete": self.user_turn_complete,
                "processing_active": self.processing_active,
                "processing_elapsed": elapsed,
                "active_tool": self.active_tool,
                "queue_size": self.feedback_queue.qsize(),
                "scheduled_feedback": len(
                    self.scheduled_feedback
                ),
                "initial_acknowledgment_sent":
                    self.initial_acknowledgment_sent,
                "processing_feedback_sent":
                    self.processing_feedback_sent,
            }
