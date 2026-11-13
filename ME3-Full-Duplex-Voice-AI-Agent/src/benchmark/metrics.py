
import json
import threading
import time
from pathlib import Path


class BenchmarkMetrics:

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "stage_metrics.jsonl"

        self.lock = threading.Lock()
        self.events = {}

    def record(self, event_name, **metadata):
        """Record an event using the current monotonic clock."""

        return self.record_at(
            event_name,
            time.monotonic(),
            **metadata
        )

    def record_at(self, event_name, monotonic_s, **metadata):
        """
        Record an event using an existing monotonic timestamp.

        Useful when an event was timestamped at its source
        but consumed by another worker later.
        """

        timestamp = float(monotonic_s)

        event = {
            "event": event_name,
            "monotonic_s": timestamp,
            "metadata": metadata
        }

        with self.lock:
            self.events.setdefault(event_name, []).append(timestamp)

            with self.log_file.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(event, ensure_ascii=False) + "\n"
                )

        return timestamp

    def elapsed(self, start_event, end_event):
        """Calculate elapsed time using the latest matching events."""

        with self.lock:
            starts = self.events.get(start_event, [])
            ends = self.events.get(end_event, [])

            if not starts or not ends:
                return None

            start = starts[-1]
            end = ends[-1]

        if end < start:
            return None

        return end - start
