# SPDX-License-Identifier: Apache-2.0
"""Real-time terminal display for ``lmcache bench engine``."""

# Standard
import threading
import time

# First Party
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector

# ANSI escape codes
CLEAR_LINE = "\033[2K"
CURSOR_UP = "\033[A"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"


class ProgressMonitor:
    """Real-time terminal display for benchmark progress.

    Runs a daemon thread that redraws stats every second using ANSI
    cursor control for in-place updates.  Reads aggregated metrics
    from a ``StatsCollector`` and tracks in-flight count internally.
    """

    LOG_LINES = 5
    DISPLAY_LINES = 9 + LOG_LINES  # stats block + log lines

    def __init__(
        self,
        stats_collector: StatsCollector,
        quiet: bool = False,
    ) -> None:
        self._stats_collector = stats_collector
        self._quiet = quiet

        self._lock = threading.Lock()
        self._in_flight: int = 0
        self._log_messages: list[str] = []

        self._running: bool = False
        self._thread: threading.Thread = None  # type: ignore[assignment]
        self._first_draw: bool = True
        self._start_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the display daemon thread."""
        if self._quiet:
            return
        self._start_time = time.monotonic()
        self._running = True
        self._first_draw = True
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the display thread and print final state."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        if not self._quiet:
            self._draw()
            print()  # final newline

    def on_request_sent(self, request_id: str) -> None:
        """Called when a request is dispatched. Increments in-flight."""
        with self._lock:
            self._in_flight += 1

    def on_request_finished(self, request_id: str, successful: bool) -> None:
        """Called when a request completes. Decrements in-flight."""
        with self._lock:
            self._in_flight = max(self._in_flight - 1, 0)

    def log_message(self, message: str) -> None:
        """Add a log line to the display. Thread-safe."""
        with self._lock:
            self._log_messages.append(message)
            if len(self._log_messages) > self.LOG_LINES:
                self._log_messages = self._log_messages[-self.LOG_LINES :]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _display_loop(self) -> None:
        """Background loop: redraw every second."""
        while self._running:
            self._draw()
            time.sleep(1.0)

    def _draw(self) -> None:
        """Render the real-time stats display."""
        # Read stats from collector (already thread-safe)
        stats = self._stats_collector.get_current_stats()

        # Read local state under lock
        with self._lock:
            in_flight = self._in_flight
            log_messages = list(self._log_messages)

        elapsed = time.monotonic() - self._start_time

        # Move cursor up to overwrite previous display
        if not self._first_draw:
            print(f"\r{CURSOR_UP * self.DISPLAY_LINES}", end="")
        self._first_draw = False

        lines = [
            f"{BOLD}{'─' * 50}{RESET}",
            (f"{BOLD} Engine Benchmark{RESET}   elapsed: {CYAN}{elapsed:.0f}s{RESET}"),
            (
                f"  {GREEN}{stats.successful_requests}{RESET} successful"
                f"   {CYAN}{in_flight}{RESET} in-flight"
                f"   {RED}{stats.failed_requests}{RESET} failed"
            ),
            f"  Avg TTFT:       {YELLOW}{stats.mean_ttft_ms:.1f} ms{RESET}",
            (f"  Avg decode:     {YELLOW}{stats.mean_decode_speed:.1f} tok/s{RESET}"),
            f"  Input tokens:   {stats.total_input_tokens:,}",
            f"  Output tokens:  {stats.total_output_tokens:,}",
            (
                f"  Throughput:     "
                f"{stats.input_throughput:.1f} in "
                f"/ {stats.output_throughput:.1f} out tok/s"
            ),
            f"{BOLD}{'─' * 50}{RESET}",
        ]

        # Pad log lines to fixed count
        for i in range(self.LOG_LINES):
            if i < len(log_messages):
                lines.append(f"  {YELLOW}[log] {log_messages[i]}{RESET}")
            else:
                lines.append("")

        for line in lines:
            print(f"{CLEAR_LINE}{line}")
