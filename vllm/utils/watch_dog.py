# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import faulthandler
import os
import threading
import time

from vllm.utils.safe_fs import get_user_root_dir, prepare_private_dir, safe_open_file

_DEFAULT_NAME = "vllm"
_DEFAULT_TIMEOUT = 300
_DEFAULT_INTERVAL = 10


class WatchDog:
    def __init__(self):
        """Initialize the watchdog with default name, timeout, and check
        interval."""
        self._name = _DEFAULT_NAME
        self._timeout = _DEFAULT_TIMEOUT
        self._check_interval = _DEFAULT_INTERVAL
        self._sequence = 1

        # Initialize feed time to current time to avoid immediate timeout on startup
        self._last_feed_time = time.monotonic()
        # Last timeout log time, initialized to 0 to ensure the first
        # timeout is always logged
        self._last_timeout_log = 0.0

        self._stop_event = threading.Event()
        self._thread = None
        self._dump_lock = threading.Lock()
        self._dump_dir = os.path.join(get_user_root_dir(), "dump")
        self._dump_file = os.path.join(
            self._dump_dir,
            f"VLLM_STACK_DUMP_for_{self._name}_{os.getpid()}.log",
        )
        self._logger = None

    def set_name(self, name):
        """Set the watchdog name for debugging."""
        self._name = name
        self._dump_file = os.path.join(
            self._dump_dir,
            f"VLLM_STACK_DUMP_for_{self._name}_{os.getpid()}.log",
        )

    def set_logger(self, logger):
        """Set the logger used to report stack-dump failures."""
        self._logger = logger

    def feed(self):
        """Feed interface, external callers use this to update last feed time"""
        # Single float assignment is atomic in CPython
        self._last_feed_time = time.monotonic()

    def dump_stack(self, reason):
        """Dump all thread stack traces to the dump file for the given reason."""
        try:
            with self._dump_lock:
                with safe_open_file(self._dump_file, "a") as f:
                    f.write(
                        f"\nCall stack dump #{self._sequence} at "
                        f"{time.ctime()} due to {reason}\n\n"
                    )
                    faulthandler.dump_traceback(file=f, all_threads=True)
                    f.write(
                        "\n================================================================\n"
                    )
                self._sequence += 1
                if self._logger is not None:
                    self._logger.info(
                        "Call stack dumped to %s due to %s",
                        self._dump_file,
                        reason,
                    )
        except Exception as e:
            if self._logger is not None:
                self._logger.warning(
                    "Failed to dump stack trace to %s: %s",
                    self._dump_file,
                    e,
                )

    def _check_loop(self):
        """Main loop of the background check thread"""
        while not self._stop_event.is_set():
            # Wait on the stop event so a stop request wakes the loop
            # immediately instead of sleeping the full check interval.
            if self._stop_event.wait(self._check_interval):
                break

            now = time.monotonic()
            # Calculate time elapsed since last feed
            if (
                now - self._last_feed_time > self._timeout
                and self._last_timeout_log < self._last_feed_time
            ):
                # If last timeout log time is earlier than last feed time,
                # the timeout hasn't been logged since the last feed; update
                # the log timestamp to avoid duplicate logs.
                self._last_timeout_log = now
                self.dump_stack("feed timeout")
            # If not timed out, do nothing and keep _last_timeout_log unchanged

    def start(self):
        """Start the watchdog background thread (daemon thread)"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        faulthandler.enable(all_threads=True)
        prepare_private_dir(self._dump_dir)
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self.feed()
        self._thread.start()

    def stop(self):
        """Stop the watchdog background thread"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if not self._thread.is_alive():
                # Only clear the reference once the thread has actually
                # exited; keeping it otherwise prevents a later start()
                # from creating a duplicate live monitor thread.
                self._thread = None


_watch_dog = WatchDog()


def get_watch_dog() -> WatchDog:
    """Return the process-wide WatchDog singleton."""
    return _watch_dog
