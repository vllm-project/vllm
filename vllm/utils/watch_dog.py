# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import faulthandler
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.utils.safe_fs import get_user_root_dir, prepare_private_dir, safe_open_file

if TYPE_CHECKING:
    from logging import Logger

    from vllm.config import WatchdogConfig

_DEFAULT_NAME = "vllm"
_DEFAULT_TIMEOUT = 300
_DEFAULT_INTERVAL = 10


@dataclass
class WatchdogStat:
    """Statistics snapshot of the watchdog for one reporting window."""

    name: str
    num_timeouts: int
    num_recoveries: int


class WatchDog:
    def __init__(self):
        """Initialize the watchdog with default name, timeout, and check
        interval."""
        self._name = _DEFAULT_NAME
        self._timeout = _DEFAULT_TIMEOUT
        self._check_interval = _DEFAULT_INTERVAL
        self._dump_seq = 1

        # Initialize feed time to current time to avoid immediate timeout on startup
        self._last_feed_time = time.monotonic()
        # Last timeout log time, initialized to 0 to ensure the first
        # timeout is always logged
        self._last_timeout_log = 0.0

        # Timeout statistics.
        self._num_timeouts = 0
        self._num_recoveries = 0
        # Whether the watchdog is currently in a timeout state.
        self._is_timeout_active = False

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

    def set_config(
        self,
        timeout: int | None = None,
        check_interval: int | None = None,
        dump_dir: str | None = None,
    ) -> None:
        """Override watchdog parameters before start()."""
        if timeout is not None:
            self._timeout = timeout
        if check_interval is not None:
            self._check_interval = check_interval
        if dump_dir is not None:
            self._dump_dir = dump_dir
            self._dump_file = os.path.join(
                dump_dir,
                f"VLLM_STACK_DUMP_for_{self._name}_{os.getpid()}.log",
            )

    def set_logger(self, logger):
        """Set the logger used to report stack-dump failures."""
        self._logger = logger

    def feed(self):
        """Feed interface, external callers use this to update last feed time"""
        now = time.monotonic()
        if self._is_timeout_active:
            # The process hung and is now responsive again: record the
            # recovery.
            self._num_recoveries += 1
            self._is_timeout_active = False
        # Single float assignment is atomic in CPython
        self._last_feed_time = now

    def dump_stack(self, reason):
        """Dump all thread stack traces to the dump file for the given reason."""
        try:
            with self._dump_lock:
                with safe_open_file(self._dump_file, "a") as f:
                    f.write(
                        f"\nCall stack dump #{self._dump_seq} at "
                        f"{time.ctime()} due to {reason}\n\n"
                    )
                    faulthandler.dump_traceback(file=f, all_threads=True)
                    f.write(
                        "\n================================================================\n"
                    )
                self._dump_seq += 1
                dump_msg = (
                    f"[Watchdog]Dumped stack to {self._dump_file} due to {reason}"
                )
                if self._logger is not None:
                    self._logger.info(dump_msg)
                else:
                    print(dump_msg)
        except Exception as e:
            err_msg = f"[Watchdog]Failed to dump stack trace to {self._dump_file}: {e}"
            if self._logger is not None:
                self._logger.warning(err_msg)
            else:
                print(err_msg)

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
                # The timeout started once the last feed went stale; the
                # recovery is settled in feed().
                self._is_timeout_active = True
                self._num_timeouts += 1
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
        start_msg = f"[Watchdog]Started thread for {self._name} (pid={os.getpid()})"
        if self._logger is not None:
            self._logger.info(start_msg)
        else:
            print(start_msg)

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

    @property
    def num_timeouts(self) -> int:
        """Total number of feed timeouts observed."""
        return self._num_timeouts

    @property
    def num_recoveries(self) -> int:
        """Total number of times the process recovered from a timeout."""
        return self._num_recoveries

    def take_timeout_stats(self) -> WatchdogStat:
        """Return the accumulated timeout statistics and reset them.

        Intended for periodic reporting (e.g. attached to per-step outputs)
        so that each consumer observes only the stats since the last read.

        Returns:
            A ``WatchdogStat`` carrying the watchdog name plus
            (num_timeouts, num_recoveries).

        """
        stats = WatchdogStat(
            name=self._name,
            num_timeouts=self._num_timeouts,
            num_recoveries=self._num_recoveries,
        )
        self._num_timeouts = 0
        self._num_recoveries = 0
        return stats


_watch_dog = WatchDog()


def get_watch_dog() -> WatchDog:
    """Return the process-wide WatchDog singleton."""
    return _watch_dog


def start_watch_dog(
    name: str,
    watchdog_config: "WatchdogConfig",
    logger: "Logger | None" = None,
) -> WatchDog:
    """Start the process-wide WatchDog background thread if enabled.

    The watchdog is enabled only when ``watchdog_config.dump_dir`` is set;
    otherwise it stays dormant. Returns the WatchDog singleton either way.
    """
    config_msg = (
        "[Watchdog]Start with "
        f"timeout={watchdog_config.timeout} "
        f"check_interval={watchdog_config.check_interval} "
        f"dump_dir={watchdog_config.dump_dir!r}"
    )
    if logger is not None:
        logger.info(config_msg)
    else:
        print(config_msg)
    if not watchdog_config.dump_dir:
        return _watch_dog
    _watch_dog.set_name(name)
    _watch_dog.set_config(
        timeout=watchdog_config.timeout,
        check_interval=watchdog_config.check_interval,
        dump_dir=watchdog_config.dump_dir,
    )
    if logger is not None:
        _watch_dog.set_logger(logger)
    _watch_dog.start()
    return _watch_dog
