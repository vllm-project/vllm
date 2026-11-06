# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import faulthandler
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from logging import Logger

from vllm.config import WatchdogConfig


class WatchDog(ABC):
    @abstractmethod
    def feed(self) -> None:
        """Mark that the process is making progress."""

    @abstractmethod
    def disable(self) -> AbstractContextManager[None]:
        """Suspend timeout detection, e.g. while blocking on idle input."""

    @abstractmethod
    def handle_failure(self, reason: str) -> None:
        """Record the state of all threads for debugging."""


class NoopWatchDog(WatchDog):
    def feed(self) -> None:
        pass

    @contextmanager
    def disable(self) -> Iterator[None]:
        yield

    def handle_failure(self, reason: str) -> None:
        pass


class ThreadedWatchDog(WatchDog):
    """Dumps the stacks of all threads when not fed within the timeout."""

    def __init__(self, name: str, watchdog_config: WatchdogConfig, logger: Logger):
        self.name = name
        self.timeout = watchdog_config.timeout
        self.check_interval = watchdog_config.check_interval
        self.dump_dir = watchdog_config.dump_dir
        self.dump_file = os.path.join(
            self.dump_dir, f"VLLM_STACK_DUMP_for_{name}_{os.getpid()}.log"
        )
        self.dump_seq = 1
        self.logger = logger
        self.last_feed_time = time.monotonic()
        # 0.0 so that the first timeout is always dumped.
        self.last_timeout_log = 0.0
        self.active = True
        # Reentrant: handle_failure() may be re-entered from a signal handler.
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.handler = None

    def feed(self) -> None:
        self.last_feed_time = time.monotonic()

    @contextmanager
    def disable(self) -> Iterator[None]:
        assert self.active, "WatchDog.disable() is not re-entrant"
        self.active = False
        try:
            yield
        finally:
            self.active = True

    def handle_failure(self, reason: str) -> None:
        """Append a numbered dump of all thread stacks to the dump file."""
        with self.lock:
            if self.handler is None:
                self.dump_stack(reason)
            else:
                self.handler(reason)

    def dump_stack(self, reason: str) -> None:
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            with open(self.dump_file, "a") as f:
                f.write(
                    f"\n>>> Call stack dump #{self.dump_seq} at "
                    f"{time.ctime()} due to {reason}\n\n"
                )
                faulthandler.dump_traceback(file=f, all_threads=True)
                f.write("\n" + "=" * 64 + "\n")
            self.dump_seq += 1
            self.logger.info(
                "[Watchdog]Dumped stack to %s due to %s", self.dump_file, reason
            )
        except Exception as e:
            self.logger.warning(
                "[Watchdog]Failed to dump stack trace to %s: %s",
                self.dump_file,
                e,
            )

    def set_handler(self, handler) -> None:
        self.handler = handler

    def _check_loop(self) -> None:
        while not self.stop_event.wait(self.check_interval):
            now = time.monotonic()
            # While disabled, treat the watchdog as continuously fed so the
            # timeout window restarts once it is re-enabled.
            if not self.active:
                self.last_feed_time = now
                continue
            # Dump at most once per stall.
            if (
                now - self.last_feed_time > self.timeout
                and self.last_timeout_log < self.last_feed_time
            ):
                self.last_timeout_log = now
                self.handle_failure("feed timeout")

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._check_loop, daemon=True)
        self.feed()
        self.thread.start()
        self.logger.info(
            "[Watchdog]Started thread for %s (pid=%d)", self.name, os.getpid()
        )

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            # Keep a still-alive thread so a later start() won't spawn another.
            if not self.thread.is_alive():
                self.thread = None


_watch_dog: WatchDog = NoopWatchDog()


def get_watch_dog() -> WatchDog:
    """Return the process-wide WatchDog singleton."""
    return _watch_dog


def start_watch_dog(
    name: str,
    watchdog_config: WatchdogConfig,
    logger: Logger,
) -> WatchDog:
    """Start the process-wide watchdog if ``watchdog_config.dump_dir`` is set.

    Returns the WatchDog singleton either way.
    """
    global _watch_dog
    if not watchdog_config.dump_dir:
        return _watch_dog

    logger.info(
        "[Watchdog]Start with timeout=%s check_interval=%s dump_dir=%s",
        watchdog_config.timeout,
        watchdog_config.check_interval,
        watchdog_config.dump_dir,
    )
    faulthandler.enable(all_threads=True)
    _watch_dog = ThreadedWatchDog(name, watchdog_config, logger)
    _watch_dog.start()
    return _watch_dog
