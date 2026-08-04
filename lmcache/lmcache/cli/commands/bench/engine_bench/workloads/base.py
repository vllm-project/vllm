# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for engine benchmark workloads."""

# Standard
from abc import ABC, abstractmethod
import asyncio
import queue
import time

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.logging import init_logger

logger = init_logger(__name__)


class BaseWorkload(ABC):
    """Abstract base class for all engine benchmark workloads.

    Owns the internal dispatch loop that calls ``step()`` on the
    concrete workload.  Provides a thread-safe ``request_finished``
    callback that enqueues completed requests for the loop to drain.
    """

    def __init__(
        self,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
    ) -> None:
        self._request_sender = request_sender
        self._stats_collector = stats_collector
        self._progress_monitor = progress_monitor
        self._finished_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def warmup(self) -> None:
        """Run warmup requests.  Blocks until all warmup is done."""

    @abstractmethod
    async def step(self, time_offset: float) -> float:
        """Execute one step of the workload.

        Args:
            time_offset: seconds since benchmark start.

        Returns:
            Next wakeup time offset (absolute, from benchmark start).
            Return a negative value to signal that the workload is done.
        """

    @abstractmethod
    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""

    @abstractmethod
    def on_request_finished(self, request_id: str, output: str) -> None:
        """Called when a request finishes (from the loop thread).

        Stateless workloads can implement this as a no-op.
        Stateful workloads use it to record responses in session history.
        """

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run warmup + benchmark, closing the HTTP client in the same loop.

        Blocks until the workload is complete. The ``RequestSender``'s
        httpx transports are bound to this loop, so they must be closed
        before ``asyncio.run`` destroys it — otherwise a later close
        from a fresh loop raises ``RuntimeError: Event loop is closed``.
        """

        async def _run_and_close() -> None:
            try:
                await self._run_async()
            finally:
                await self._request_sender.close()

        asyncio.run(_run_and_close())

    async def _run_async(self) -> None:
        """Internal async implementation of the run loop."""
        self._progress_monitor.log_message("Starting warmup phase")
        await self.warmup()

        self._progress_monitor.log_message(
            "Warmup complete, starting benchmark",
        )
        self._stats_collector.reset()
        self._drain_finished_queue()  # discard warmup completions

        start_time = time.monotonic()

        while True:
            self._drain_finished_queue()
            time_offset = time.monotonic() - start_time
            next_wakeup = await self.step(time_offset)
            if next_wakeup < 0:
                break
            sleep_duration = max(0.0, next_wakeup - (time.monotonic() - start_time))
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

        self._drain_finished_queue()  # final drain
        self._progress_monitor.log_message("Benchmark complete")

    def request_finished(self, result, response_text: str) -> None:
        """Thread-safe callback matching ``OnFinishedCallback`` signature.

        Enqueues ``(request_id, response_text)`` for the loop to drain.
        Registered on ``RequestSender.on_finished`` by the orchestrator.
        """
        self._finished_queue.put((result.request_id, response_text))

    def _drain_finished_queue(self) -> None:
        """Drain all completed requests and call ``on_request_finished``."""
        while True:
            try:
                request_id, output = self._finished_queue.get_nowait()
            except queue.Empty:
                break
            self.on_request_finished(request_id, output)
