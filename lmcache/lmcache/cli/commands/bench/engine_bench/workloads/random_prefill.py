# SPDX-License-Identifier: Apache-2.0
"""Random-prefill workload for ``lmcache bench engine``."""

# Standard
from dataclasses import dataclass
import asyncio

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import (
    RequestSender,
)
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class RandomPrefillConfig:
    """Workload-specific config for the random-prefill workload."""

    request_length: int = 10000
    num_requests: int = 50

    def __post_init__(self) -> None:
        if self.request_length <= 0:
            raise ValueError(
                f"request_length must be positive, got {self.request_length}"
            )
        if self.num_requests < 1:
            raise ValueError(f"num_requests must be >= 1, got {self.num_requests}")

    @classmethod
    def resolve(
        cls,
        request_length: int = 10000,
        num_requests: int = 50,
    ) -> "RandomPrefillConfig":
        """Create a config from CLI args.

        Unlike other workloads, random-prefill does not use the KV cache
        budget to compute request count — the user specifies it directly.
        """
        return cls(
            request_length=request_length,
            num_requests=num_requests,
        )


class RandomPrefillWorkload(BaseWorkload):
    """Workload that tests prefill speed by firing all requests at once.

    Generates synthetic prompts of ``request_length`` tokens and dispatches
    all ``num_requests`` simultaneously with ``max_tokens=1``.  There is
    no warmup phase.
    """

    def __init__(
        self,
        config: RandomPrefillConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        seed: int = 42,
    ) -> None:
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._seed = seed

        self._prompts = self._generate_prompts()
        self._dispatched = False
        self._pending_tasks: set[asyncio.Task] = set()

    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""
        c = self._config
        B = "\033[1m"  # bold
        C = "\033[96m"  # cyan
        Y = "\033[93m"  # yellow
        R = "\033[0m"  # reset
        print(
            f"{B}{'═' * 50}{R}\n"
            f"{B} Workload: {C}random-prefill{R}\n"
            f"{B}{'─' * 50}{R}\n"
            f"  Requests:         {Y}{c.num_requests}{R}\n"
            f"  Request length:   {Y}{c.request_length}{R} tokens\n"
            f"  Max output:       {Y}1{R} token\n"
            f"{B}{'═' * 50}{R}"
        )

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _generate_prompts(self) -> list[str]:
        """Generate synthetic prompts of approximately ``request_length`` tokens."""
        prompts: list[str] = []
        for i in range(self._config.num_requests):
            prefix = f"Request {i}: "
            body = " ".join(["hi"] * max(self._config.request_length - 10, 1))
            prompts.append(prefix + body)
        logger.debug(
            "Generated %d prompts of ~%d tokens each",
            len(prompts),
            self._config.request_length,
        )
        return prompts

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """No warmup for random-prefill."""

    # ------------------------------------------------------------------
    # Benchmark dispatch
    # ------------------------------------------------------------------

    async def step(self, time_offset: float) -> float:
        """Dispatch all requests at once on the first call.

        Returns:
            0.0 while tasks are pending, -1.0 when all done.
        """
        if not self._dispatched:
            self._dispatched = True
            for i, prompt in enumerate(self._prompts):
                request_id = f"prefill_{i}"
                messages = [{"role": "user", "content": prompt}]
                self._progress_monitor.on_request_sent(request_id)

                task = asyncio.create_task(
                    self._dispatch(request_id, messages),
                )
                self._pending_tasks.add(task)
                task.add_done_callback(self._on_task_done)

            self._progress_monitor.log_message(
                f"Dispatched all {self._config.num_requests} requests"
            )
            return 0.0

        # Wait for pending tasks
        if self._pending_tasks:
            await asyncio.wait(
                self._pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            return 0.0

        return -1.0

    async def _dispatch(
        self,
        request_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Send a single prefill request with max_tokens=1."""
        await self._request_sender.send_request(
            request_id,
            messages,
            max_tokens=1,
        )

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Clean up completed tasks and log unexpected errors."""
        self._pending_tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                self._progress_monitor.log_message(f"Dispatch task failed: {exc}")

    def on_request_finished(self, request_id: str, output: str) -> None:
        """No-op — this workload is stateless."""
        self._progress_monitor.log_message(f"Request {request_id} finished")
