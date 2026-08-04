# SPDX-License-Identifier: Apache-2.0
"""Long-document Q&A workload for ``lmcache bench engine``."""

# Standard
from dataclasses import dataclass
import asyncio
import random

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
class LongDocQAConfig:
    """Workload-specific config for the long-doc-qa workload."""

    document_length: int = 10000
    query_per_document: int = 2
    num_documents: int = 1
    shuffle_policy: str = "random"
    num_inflight_requests: int = 3
    max_output_length: int = 128

    def __post_init__(self) -> None:
        if self.document_length <= 0:
            raise ValueError(
                f"document_length must be positive, got {self.document_length}"
            )
        if self.query_per_document < 1:
            raise ValueError(
                f"query_per_document must be >= 1, got {self.query_per_document}"
            )
        if self.max_output_length < 1:
            raise ValueError(
                f"max_output_length must be >= 1, got {self.max_output_length}"
            )
        if self.num_inflight_requests < 1:
            raise ValueError(
                f"num_inflight_requests must be >= 1, got {self.num_inflight_requests}"
            )
        if self.shuffle_policy not in ("random", "tile"):
            raise ValueError(
                f"shuffle_policy must be 'random' or 'tile', "
                f"got {self.shuffle_policy!r}"
            )

    @classmethod
    def resolve(
        cls,
        kv_cache_volume_gb: float,
        tokens_per_gb_kvcache: int,
        document_length: int = 10000,
        query_per_document: int = 2,
        shuffle_policy: str = "random",
        num_inflight_requests: int = 3,
        max_output_length: int = 128,
    ) -> "LongDocQAConfig":
        """Create a config with ``num_documents`` computed from KV cache budget.

        Args:
            kv_cache_volume_gb: Target active KV cache volume in GB.
            tokens_per_gb_kvcache: Tokens fitting in 1 GB of KV cache.
            document_length: Token length of each document.
            query_per_document: Number of questions per document.
            shuffle_policy: Request ordering — ``"random"`` or ``"tile"``.
            num_inflight_requests: Max concurrent in-flight requests.
            max_output_length: Max tokens to generate per benchmark query.

        Returns:
            A fully-resolved LongDocQAConfig with computed num_documents.
        """
        num_documents = max(
            int(kv_cache_volume_gb * tokens_per_gb_kvcache / document_length),
            1,
        )
        logger.debug(
            "Computed num_documents=%d from kv_cache_volume_gb=%.1f, "
            "tokens_per_gb_kvcache=%d, document_length=%d",
            num_documents,
            kv_cache_volume_gb,
            tokens_per_gb_kvcache,
            document_length,
        )
        return cls(
            document_length=document_length,
            query_per_document=query_per_document,
            num_documents=num_documents,
            shuffle_policy=shuffle_policy,
            num_inflight_requests=num_inflight_requests,
            max_output_length=max_output_length,
        )


_QUESTIONS = [
    "What is this document about?",
    "Summarize the key points.",
    "What is the main topic discussed?",
    "Provide a brief overview.",
]


class LongDocQAWorkload(BaseWorkload):
    """Workload that simulates repeated Q&A over long documents.

    Generates synthetic documents, builds a request schedule, warms up
    the KV cache by sending each document once, then dispatches benchmark
    requests with semaphore-controlled concurrency.
    """

    def __init__(
        self,
        config: LongDocQAConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        seed: int = 42,
    ) -> None:
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._seed = seed

        self._documents = self._generate_documents()
        self._schedule = self._build_schedule()
        self._schedule_index = 0

        self._semaphore = asyncio.Semaphore(config.num_inflight_requests)
        self._pending_tasks: set[asyncio.Task] = set()

    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""
        c = self._config
        B = "\033[1m"  # bold
        C = "\033[96m"  # cyan
        Y = "\033[93m"  # yellow
        R = "\033[0m"  # reset
        total = c.num_documents * c.query_per_document
        print(
            f"{B}{'═' * 50}{R}\n"
            f"{B} Workload: {C}long-doc-qa{R}\n"
            f"{B}{'─' * 50}{R}\n"
            f"  Documents:        {Y}{c.num_documents}{R}\n"
            f"  Queries/doc:      {Y}{c.query_per_document}{R}\n"
            f"  Total requests:   {Y}{total}{R}\n"
            f"  Document length:  {Y}{c.document_length}{R} tokens\n"
            f"  Max inflight:     {Y}{c.num_inflight_requests}{R}\n"
            f"  Shuffle policy:   {Y}{c.shuffle_policy}{R}\n"
            f"{B}{'═' * 50}{R}"
        )

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _generate_documents(self) -> list[str]:
        """Generate synthetic documents of approximately ``document_length`` tokens."""
        documents = []
        for doc_id in range(self._config.num_documents):
            prefix = f"Document {doc_id}: "
            body = " ".join(["hi"] * max(self._config.document_length - 10, 1))
            documents.append(prefix + body)
        logger.debug(
            "Generated %d documents of ~%d tokens each",
            len(documents),
            self._config.document_length,
        )
        return documents

    def _build_schedule(self) -> list[tuple[int, int]]:
        """Build the request schedule as ``(doc_index, query_index)`` pairs.

        Tile policy uses query-major order: all documents for query 0,
        then all documents for query 1, etc.  Random policy shuffles
        the same pairs with a seeded RNG.
        """
        schedule: list[tuple[int, int]] = []
        for q_idx in range(self._config.query_per_document):
            for doc_idx in range(self._config.num_documents):
                schedule.append((doc_idx, q_idx))

        if self._config.shuffle_policy == "random":
            rng = random.Random(self._seed)
            rng.shuffle(schedule)

        logger.debug(
            "Built schedule with %d requests (policy=%s)",
            len(schedule),
            self._config.shuffle_policy,
        )
        return schedule

    def _build_messages(
        self,
        doc_index: int,
        query_index: int,
    ) -> list[dict[str, str]]:
        """Build chat messages for a benchmark request."""
        document = self._documents[doc_index]
        question = _QUESTIONS[query_index % len(_QUESTIONS)]
        content = f"{document}\n\nQuestion {query_index}: {question}"
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Send each document once with ``max_tokens=1`` to populate KV cache."""
        num_docs = self._config.num_documents
        for doc_idx in range(num_docs):
            request_id = f"warmup_doc{doc_idx}"
            messages = [{"role": "user", "content": self._documents[doc_idx]}]
            self._progress_monitor.log_message(f"Warmup {doc_idx + 1}/{num_docs}")
            self._progress_monitor.on_request_sent(request_id)
            result = await self._request_sender.send_warmup_request(
                request_id,
                messages,
            )
            if not result.successful:
                self._progress_monitor.log_message(
                    f"Warmup {request_id} failed: {result.error}"
                )
        self._progress_monitor.log_message(
            f"Warmup complete: {num_docs} documents sent",
        )

    # ------------------------------------------------------------------
    # Benchmark dispatch
    # ------------------------------------------------------------------

    async def step(self, time_offset: float) -> float:
        """Dispatch the next request if semaphore allows."""
        if self._schedule_index < len(self._schedule):
            await self._semaphore.acquire()
            doc_idx, q_idx = self._schedule[self._schedule_index]
            self._schedule_index += 1

            task = asyncio.create_task(self._dispatch(doc_idx, q_idx))
            self._pending_tasks.add(task)
            task.add_done_callback(self._on_task_done)
            return 0.0  # immediate re-call

        # All dispatched — wait for pending tasks.
        if self._pending_tasks:
            await asyncio.wait(
                self._pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            return 0.0

        return -1.0  # all done

    async def _dispatch(
        self,
        doc_index: int,
        query_index: int,
    ) -> None:
        """Send a single benchmark request, then release the semaphore."""
        request_id = f"doc{doc_index}_q{query_index}"
        messages = self._build_messages(doc_index, query_index)
        self._progress_monitor.on_request_sent(request_id)
        self._progress_monitor.log_message(
            f"Dispatched request {request_id} (doc {doc_index}, query {query_index})"
        )
        try:
            await self._request_sender.send_request(
                request_id,
                messages,
                max_tokens=self._config.max_output_length,
            )
        finally:
            self._semaphore.release()

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Clean up completed tasks and log unexpected errors."""
        self._pending_tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                self._progress_monitor.log_message(f"Dispatch task failed: {exc}")

    def on_request_finished(self, request_id: str, output: str) -> None:
        """No-op — this workload is stateless."""
