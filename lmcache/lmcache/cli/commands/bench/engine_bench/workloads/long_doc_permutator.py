# SPDX-License-Identifier: Apache-2.0
"""Long-document permutator workload for ``lmcache bench engine``.

Stress-tests blended KV cache reuse by sending permutations of a set of
context documents.  Each request is:

    [System Prompt] + [Doc_i1] + [Doc_i2] + ... + [Doc_iN]

where (i1, ..., iN) is one permutation of the N contexts.

Stress axes (controlled by config):
    1. Blended Context Boundaries  -> num_contexts
    2. Eviction                    -> num_permutations
    3. Chunk Homogeneity           -> vocab_size
    4. Prefix Domination           -> system_prompt_length
    5. Concurrency                 -> num_inflight_requests
"""

# Standard
from dataclasses import dataclass
import asyncio
import itertools
import math
import random

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class LongDocPermutatorConfig:
    """Workload-specific config for the long-doc-permutator workload."""

    num_contexts: int = 5
    context_length: int = 5000
    system_prompt_length: int = 1000
    num_permutations: int = 10
    vocab_size: int = 8000
    num_inflight_requests: int = 1

    def __post_init__(self) -> None:
        if self.num_contexts < 1:
            raise ValueError(f"num_contexts must be >= 1, got {self.num_contexts}")
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )
        if self.num_permutations < 1:
            raise ValueError(
                f"num_permutations must be >= 1, got {self.num_permutations}"
            )
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.num_inflight_requests < 1:
            raise ValueError(
                f"num_inflight_requests must be >= 1, got {self.num_inflight_requests}"
            )

    @classmethod
    def resolve(
        cls,
        num_contexts: int = 5,
        context_length: int = 5000,
        system_prompt_length: int = 1000,
        num_permutations: int = 10,
        vocab_size: int = 8000,
        num_inflight_requests: int = 1,
    ) -> "LongDocPermutatorConfig":
        """Create a config directly from the provided parameters.

        Args:
            num_contexts: Number of unique context documents.
            context_length: Token length of each context.
            system_prompt_length: Token length of the shared system prompt.
                Use 0 for no system prompt.
            num_permutations: Number of distinct permutations to send.
                Capped at N! where N = num_contexts.
            vocab_size: Vocabulary pool size for context generation.
                Smaller values increase chunk hash collision risk.
            num_inflight_requests: Max concurrent in-flight requests.

        Returns:
            A fully-resolved LongDocPermutatorConfig.
        """
        return cls(
            num_contexts=num_contexts,
            context_length=context_length,
            system_prompt_length=system_prompt_length,
            num_permutations=num_permutations,
            vocab_size=vocab_size,
            num_inflight_requests=num_inflight_requests,
        )


# ---------------------------------------------------------------------------
# Prompt generation helpers (module-level, before classes)
# ---------------------------------------------------------------------------


def _generate_vocab_pool(size: int, seed: int = 42) -> list[str]:
    """Generate a vocabulary pool of ``size`` unique pseudo-words.

    Deterministically generates synthetic words so every token is unique.

    Args:
        size: Number of unique words to generate.
        seed: Random seed for reproducibility.

    Returns:
        Sorted list of unique pseudo-words.
    """
    rng = random.Random(seed)
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    pool: set[str] = set()
    while len(pool) < size:
        length = rng.randint(3, 7)
        word = ""
        for j in range(length):
            if j % 2 == 0:
                word += rng.choice(consonants)
            else:
                word += rng.choice(vowels)
        word = f"{word}{len(pool)}"
        pool.add(word)
    return sorted(pool)


def _generate_system_prompt(length: int, seed: int = 42) -> str:
    """Generate a deterministic shared system prompt of ~``length`` tokens.

    Args:
        length: Approximate token length of the system prompt.
        seed: Random seed for reproducibility.

    Returns:
        A string of ``length`` space-separated words.
    """
    if length == 0:
        return ""
    rng = random.Random(seed)
    words = [
        "the",
        "system",
        "will",
        "process",
        "your",
        "request",
        "and",
        "provide",
        "an",
        "answer",
        "based",
        "on",
        "context",
    ]
    return " ".join(rng.choices(words, k=length))


def _generate_contexts(
    num_contexts: int,
    length: int,
    vocab_pool: list[str],
    seed: int = 123,
) -> list[str]:
    """Generate ``num_contexts`` unique context blocks of ~``length`` tokens.

    Each context draws from ``vocab_pool`` with a per-context seed so the
    token sequences genuinely diverge.

    Args:
        num_contexts: Number of context documents to generate.
        length: Approximate token length of each context.
        vocab_pool: Pool of words to sample from.
        seed: Base random seed; each context uses seed + i.

    Returns:
        List of context strings.
    """
    contexts = []
    for i in range(num_contexts):
        rng = random.Random(seed + i)
        body = " ".join(rng.choices(vocab_pool, k=length))
        contexts.append(body)
    return contexts


def _enumerate_permutations(
    num_contexts: int,
    num_permutations: int,
    seed: int = 0,
) -> list[tuple[int, ...]]:
    """Enumerate up to ``num_permutations`` distinct permutations.

    Returns all N! permutations when num_permutations >= N!.  For large N
    uses random sampling to avoid iterating an enormous search space.

    Args:
        num_contexts: Number of contexts (N).
        num_permutations: Maximum number of permutations to return.
        seed: Random seed used when sampling is needed.

    Returns:
        List of permutation tuples.
    """
    total_possible = math.factorial(num_contexts)
    if num_permutations >= total_possible:
        return list(itertools.permutations(range(num_contexts)))

    if total_possible > num_permutations * 10:
        rng = random.Random(seed)
        seen: set[tuple[int, ...]] = set()
        indices = list(range(num_contexts))
        while len(seen) < num_permutations:
            perm = tuple(rng.sample(indices, len(indices)))
            seen.add(perm)
        return sorted(seen)

    result = []
    for perm in itertools.permutations(range(num_contexts)):
        result.append(perm)
        if len(result) >= num_permutations:
            break
    return result


# ---------------------------------------------------------------------------
# Workload class
# ---------------------------------------------------------------------------


class LongDocPermutatorWorkload(BaseWorkload):
    """Workload that sends permutations of context documents.

    Generates synthetic contexts from a vocab pool, enumerates permutations,
    and dispatches requests with semaphore-controlled concurrency.  Includes
    a single dummy warmup request to prime the engine.
    """

    def __init__(
        self,
        config: LongDocPermutatorConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        seed: int = 42,
    ) -> None:
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._seed = seed

        vocab_pool = _generate_vocab_pool(config.vocab_size, seed=seed)
        self._system_prompt = _generate_system_prompt(
            config.system_prompt_length, seed=seed
        )
        self._contexts = _generate_contexts(
            config.num_contexts, config.context_length, vocab_pool, seed=seed + 1
        )
        self._permutations = _enumerate_permutations(
            config.num_contexts, config.num_permutations, seed=seed
        )
        self._request_list = self._build_request_list()
        self._request_index = 0

        self._semaphore = asyncio.Semaphore(config.num_inflight_requests)
        self._pending_tasks: set[asyncio.Task] = set()

        logger.debug(
            "LongDocPermutator: %d contexts x %d permutations = %d requests",
            config.num_contexts,
            len(self._permutations),
            len(self._request_list),
        )

    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""
        c = self._config
        B = "\033[1m"
        C = "\033[96m"
        Y = "\033[93m"
        R = "\033[0m"
        total = len(self._request_list)
        actual_perms = len(self._permutations)
        print(
            f"{B}{'═' * 50}{R}\n"
            f"{B} Workload: {C}long-doc-permutator{R}\n"
            f"{B}{'─' * 50}{R}\n"
            f"  Contexts:            {Y}{c.num_contexts}{R}\n"
            f"  Context length:      {Y}{c.context_length}{R} tokens\n"
            f"  System prompt:       {Y}{c.system_prompt_length}{R} tokens\n"
            f"  Permutations:        {Y}{actual_perms}{R} "
            f"(of {math.factorial(c.num_contexts)} possible)\n"
            f"  Total requests:      {Y}{total}{R}\n"
            f"  Vocab size:          {Y}{c.vocab_size}{R}\n"
            f"  Max inflight:        {Y}{c.num_inflight_requests}{R}\n"
            f"{B}{'═' * 50}{R}"
        )

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _build_request_list(self) -> list[tuple[list[dict[str, str]], int]]:
        """Build the full request list from the enumerated permutations.

        Each entry is ``(messages, permutation_index)`` where messages
        concatenates all contexts in permutation order into a single user
        message, preceded by the system prompt.

        Returns:
            List of (messages, permutation_index) tuples.
        """
        requests: list[tuple[list[dict[str, str]], int]] = []
        for perm_idx, perm in enumerate(self._permutations):
            concatenated = "\n\n".join(self._contexts[i] for i in perm)
            messages: list[dict[str, str]] = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": concatenated})
            requests.append((messages, perm_idx))
        return requests

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Send a single dummy warmup request to prime the engine."""
        request_id = "warmup_0"
        dummy_content = " ".join(["warmup"] * 500)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": dummy_content},
        ]
        self._progress_monitor.log_message("Warmup (1 dummy request)")
        self._progress_monitor.on_request_sent(request_id)
        result = await self._request_sender.send_warmup_request(request_id, messages)
        if not result.successful:
            self._progress_monitor.log_message(f"Warmup request failed: {result.error}")
        self._progress_monitor.log_message("Warmup complete")

    # ------------------------------------------------------------------
    # Benchmark dispatch
    # ------------------------------------------------------------------

    async def step(self, time_offset: float) -> float:
        """Dispatch the next permutation request if semaphore allows.

        Args:
            time_offset: Seconds since benchmark start (unused).

        Returns:
            0.0 to request an immediate re-call, or -1.0 when all done.
        """
        if self._request_index < len(self._request_list):
            await self._semaphore.acquire()
            req_idx = self._request_index
            messages, perm_idx = self._request_list[req_idx]
            self._request_index += 1

            task = asyncio.create_task(self._dispatch(messages, perm_idx, req_idx))
            self._pending_tasks.add(task)
            task.add_done_callback(self._on_task_done)
            return 0.0

        if self._pending_tasks:
            await asyncio.wait(
                self._pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            return 0.0

        return -1.0

    async def _dispatch(
        self,
        messages: list[dict[str, str]],
        perm_idx: int,
        req_idx: int,
    ) -> None:
        """Send a single benchmark request, then release the semaphore.

        Args:
            messages: Chat messages for the request.
            perm_idx: Index of the permutation (used for request ID).
            req_idx: The request index captured before incrementing the counter.
        """
        request_id = f"perm{perm_idx}_req{req_idx}"
        self._progress_monitor.on_request_sent(request_id)
        self._progress_monitor.log_message(f"Dispatched permutation {perm_idx}")
        try:
            await self._request_sender.send_request(request_id, messages)
        finally:
            self._semaphore.release()

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Clean up completed tasks and log unexpected errors.

        Args:
            task: The completed asyncio Task.
        """
        self._pending_tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                self._progress_monitor.log_message(f"Dispatch task failed: {exc}")

    def on_request_finished(self, request_id: str, output: str) -> None:
        """No-op — this workload is stateless."""
