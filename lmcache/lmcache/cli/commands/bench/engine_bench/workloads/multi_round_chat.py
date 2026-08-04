# SPDX-License-Identifier: Apache-2.0
"""Multi-round chat workload for ``lmcache bench engine``."""

# Standard
from dataclasses import dataclass, field
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
class MultiRoundChatConfig:
    """Workload-specific config for the multi-round-chat workload."""

    shared_prompt_length: int = 2000
    chat_history_length: int = 10000
    user_input_length: int = 50
    output_length: int = 200
    qps: float = 1.0
    duration: float = 60.0
    num_concurrent_users: int = 1

    def __post_init__(self) -> None:
        if self.shared_prompt_length <= 0:
            raise ValueError(
                f"shared_prompt_length must be positive, "
                f"got {self.shared_prompt_length}"
            )
        if self.chat_history_length <= 0:
            raise ValueError(
                f"chat_history_length must be positive, got {self.chat_history_length}"
            )
        if self.user_input_length < 1:
            raise ValueError(
                f"user_input_length must be >= 1, got {self.user_input_length}"
            )
        if self.output_length < 1:
            raise ValueError(f"output_length must be >= 1, got {self.output_length}")
        if self.qps <= 0:
            raise ValueError(f"qps must be positive, got {self.qps}")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.num_concurrent_users < 1:
            raise ValueError(
                f"num_concurrent_users must be >= 1, got {self.num_concurrent_users}"
            )

    @classmethod
    def resolve(
        cls,
        kv_cache_volume_gb: float,
        tokens_per_gb_kvcache: int,
        shared_prompt_length: int = 2000,
        chat_history_length: int = 10000,
        user_input_length: int = 50,
        output_length: int = 200,
        qps: float = 1.0,
        duration: float = 60.0,
    ) -> "MultiRoundChatConfig":
        """Create a config with ``num_concurrent_users`` computed from KV cache budget.

        Args:
            kv_cache_volume_gb: Target active KV cache volume in GB.
            tokens_per_gb_kvcache: Tokens fitting in 1 GB of KV cache.
            shared_prompt_length: Token length of the system prompt.
            chat_history_length: Token length of pre-filled history.
            user_input_length: Token length per user query.
            output_length: Max tokens to generate per response.
            qps: Queries per second.
            duration: Benchmark duration in seconds.

        Returns:
            A fully-resolved MultiRoundChatConfig.
        """
        tokens_per_session = shared_prompt_length + chat_history_length
        total_tokens = kv_cache_volume_gb * tokens_per_gb_kvcache
        num_users = max(1, int(total_tokens / tokens_per_session))
        logger.debug(
            "Computed num_concurrent_users=%d from kv_cache_volume_gb=%.1f, "
            "tokens_per_gb_kvcache=%d, tokens_per_session=%d",
            num_users,
            kv_cache_volume_gb,
            tokens_per_gb_kvcache,
            tokens_per_session,
        )
        return cls(
            shared_prompt_length=shared_prompt_length,
            chat_history_length=chat_history_length,
            user_input_length=user_input_length,
            output_length=output_length,
            qps=qps,
            duration=duration,
            num_concurrent_users=num_users,
        )


@dataclass
class Session:
    """A single stateful chat session.

    Maintains a system prompt, pre-filled history, and accumulated
    Q&A exchanges.  The ``in_flight`` flag tracks whether a request
    is currently pending for this session.
    """

    session_id: int
    system_prompt: str
    history_text: str
    exchanges: list[tuple[str, str]] = field(default_factory=list)
    in_flight: bool = False

    def build_messages(self, query: str) -> list[dict[str, str]]:
        """Construct OpenAI-format messages for a request."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if self.history_text:
            messages.append(
                {"role": "user", "content": self.history_text},
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Understood, I have read the context above.",
                },
            )
        for q, a in self.exchanges:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": query})
        return messages

    def record_answer(self, query: str, answer: str) -> None:
        """Record a completed exchange and mark session as ready."""
        self.exchanges.append((query, answer))
        self.in_flight = False


class MultiRoundChatWorkload(BaseWorkload):
    """Workload that simulates multi-round chat with stateful sessions.

    Creates multiple concurrent user sessions, dispatches requests at a
    fixed QPS rate using round-robin scheduling, and records responses
    in session history so subsequent queries include prior context.
    """

    def __init__(
        self,
        config: MultiRoundChatConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        seed: int = 42,
    ) -> None:
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._seed = seed

        self._sessions = self._create_sessions()
        self._global_index = 0
        self._interval = 1.0 / config.qps
        self._pending_info: dict[str, tuple[int, str]] = {}
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
            f"{B} Workload: {C}multi-round-chat{R}\n"
            f"{B}{'─' * 50}{R}\n"
            f"  Concurrent users: {Y}{c.num_concurrent_users}{R}\n"
            f"  Prompt length:    {Y}{c.shared_prompt_length}{R} tokens\n"
            f"  History length:   {Y}{c.chat_history_length}{R} tokens\n"
            f"  Query length:     {Y}{c.user_input_length}{R} tokens\n"
            f"  Output length:    {Y}{c.output_length}{R} tokens\n"
            f"  QPS:              {Y}{c.qps}{R}\n"
            f"  Duration:         {Y}{c.duration}s{R}\n"
            f"{B}{'═' * 50}{R}"
        )

    # ------------------------------------------------------------------
    # Session creation
    # ------------------------------------------------------------------

    def _create_sessions(self) -> list[Session]:
        """Create sessions with synthetic prompts and history."""
        sessions: list[Session] = []
        for i in range(self._config.num_concurrent_users):
            system_prompt = self._generate_system_prompt(i)
            history = self._generate_history(i)
            sessions.append(
                Session(
                    session_id=i,
                    system_prompt=system_prompt,
                    history_text=history,
                )
            )
        logger.debug(
            "Created %d sessions (prompt=%d, history=%d tokens each)",
            len(sessions),
            self._config.shared_prompt_length,
            self._config.chat_history_length,
        )
        return sessions

    def _generate_system_prompt(self, session_id: int) -> str:
        """Generate a system prompt of approximately ``shared_prompt_length`` tokens."""
        prefix = f"Session {session_id}. You are a helpful assistant. "
        remaining = max(0, self._config.shared_prompt_length - len(prefix.split()))
        return prefix + " ".join(["help"] * remaining)

    def _generate_history(self, session_id: int) -> str:
        """
        Generate pre-filled history of approximately ``chat_history_length`` tokens.
        """
        prefix = f"[Session {session_id} history] "
        remaining = max(0, self._config.chat_history_length - len(prefix.split()))
        return prefix + " ".join(["hi"] * remaining)

    def _generate_query(self) -> str:
        """Generate a user query of approximately ``user_input_length`` tokens."""
        return " ".join(["tell"] * self._config.user_input_length)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Send one warmup request per session with ``max_tokens=1``."""
        num_sessions = len(self._sessions)
        for session in self._sessions:
            request_id = f"warmup_s{session.session_id}"
            messages = session.build_messages("Hello")
            self._progress_monitor.log_message(
                f"Warmup {session.session_id + 1}/{num_sessions}"
            )
            self._progress_monitor.on_request_sent(request_id)
            result = await self._request_sender.send_warmup_request(
                request_id,
                messages,
            )
            if not result.successful:
                self._progress_monitor.log_message(
                    f"Warmup session {session.session_id} failed: {result.error}"
                )
        self._progress_monitor.log_message(f"Warmup complete: {num_sessions} sessions")

    # ------------------------------------------------------------------
    # Benchmark dispatch
    # ------------------------------------------------------------------

    async def step(self, time_offset: float) -> float:
        """Dispatch the next request at the QPS-controlled rate.

        Returns:
            Next wakeup time offset, or negative when done.
        """
        # Check duration — stop dispatching new requests
        if time_offset >= self._config.duration:
            if self._pending_tasks:
                await asyncio.wait(
                    self._pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                return 0.0
            return -1.0

        # Find ready session via round-robin
        target_idx = self._global_index % len(self._sessions)
        session = self._sessions[target_idx]

        if session.in_flight:
            # Session busy — sleep briefly, let loop drain queue
            return time_offset + 0.01

        # Dispatch request
        query = self._generate_query()
        request_id = f"s{session.session_id}_r{self._global_index}"
        messages = session.build_messages(query)

        session.in_flight = True
        self._pending_info[request_id] = (session.session_id, query)
        self._progress_monitor.on_request_sent(request_id)
        self._progress_monitor.log_message(
            f"Session {session.session_id} dispatched request {self._global_index}"
        )

        task = asyncio.create_task(
            self._dispatch(request_id, messages),
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._on_task_done)

        self._global_index += 1
        return self._global_index * self._interval

    async def _dispatch(
        self,
        request_id: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Send a single benchmark request."""
        await self._request_sender.send_request(
            request_id,
            messages,
            max_tokens=self._config.output_length,
        )

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Clean up completed tasks and log unexpected errors."""
        self._pending_tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                self._progress_monitor.log_message(f"Dispatch task failed: {exc}")

    def on_request_finished(self, request_id: str, output: str) -> None:
        """Record the response in the session's conversation history."""
        info = self._pending_info.pop(request_id, None)
        if info is None:
            return  # warmup request or already processed
        session_id, query = info
        self._sessions[session_id].record_answer(query, output)
