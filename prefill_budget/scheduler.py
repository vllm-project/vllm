"""vLLM scheduler that sizes each prefill chunk from a step-time budget.

Stock vLLM caps a chunk with a fixed token count. This subclass keeps that
loop and only replaces the cap: `ChunkPolicy` fits a quadratic cost from
completed steps on this GPU, then picks the largest chunk that stays inside
the time budget.

The stock scheduler has one `long_prefill_token_threshold` for the whole
step, so the cap hits every request scheduled in that step. Run with
`max_num_seqs=1` until the cap is per-request. Async scheduling must be off:
the next `schedule()` would otherwise run while the GPU step is still in
flight, and the timer would not see the step time.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler

from prefill_budget.cost_model import ChunkPolicy

logger = init_logger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


class TimeBudgetScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling:
            raise RuntimeError(
                "TimeBudgetScheduler requires async_scheduling=False. "
                "Async scheduling starts the next step before the GPU "
                "finishes, so the measured time would not be the step time."
            )
        self._policy = ChunkPolicy(
            mode=os.environ.get("VLLM_PREFILL_MODE", "budget").strip().lower(),
            budget_ms=_env_float("VLLM_PREFILL_TIME_BUDGET_MS", 30.0),
            fixed_chunk=_env_int("VLLM_PREFILL_FIXED_CHUNK", 1024),
            min_chunk=_env_int("VLLM_PREFILL_MIN_CHUNK", 128),
            max_chunk=_env_int("VLLM_PREFILL_MAX_CHUNK", 2048),
            warmup_steps=_env_int("VLLM_PREFILL_WARMUP_STEPS", 8),
            block_size=int(self.cache_config.block_size),
        )
        self._pending: tuple[int, int, str] | None = None
        self._pending_t: float | None = None
        log_path = os.environ.get("VLLM_PREFILL_BUDGET_LOG", "prefill_budget_log.jsonl")
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text("", encoding="utf-8")
        logger.info(
            "TimeBudgetScheduler mode=%s budget_ms=%s chunk=[%s,%s] block_size=%s log=%s",
            self._policy.mode,
            self._policy.budget_ms,
            self._policy.min_chunk,
            self._policy.max_chunk,
            self._policy.block_size,
            self._log_path,
        )

    def schedule(self, throttle_prefills: bool = False):
        self._finish_pending_step()
        nxt = self._next_prefill()
        cap: int | None = None
        phase = "decode"
        if nxt is not None:
            _, prefix, remaining = nxt
            cap, phase = self._policy.cap(prefix, remaining)
        previous = self.scheduler_config.long_prefill_token_threshold
        if cap is not None and cap > 0:
            self.scheduler_config.long_prefill_token_threshold = cap
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self.scheduler_config.long_prefill_token_threshold = previous
        self._arm_pending(output, nxt, phase)
        return output

    def _finish_pending_step(self) -> None:
        if self._pending is None or self._pending_t is None:
            return
        elapsed_ms = (time.perf_counter() - self._pending_t) * 1000.0
        q, c, phase = self._pending
        self._pending = None
        self._pending_t = None
        used = self._policy.observe(q, c, elapsed_ms)
        predicted = None
        if self._policy.fit is not None:
            predicted = round(self._policy.fit.predict_ms(q, c), 3)
        record = {
            "q": q,
            "c": c,
            "elapsed_ms": round(elapsed_ms, 3),
            "phase": phase,
            "usable": used,
            "target_ms": self._policy.budget_ms,
            "predicted_ms": predicted,
        }
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info(
            "prefill-budget c=%s q=%s step_ms=%.1f phase=%s usable=%s",
            c,
            q,
            elapsed_ms,
            phase,
            int(used),
        )

    def _next_prefill(self) -> tuple[str, int, int] | None:
        """Return (request_id, prefix_len, remaining_prompt_tokens)."""
        for req in self.running:
            remaining = req.num_prompt_tokens - req.num_computed_tokens
            if remaining > 0:
                return req.request_id, req.num_computed_tokens, remaining
        if len(self.waiting) > 0:
            req = self.waiting.peek_request()
            remaining = req.num_prompt_tokens - req.num_computed_tokens
            if remaining > 0:
                return req.request_id, req.num_computed_tokens, remaining
        return None

    def _arm_pending(self, output, nxt, phase: str) -> None:
        if nxt is None:
            return
        req_id, prefix, _remaining = nxt
        q = output.num_scheduled_tokens.get(req_id, 0)
        if q <= 1:
            return
        self._pending = (q, prefix, phase)
        self._pending_t = time.perf_counter()
