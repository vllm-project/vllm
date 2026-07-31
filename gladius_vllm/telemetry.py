"""Append-only telemetry.jsonl writer, one line per scheduling step.

Owned directly by GladiusScheduler (not the StatLoggerBase path) since it
already has everything it needs -- the scheduler instance, the fresh
SchedulerOutput, and the current PolicyDecision -- with no cross-process
ambiguity. See gladius_vllm.stat_logger for the optional secondary path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gladius_vllm.policy import PolicyDecision
from gladius_vllm.schema import DEFAULT_TELEMETRY_SAMPLE_N, SCHEMA_VERSION, format_iso8601

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


def _count_prefill_decode(scheduler: Any, output: "SchedulerOutput") -> tuple[int, int]:
    """Derive prefill/decode counts from SchedulerOutput + Request state.

    A request counts as "prefill" this step if it's a brand-new admission (in
    scheduled_new_reqs) or a cached continuation still mid-prompt
    (num_computed_tokens < num_prompt_tokens, i.e. a chunked-prefill
    continuation); everything else scheduled this step is "decode".
    """
    new_req_ids = {req.req_id for req in output.scheduled_new_reqs}
    num_prefill = 0
    num_decode = 0
    for req_id in output.num_scheduled_tokens:
        if req_id in new_req_ids:
            num_prefill += 1
            continue
        request = scheduler.requests.get(req_id)
        if request is not None and request.num_computed_tokens < request.num_prompt_tokens:
            num_prefill += 1
        else:
            num_decode += 1
    return num_prefill, num_decode


def _resolve_sample_every_n_steps(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    env_value = os.environ.get("GLADIUS_TELEMETRY_SAMPLE_N")
    return int(env_value) if env_value else DEFAULT_TELEMETRY_SAMPLE_N


class TelemetryWriter:
    """Appends one JSON line per scheduling step to telemetry.jsonl."""

    def __init__(
        self,
        path: Path | None,
        engine_id: str,
        model_id: str,
        sample_every_n_steps: int | None = None,
    ) -> None:
        self._path = path
        self._engine_id = engine_id
        self._model_id = model_id
        self._sample_every_n_steps = _resolve_sample_every_n_steps(sample_every_n_steps)
        self._step = 0
        self._file = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "a")

    def record(
        self,
        scheduler: Any,
        output: "SchedulerOutput",
        decision: PolicyDecision,
    ) -> None:
        self._step += 1
        if self._file is None:
            return
        if self._step % self._sample_every_n_steps != 0:
            return

        num_prefill, num_decode = _count_prefill_decode(scheduler, output)
        stats = scheduler.make_stats()

        clamped_max_num_seqs = decision.max_num_seqs > scheduler.startup_max_num_seqs
        clamped_max_num_batched_tokens = (
            decision.max_num_batched_tokens > scheduler.startup_max_num_batched_tokens
        )

        record = {
            "schema_version": SCHEMA_VERSION,
            "generation": decision.generation,
            "policy_id": decision.policy_id,
            "model_id": self._model_id,
            "engine_id": self._engine_id,
            "created_at": format_iso8601(),
            "expires_at": None,
            "step": self._step,
            "num_running_reqs": stats.num_running_reqs if stats else len(scheduler.running),
            "num_waiting_reqs": stats.num_waiting_reqs if stats else len(scheduler.waiting),
            "num_skipped_waiting_reqs": (
                stats.num_skipped_waiting_reqs if stats else len(scheduler.skipped_waiting)
            ),
            "num_scheduled_reqs": len(output.num_scheduled_tokens),
            "num_scheduled_tokens": output.total_num_scheduled_tokens,
            "num_prefill_reqs": num_prefill,
            "num_decode_reqs": num_decode,
            "kv_cache_usage": stats.kv_cache_usage if stats else None,
            "policy_status": decision.status,
            "policy_source": decision.source,
            "clamped": {
                "max_num_seqs": clamped_max_num_seqs,
                "max_num_batched_tokens": clamped_max_num_batched_tokens,
            },
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
