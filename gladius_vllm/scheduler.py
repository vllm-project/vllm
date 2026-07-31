"""GladiusScheduler: a vLLM V1 Scheduler that hot-reloads admission ceilings
from an externally-published policy_snapshot.json.

Wired in via vLLM's existing `--scheduler-cls` plugin mechanism (no upstream
vllm/ changes needed):

    vllm serve ... --scheduler-cls gladius_vllm.scheduler.GladiusScheduler

Configuration is via environment variables (not VllmConfig/EngineArgs
plumbing, to keep this a pure additive plugin):

    GLADIUS_POLICY_DIR                directory holding policy_snapshot.json
                                       and telemetry.jsonl. If unset, this
                                       scheduler behaves exactly like a
                                       vanilla Scheduler.
    GLADIUS_ENGINE_ID                 stable engine id across restarts.
    GLADIUS_POLICY_POLL_INTERVAL_MS   rate limit for policy file re-stat.
    GLADIUS_TELEMETRY_SAMPLE_N        write 1-in-N telemetry lines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.v1.core.sched.scheduler import Scheduler

from gladius_vllm.policy import PolicyLoader
from gladius_vllm.registry import register_scheduler
from gladius_vllm.schema import (
    DEFAULT_POLICY_POLL_INTERVAL_MS,
    resolve_engine_id,
    resolve_model_id,
)
from gladius_vllm.telemetry import TelemetryWriter

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

POLICY_SNAPSHOT_FILENAME = "policy_snapshot.json"
TELEMETRY_FILENAME = "telemetry.jsonl"


class GladiusScheduler(Scheduler):
    """Thin subclass: only __init__ and schedule() are overridden.

    The safe-clamp invariant `effective = min(policy_requested, startup)` is
    enforced fresh every step, so this scheduler degrades to byte-for-byte
    vanilla `Scheduler` behavior whenever no policy is configured/active
    (min(x, x) == x is a no-op), and can never admit more than the engine was
    started with -- max_num_seqs/max_num_batched_tokens are baked into CUDA
    graph capture sizes and torch.compile shapes at startup, so raising them
    post-hoc is unsafe; this scheduler only ever clamps down.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.engine_id = resolve_engine_id(self.vllm_config)
        self.model_id = resolve_model_id(self.vllm_config)

        # Captured once, after super().__init__ has set these -- these are
        # the values already baked into CUDA graph capture / compiled kernel
        # shapes, and are never mutated by this class.
        self.startup_max_num_seqs = self.max_num_running_reqs
        self.startup_max_num_batched_tokens = self.max_num_scheduled_tokens

        policy_dir_env = os.environ.get("GLADIUS_POLICY_DIR")
        policy_dir = Path(policy_dir_env) if policy_dir_env else None
        poll_interval_env = os.environ.get("GLADIUS_POLICY_POLL_INTERVAL_MS")
        poll_interval_ms = (
            int(poll_interval_env) if poll_interval_env else DEFAULT_POLICY_POLL_INTERVAL_MS
        )

        self._policy_loader = PolicyLoader(
            snapshot_path=policy_dir / POLICY_SNAPSHOT_FILENAME if policy_dir else None,
            engine_id=self.engine_id,
            model_id=self.model_id,
            startup_max_num_seqs=self.startup_max_num_seqs,
            startup_max_num_batched_tokens=self.startup_max_num_batched_tokens,
            poll_interval_ms=poll_interval_ms,
        )
        self._telemetry_writer = TelemetryWriter(
            path=policy_dir / TELEMETRY_FILENAME if policy_dir else None,
            engine_id=self.engine_id,
            model_id=self.model_id,
        )

        register_scheduler(self)

    def schedule(self) -> "SchedulerOutput":
        decision = self._policy_loader.poll()
        target_max_num_seqs = min(decision.max_num_seqs, self.startup_max_num_seqs)
        # Never shrink below the number of requests already admitted: the
        # base Scheduler enforces `len(self.running) <= max_num_running_reqs`
        # as a forward-looking admission check only -- it has no preemption
        # path to evict already-running requests down to a newly-lowered
        # ceiling, and asserts the invariant unconditionally. A policy asking
        # for fewer than what's already running takes effect gradually, as
        # attrition (requests finishing) brings the running count back down.
        self.max_num_running_reqs = max(target_max_num_seqs, len(self.running))
        self.max_num_scheduled_tokens = min(
            decision.max_num_batched_tokens, self.startup_max_num_batched_tokens
        )

        output = super().schedule()

        self._telemetry_writer.record(self, output, decision)
        return output

    def shutdown(self) -> None:
        self._telemetry_writer.close()
        super().shutdown()
