# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for simulated execution."""

from typing import Any

from pydantic import Field

from vllm.config.utils import config


@config
class SimulationConfig:
    """Simulated execution: the real scheduler, KV-cache manager and frontend
    run unmodified while model execution is replaced by a timing model.

    The model is built on the meta device so vLLM's own KV-cache specs and
    attention metadata are available without weights; KV-cache blocks are
    virtual and sized by `CacheConfig.kv_cache_memory_bytes`; output tokens
    come from `SamplingParams.trace_decode_token_ids` or `fallback_token_id`;
    and each step takes the time a `TimingProvider` predicts for it."""

    simulate_forward: bool = False
    """Replace model forward and sampling with simulated execution."""

    timing: str = "zero"
    """Timing provider for simulated steps: `zero`, `constant`, `table`,
    `replay`, or the fully-qualified name of a class implementing
    `vllm.v1.worker.simulated.timing.TimingProvider`."""

    timing_args: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments passed to the timing provider."""

    log_path: str | None = None
    """If set, append one JSON line per step (the step descriptor, its cost and
    the clock) to this file: predicted costs under `simulate_forward`, measured
    costs under `record_step_timing`. Both use the same format, which the
    `table` and `replay` timing providers read."""

    record_step_timing: bool = False
    """On a real (non-simulated) CUDA deployment, measure every step's device
    time with CUDA events and its host time with the wall clock, and append
    them to `log_path`. This is the calibration source for simulated runs of
    the same model and hardware."""

    fallback_token_id: int = Field(default=0, ge=0)
    """Token id emitted at every decode step for requests that carry no
    `trace_decode_token_ids`. Such requests end by `max_tokens` (or by this
    token being an EOS/stop token)."""

    @property
    def enabled(self) -> bool:
        return self.simulate_forward or self.record_step_timing

    def compute_hash(self) -> str:
        """Simulation never compiles a graph; only the mode enters the hash."""
        from vllm.config.utils import hash_factors

        return hash_factors({"simulate_forward": self.simulate_forward})
