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
    """If set, append one JSON line per simulated step (the step descriptor,
    the predicted cost and the simulated clock) to this file."""

    fallback_token_id: int = Field(default=0, ge=0)
    """Token id emitted at every decode step for requests that carry no
    `trace_decode_token_ids`. Such requests end by `max_tokens` (or by this
    token being an EOS/stop token)."""

    @property
    def enabled(self) -> bool:
        return self.simulate_forward

    def compute_hash(self) -> str:
        """Simulation never compiles a graph; only the mode enters the hash."""
        from vllm.config.utils import hash_factors

        return hash_factors({"simulate_forward": self.simulate_forward})
