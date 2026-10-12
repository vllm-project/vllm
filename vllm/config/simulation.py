# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for simulated execution."""

from typing import Any, Literal

from pydantic import Field, field_validator

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

    clock: Literal["real", "virtual", "fast_forward"] = "real"
    """`real`: a simulated step takes its predicted time on the wall clock, so
    the server, benchmark clients and metrics observe realistic latencies.
    The other two need `simulate_forward` and an engine built by
    `vllm.v1.worker.simulated.offline`, whose driver timestamps requests on
    the same clock. `fast_forward`: host work takes its wall time and only the
    waits (device time, idle gaps between arrivals) are skipped.
    `virtual`: the waits are skipped and the engine's own host loop
    (scheduling, output processing) takes no time either, which makes runs
    deterministic; a step's predicted `host_s` is still charged."""

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

    spec_acceptance_rates: list[float] | str | None = None
    """Speculative decoding: the probability that the draft token at each
    position is accepted given that all earlier drafts were (the last value
    applies to later positions), or the path of a step log recorded with
    speculative decoding to estimate them from. Accepted counts are sampled
    per request and step unless the timing provider sets
    `StepCost.num_accepted`."""

    kv_offload_timing: dict[str, float] | str | None = None
    """KV cache offloading (the `OffloadingConnector` with its CPU spec): a
    transfer between device and host memory takes `latency + bytes /
    bandwidth`, per direction, given as `load_gbps` and `store_gbps` (1e9
    bytes/s) with optional `load_latency_us` and `store_latency_us`, or the
    path of a step log recorded with offloading to fit them from. No data
    moves; the connector's scheduler side is the real one."""

    fallback_token_id: int = Field(default=0, ge=0)
    """Token id emitted at every decode step for requests that carry no
    `trace_decode_token_ids`. Such requests end by `max_tokens` (or by this
    token being an EOS/stop token)."""

    @field_validator("spec_acceptance_rates")
    @classmethod
    def _check_rates(cls, rates: list[float] | str | None) -> list[float] | str | None:
        if isinstance(rates, list) and (
            not rates or not all(0.0 <= r <= 1.0 for r in rates)
        ):
            raise ValueError("spec_acceptance_rates must be probabilities in [0, 1].")
        return rates

    @field_validator("kv_offload_timing")
    @classmethod
    def _check_kv_offload_timing(
        cls, timing: dict[str, float] | str | None
    ) -> dict[str, float] | str | None:
        if isinstance(timing, dict):
            required = {"load_gbps", "store_gbps"}
            allowed = required | {"load_latency_us", "store_latency_us"}
            if not required <= timing.keys() <= allowed:
                raise ValueError(
                    f"kv_offload_timing needs {sorted(required)} and accepts "
                    f"{sorted(allowed - required)}; got {sorted(timing)}."
                )
            if any(timing[k] <= 0 for k in required) or any(
                v < 0 for v in timing.values()
            ):
                raise ValueError(
                    "kv_offload_timing bandwidths must be positive and "
                    "latencies non-negative."
                )
        return timing

    @property
    def enabled(self) -> bool:
        return self.simulate_forward or self.record_step_timing

    def compute_hash(self) -> str:
        """Simulation never compiles a graph; only the mode enters the hash."""
        from vllm.config.utils import hash_factors

        return hash_factors({"simulate_forward": self.simulate_forward})
