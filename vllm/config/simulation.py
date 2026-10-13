# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for simulated execution."""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator

from vllm.config.utils import config, hash_factors

if TYPE_CHECKING:
    from vllm.config.vllm import VllmConfig

# Simulated execution replaces these connectors' worker side.
_NIXL_CONNECTORS = ("NixlConnector", "NixlPushConnector")
_SIMULATED_NIXL_MODULE = "vllm.v1.worker.simulated.kv_nixl"


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
    """Timing provider for simulated steps: `zero`, `linear`, `table`,
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

    kv_transfer_timing: dict[str, float] | str | None = None
    """KV connector transfers, which the connector's (real) scheduler side
    issues and which move no data: each takes `latency + bytes / bandwidth`.
    Loads fill this engine's KV cache: from the CPU cache for the
    `OffloadingConnector`'s CPU spec, from a prefill instance for the
    `NixlConnector`. Stores (offloading only) go to the CPU cache. Given as
    `load_gbps` and `store_gbps` (1e9 bytes/s) with optional
    `load_latency_us` and `store_latency_us`, or the path of a step log
    recorded with the connector to fit them from. A NIXL decode instance also
    holds its engine thread while posting reads: `post_us_per_read` plus
    `post_us_per_run` per run of blocks consecutive on both instances, also
    fitted from a step log."""

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

    @field_validator("kv_transfer_timing")
    @classmethod
    def _check_kv_transfer_timing(
        cls, timing: dict[str, float] | str | None
    ) -> dict[str, float] | str | None:
        if isinstance(timing, dict):
            allowed = {
                "load_gbps",
                "store_gbps",
                "load_latency_us",
                "store_latency_us",
                "post_us_per_read",
                "post_us_per_run",
            }
            if "load_gbps" not in timing or not timing.keys() <= allowed:
                raise ValueError(
                    f"kv_transfer_timing needs load_gbps and accepts "
                    f"{sorted(allowed)}; got {sorted(timing)}."
                )
            bandwidths = [timing[k] for k in ("load_gbps", "store_gbps") if k in timing]
            if any(b <= 0 for b in bandwidths) or any(v < 0 for v in timing.values()):
                raise ValueError(
                    "kv_transfer_timing bandwidths must be positive and "
                    "latencies non-negative."
                )
        return timing

    def compute_hash(self) -> str:
        """Simulation never compiles a graph; only the mode enters the hash."""
        return hash_factors({"simulate_forward": self.simulate_forward})

    def verify_and_update_config(self, vllm_config: "VllmConfig") -> None:
        """Check `vllm_config` against simulated execution, and set what it
        implies: weights on the meta device and the simulated NIXL worker."""
        if self.record_step_timing:
            if self.simulate_forward:
                raise ValueError(
                    "record_step_timing records a real deployment; it cannot be "
                    "combined with simulate_forward."
                )
            if self.log_path is None:
                raise ValueError("record_step_timing needs log_path.")
        if not self.simulate_forward:
            return

        import vllm.envs as envs
        from vllm.platforms import current_platform
        from vllm.triton_utils import HAS_TRITON

        if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
            raise ValueError("Simulated forward needs a CUDA-alike or CPU platform.")
        if not HAS_TRITON or envs.VLLM_USE_V2_MODEL_RUNNER is False:
            raise ValueError(
                "Simulated forward runs on Model Runner V2, which needs Triton "
                "and VLLM_USE_V2_MODEL_RUNNER unset or 1."
            )
        if vllm_config.lora_config is not None:
            raise ValueError("Simulated forward does not support LoRA.")
        if vllm_config.weight_transfer_config is not None:
            raise ValueError("Simulated forward does not support weight transfer.")
        if vllm_config.model_config.runner_type != "generate":
            raise ValueError("Simulated forward supports generation runners only.")
        parallel = vllm_config.parallel_config
        if parallel.prefill_context_parallel_size > 1:
            raise ValueError(
                "Simulated forward does not support prefill context parallelism."
            )
        num_ranks = (
            parallel.tensor_parallel_size
            * parallel.pipeline_parallel_size
            * parallel.data_parallel_size
        )
        if self.clock != "real" and num_ranks > 1:
            raise ValueError(
                f"The {self.clock} simulation clock supports one rank, not "
                f"{num_ranks}; use the real clock."
            )

        kv_transfer = vllm_config.kv_transfer_config
        nixl = kv_transfer is not None and kv_transfer.kv_connector in _NIXL_CONNECTORS
        if kv_transfer is not None and not nixl:
            spec = kv_transfer.kv_connector_extra_config.get(
                "spec_name", "CPUOffloadingSpec"
            )
            if (kv_transfer.kv_connector, spec) != (
                "OffloadingConnector",
                "CPUOffloadingSpec",
            ):
                raise ValueError(
                    "Simulated forward supports KV transfer by the "
                    "OffloadingConnector's CPU offloading, the NixlConnector "
                    f"and the NixlPushConnector, not {kv_transfer.kv_connector}."
                )
        if nixl:
            assert kv_transfer is not None
            if kv_transfer.kv_connector == "NixlPushConnector":
                if kv_transfer.kv_role not in ("kv_producer", "kv_consumer"):
                    raise ValueError(
                        "Simulated push-mode disaggregated prefill needs "
                        f"kv_role kv_producer or kv_consumer, not "
                        f"{kv_transfer.kv_role}."
                    )
                if self.clock != "real":
                    raise ValueError(
                        "Simulated push-mode disaggregated prefill runs served "
                        "instances, on the real clock."
                    )
            if self.clock == "fast_forward":
                # Served instances run on the real clock, in-process ones
                # (vllm.v1.worker.simulated.disaggregated) on virtual clocks.
                raise ValueError(
                    "Simulated disaggregated prefill does not support the "
                    "fast_forward clock."
                )
            module_path = kv_transfer.kv_connector_module_path
            if module_path not in (None, _SIMULATED_NIXL_MODULE):
                raise ValueError(
                    "Simulated forward replaces the NIXL connector's worker "
                    f"side; kv_connector_module_path {module_path} cannot be set."
                )
            kv_transfer.kv_connector_module_path = _SIMULATED_NIXL_MODULE

        load_configs = [vllm_config.load_config]
        speculative = vllm_config.speculative_config
        if speculative is not None and speculative.draft_load_config is not None:
            load_configs.append(speculative.draft_load_config)
        for load_config in load_configs:
            load_config.load_format = "dummy"
            load_config.device = "meta"
