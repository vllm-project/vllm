# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch layer boundaries for the experimental two-stage prefill pipeline."""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from math import isfinite
from statistics import median
from time import monotonic
from typing import Any

HEADER_KEY = "dynamic_pp_header"
COST_HISTORY_DRIFT_MULTIPLIER = 2
TIMING_FALLBACK_CONFIRMATION_MULTIPLIER = 2


def select_pp_moe_layer_names(
    layer_names: Sequence[str], layer_prefix: str, start: int, end: int
) -> list[str]:
    """Order fast-MoE dispatch by this batch's executed logical layers."""
    prefix = f"{layer_prefix}."
    by_layer: dict[int, list[str]] = {}
    for name in layer_names:
        if not name.startswith(prefix):
            continue
        index, separator, _ = name[len(prefix) :].partition(".")
        if separator and index.isdecimal():
            by_layer.setdefault(int(index), []).append(name)
    missing = [index for index in range(start, end) if index not in by_layer]
    if missing:
        raise ValueError(
            f"Missing dynamic PP MoE dispatch entries for layers {missing}"
        )
    return [name for index in range(start, end) for name in by_layer[index]]


@dataclass(frozen=True)
class DynamicPPConfig:
    enabled: bool = False
    redundant_layers: int = 2
    initial_offset: int = 0
    adaptive: bool = True
    low_watermark: int = 0
    high_watermark: int = 2
    cooldown_steps: int = 8
    max_step: int = 1
    log_interval_seconds: int = 10
    balance_tolerance_percent: int = 3
    feedback_timeout_seconds: int = 30

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DynamicPPConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("dynamic_pp_config must be an object")
        unknown = value.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown dynamic_pp_config fields: {sorted(unknown)}")
        result = cls(**value)
        for item in fields(cls):
            expected = bool if item.name in ("enabled", "adaptive") else int
            if type(getattr(result, item.name)) is not expected:
                raise ValueError(
                    f"dynamic_pp_config.{item.name} must be {expected.__name__}"
                )
        if (
            result.redundant_layers < 1
            or result.max_step < 1
            or result.cooldown_steps < 1
        ):
            raise ValueError(
                "Dynamic PP redundancy, max_step, and cooldown_steps must be positive"
            )
        if not 0 <= result.low_watermark < result.high_watermark:
            raise ValueError("Dynamic PP requires 0 <= low_watermark < high_watermark")
        if abs(result.initial_offset) > result.redundant_layers:
            raise ValueError(
                "Dynamic PP initial_offset exceeds the redundant layer range"
            )
        if result.log_interval_seconds < 0:
            raise ValueError(
                "Dynamic PP log interval must be non-negative (0 disables logging)"
            )
        if (
            not 0 <= result.balance_tolerance_percent < 100
            or result.feedback_timeout_seconds < 1
        ):
            raise ValueError(
                "Dynamic PP needs tolerance in [0, 100) and a positive feedback timeout"
            )
        return result

    def validate_runtime(self, vllm_config: Any, buffer_size: int) -> None:
        if buffer_size and not vllm_config.use_v2_model_runner:
            raise ValueError("pp_send_buffer_size requires Model Runner V2")
        if not self.enabled:
            return
        parallel = vllm_config.parallel_config
        model = vllm_config.model_config
        if not vllm_config.use_v2_model_runner or parallel.pipeline_parallel_size != 2:
            raise ValueError("Dynamic PP currently requires Model Runner V2 and PP=2")
        if not model.enforce_eager:
            raise ValueError("Dynamic PP currently requires eager execution")
        if buffer_size < self.high_watermark:
            raise ValueError(
                "pp_send_buffer_size must be at least the dynamic PP high_watermark"
            )
        if (
            parallel.data_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.use_sequence_parallel_moe
        ):
            raise ValueError(
                "Dynamic PP does not yet support DP, EP, or sequence parallel MoE"
            )
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("Dynamic PP does not yet support context parallelism")
        if (
            vllm_config.speculative_config is not None
            or vllm_config.kv_transfer_config is not None
        ):
            raise ValueError(
                "Dynamic PP does not yet support speculation or KV connectors"
            )
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Dynamic PP prefix caching requires KV replicas")
        if getattr(vllm_config, "lora_config", None) is not None:
            raise ValueError("Dynamic PP does not yet support LoRA")
        if parallel.use_ubatching:
            raise ValueError("Dynamic PP does not yet support microbatching")
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError(
                "Dynamic PP currently requires chunked prefill to be disabled"
            )
        hf_config = model.hf_config
        if getattr(hf_config, "use_index_cache", False) or "shared" in (
            getattr(hf_config, "indexer_types", ()) or ()
        ):
            raise ValueError("Dynamic PP does not yet support cross-layer index reuse")


@dataclass(frozen=True)
class PPBoundary:
    batch_id: int
    boundary: int
    version: int

    def to_wire(self) -> tuple[int, int, int]:
        return self.batch_id, self.boundary, self.version

    @classmethod
    def from_wire(cls, value: Any) -> "PPBoundary":
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 3
            or any(type(item) is not int for item in value)
        ):
            raise ValueError("Invalid dynamic PP batch header")
        if value[0] < 0 or value[2] < 0:
            raise ValueError("Dynamic PP batch and version must be non-negative")
        return cls(*value)


@dataclass(frozen=True)
class PPBalanceSample:
    batch_id: int
    boundary: int
    version: int
    num_tokens: int
    num_reqs: int
    pp0_ms: float
    pp1_ms: float


class PPBoundaryController:
    """Move one PP2 boundary within its preloaded layer range."""

    def __init__(
        self,
        config: DynamicPPConfig,
        base_boundary: int,
        num_layers: int,
        minimum_boundary: int = 1,
    ):
        if not 0 < base_boundary < num_layers:
            raise ValueError("Dynamic PP needs two non-empty base stages")
        self.config = config
        self.base_boundary = base_boundary
        self.num_layers = num_layers
        self.minimum = max(1, minimum_boundary, base_boundary - config.redundant_layers)
        self.maximum = min(num_layers - 1, base_boundary + config.redundant_layers)
        self.boundary = base_boundary + config.initial_offset
        if not self.minimum <= self.boundary <= self.maximum:
            raise ValueError(
                "Dynamic PP initial boundary violates stage or hash-routing limits"
            )
        self.version = 0
        self._samples: deque[PPBalanceSample] = deque(maxlen=config.cooldown_steps)
        self._last_sample_batch = -1
        self._samples_since_change = 0
        self._costs: dict[tuple[int, int, int], tuple[float, float]] = {}
        self.feedback_stats: dict[str, Any] = {"decision": "awaiting_completed_samples"}
        self._last_batch_id = -1
        self._last_received_version = -1
        self._last_received_boundary: int | None = None

    def loaded_range(self, rank: int) -> tuple[int, int]:
        if rank == 0:
            return 0, self.maximum
        if rank == 1:
            return self.minimum, self.num_layers
        raise ValueError("Dynamic PP currently supports only ranks 0 and 1")

    def execution_range(self, rank: int, header: PPBoundary) -> tuple[int, int]:
        if not self.minimum <= header.boundary <= self.maximum:
            raise ValueError("Dynamic PP batch boundary is outside the preloaded range")
        if rank == 0:
            return 0, header.boundary
        if rank == 1:
            return header.boundary, self.num_layers
        raise ValueError("Dynamic PP currently supports only ranks 0 and 1")

    def choose(
        self,
        batch_id: int,
        queue_depth: int,
        samples: Sequence[PPBalanceSample] = (),
        feedback_fresh: bool = True,
    ) -> PPBoundary:
        if batch_id <= self._last_batch_id or queue_depth < 0:
            raise ValueError(
                "Dynamic PP needs increasing batch ids and non-negative queue depth"
            )
        self._last_batch_id = batch_id
        if self.feedback_stats.get("feedback_version") != self.version:
            self.feedback_stats.clear()
        added = False
        for sample in samples:
            if sample.batch_id >= batch_id:
                raise ValueError("Dynamic PP feedback cannot refer to a future batch")
            if not all(
                isfinite(value) and value > 0
                for value in (sample.pp0_ms, sample.pp1_ms)
            ):
                raise ValueError("Dynamic PP stage times must be finite and positive")
            if sample.batch_id <= self._last_sample_batch:
                continue
            self._last_sample_batch = sample.batch_id
            # Queued old work must not drive the new boundary again.
            if (sample.boundary, sample.version) != (self.boundary, self.version):
                continue
            if self._samples and (sample.num_tokens, sample.num_reqs) != (
                self._samples[-1].num_tokens,
                self._samples[-1].num_reqs,
            ):
                self._samples.clear()
                self._samples_since_change = 0
                self.feedback_stats.clear()
            self._samples.append(sample)
            self._samples_since_change += 1
            added = True
        self.feedback_stats.update(
            feedback_version=self.version,
            matched_samples=len(self._samples),
            completed_samples_since_change=self._samples_since_change,
            ready_queue_depth=queue_depth,
        )
        if not self.config.adaptive:
            self.feedback_stats["decision"] = "fixed_boundary"
        elif not feedback_fresh:
            self.feedback_stats["decision"] = "stale_feedback_hold"
            self._samples.clear()
            self._samples_since_change = 0
            self.feedback_stats["matched_samples"] = 0
            self.feedback_stats["completed_samples_since_change"] = 0
        elif len(self._samples) < self.config.cooldown_steps:
            self.feedback_stats["decision"] = "awaiting_completed_samples"
        elif added:
            self._rebalance(queue_depth)
        return PPBoundary(batch_id, self.boundary, self.version)

    def _rebalance(self, queue_depth: int) -> None:
        first = self._samples[-1]
        pp0_ms = median(sample.pp0_ms for sample in self._samples)
        pp1_ms = median(sample.pp1_ms for sample in self._samples)
        cost = max(pp0_ms, pp1_ms)
        imbalance = (pp0_ms - pp1_ms) / cost
        tolerance = self.config.balance_tolerance_percent / 100
        key = (self.boundary, first.num_tokens, first.num_reqs)
        reference = self._costs.get(key)
        if reference is not None and any(
            abs(current - old) / old > COST_HISTORY_DRIFT_MULTIPLIER * tolerance
            for current, old in zip((pp0_ms, pp1_ms), reference)
        ):
            # An old neighboring cut is no longer comparable after device
            # load changes. Keep an anchor rather than refreshing it every
            # batch, so gradual drift is detected too.
            self._costs = {k: v for k, v in self._costs.items() if k[1:] != key[1:]}
        self._costs.setdefault(key, (pp0_ms, pp1_ms))
        self.feedback_stats.update(
            pp0_device_ms=round(pp0_ms, 3),
            pp1_device_ms=round(pp1_ms, 3),
            imbalance_percent=round(imbalance * 100, 3),
            timing_batch_id=first.batch_id,
            timing_num_tokens=first.num_tokens,
            timing_num_reqs=first.num_reqs,
        )
        if abs(imbalance) <= tolerance:
            self.feedback_stats["decision"] = "balanced_hold"
            return
        # Watermarks describe completed production minus credited consumption.
        # Stage times gate both directions: delayed credits or a host running
        # ahead must never turn a full send ring into unconditional growth.
        direction = 1 if imbalance < 0 else -1
        reason = "stage_time_imbalance"
        watermark_confirmed = False
        if direction == 1 and queue_depth >= self.config.high_watermark:
            reason = "high_ready_watermark"
            watermark_confirmed = True
        elif direction == -1 and queue_depth <= self.config.low_watermark:
            reason = "low_ready_watermark"
            watermark_confirmed = True
        required = self.config.cooldown_steps * (
            1 if watermark_confirmed else TIMING_FALLBACK_CONFIRMATION_MULTIPLIER
        )
        self.feedback_stats["required_completed_samples"] = required
        if self._samples_since_change < required:
            self.feedback_stats["decision"] = (
                "awaiting_watermark_or_timing_confirmation"
            )
            return
        boundary = min(
            self.maximum,
            max(self.minimum, self.boundary + direction * self.config.max_step),
        )
        if boundary == self.boundary:
            self.feedback_stats["decision"] = "boundary_limit_hold"
            return
        previous = self._costs.get((boundary, first.num_tokens, first.num_reqs))
        if previous is not None and max(previous) >= cost * (1 - tolerance):
            self.feedback_stats["decision"] = "adjacent_boundary_not_faster"
            return
        self.feedback_stats["decision"] = reason
        self.boundary = boundary
        self.version += 1
        self._samples.clear()
        self._samples_since_change = 0

    def accept(self, value: Any, expected_batch_id: int) -> PPBoundary:
        header = PPBoundary.from_wire(value)
        self.execution_range(1, header)
        if (
            header.batch_id != expected_batch_id
            or header.batch_id <= self._last_batch_id
        ):
            raise ValueError("Out-of-order dynamic PP batch")
        if header.version < self._last_received_version:
            raise ValueError("Dynamic PP boundary version moved backwards")
        if (
            header.version == self._last_received_version
            and header.boundary != self._last_received_boundary
        ):
            raise ValueError(
                "Dynamic PP changed a boundary without changing its version"
            )
        self._last_batch_id = header.batch_id
        self._last_received_version = header.version
        self._last_received_boundary = header.boundary
        self.boundary = header.boundary
        self.version = header.version
        return header


class PPStrategyReporter:
    """Summarize submitted per-batch strategies without device synchronization."""

    def __init__(self, controller: PPBoundaryController, pp_rank: int):
        self.controller = controller
        self.pp_rank = pp_rank
        self._last_log_time: float | None = None
        self._window_start: float | None = None
        self._last_boundary: int | None = None
        self._last_change_time: float | None = None
        self._boundary_counts: dict[int, int] = {}
        self._changes = 0
        self._pending_min: int | None = None
        self._pending_max: int | None = None

    def observe(
        self, header: PPBoundary, pending_sends: int | None, now: float | None = None
    ) -> dict[str, Any] | None:
        interval = self.controller.config.log_interval_seconds
        if not interval:
            return None
        now = monotonic() if now is None else now
        if self._window_start is None:
            self._window_start = now
        changed = (
            self._last_boundary is not None and header.boundary != self._last_boundary
        )
        if header.boundary != self._last_boundary:
            self._changes += int(self._last_boundary is not None)
            self._last_change_time = now
            self._last_boundary = header.boundary
        self._boundary_counts[header.boundary] = (
            self._boundary_counts.get(header.boundary, 0) + 1
        )
        if pending_sends is not None:
            self._pending_min = (
                pending_sends
                if self._pending_min is None
                else min(self._pending_min, pending_sends)
            )
            self._pending_max = (
                pending_sends
                if self._pending_max is None
                else max(self._pending_max, pending_sends)
            )
        first = self._last_log_time is None
        periodic = self._last_log_time is None or now - self._last_log_time >= interval
        if not periodic and not changed:
            return None
        assert self._last_change_time is not None
        report = {
            "trigger": "initial"
            if first
            else "periodic"
            if periodic
            else "boundary_change",
            "boundary_changed": changed,
            "periodic_snapshot": periodic,
            "observation": "host_forward_submit",
            "pp_rank": self.pp_rank,
            "tp_rank": 0,
            "batch_id": header.batch_id,
            "boundary": header.boundary,
            "version": header.version,
            "execution_range": self.controller.execution_range(self.pp_rank, header),
            "pp0_layers": header.boundary,
            "pp1_layers": self.controller.num_layers - header.boundary,
            "boundary_limits": [self.controller.minimum, self.controller.maximum],
            "at_boundary_limit": header.boundary
            in (self.controller.minimum, self.controller.maximum),
            "window_seconds": round(now - self._window_start, 3),
            "window_batches": sum(self._boundary_counts.values()),
            "window_boundary_counts": dict(sorted(self._boundary_counts.items())),
            "window_boundary_changes": self._changes,
            "unchanged_seconds": round(now - self._last_change_time, 3),
            "watermark_kind": "device_ready_minus_consumer_credit",
            "pending_send_count": pending_sends,
            "window_pending_send_min": self._pending_min,
            "window_pending_send_max": self._pending_max,
            "balance_feedback": dict(self.controller.feedback_stats)
            if self.pp_rank == 0
            else None,
        }
        if periodic:
            self._last_log_time = now
            self._window_start = now
            self._boundary_counts = {}
            self._changes = 0
            self._pending_min = self._pending_max = None
        return report


def validate_fresh_prefill(scheduler_output: Any) -> None:
    """Fail before communication rather than reading unsynchronized history."""
    cached = scheduler_output.scheduled_cached_reqs
    if cached.req_ids:
        raise ValueError(
            "Dynamic PP requires complete fresh prefills; history needs KV replicas"
        )
    for request in scheduler_output.scheduled_new_reqs:
        if request.num_computed_tokens:
            raise ValueError("Dynamic PP cannot yet consume cached prefix state")
        if request.sampling_params is None or request.sampling_params.max_tokens != 1:
            raise ValueError(
                "Dynamic PP currently supports prefill with max_tokens=1 only"
            )
        scheduled = scheduler_output.num_scheduled_tokens.get(request.req_id, 0)
        if request.prompt_token_ids is None or scheduled != len(
            request.prompt_token_ids
        ):
            raise ValueError(
                "Dynamic PP currently requires a complete prompt in one batch"
            )
