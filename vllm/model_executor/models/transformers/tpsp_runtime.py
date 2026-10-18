# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers TPSP backends, contexts, and profiling utilities."""

from __future__ import annotations

import inspect
import logging
import math
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import distributed_c10d as c10d

_LOG = logging.getLogger(__name__)
_TRIALS = 3
_COARSE_MICROCHUNK_STEP_PER_RANK = 256
_FINE_MICROCHUNK_STEP_PER_RANK = 64
_COARSE_COMPUTE_ENGINE_STEP = 6
_FINE_COMPUTE_ENGINE_STEP = 1
_FINALIST_COUNT = 3
_FINE_NEIGHBORHOOD_RADIUS = 4
_COARSE_TIMINGS = 1
_FINE_TIMINGS = 3
_COARSE_THRESHOLD_STEP = 512
_FINE_THRESHOLD_STEP = 128


class _TPSPProgress:
    def __init__(
        self,
        group_name: str,
        stage: str,
        total: int,
        rank: int,
        *,
        started_at: float | None = None,
    ) -> None:
        self.group_name = group_name
        self.stage = stage
        self.total = max(total, 1)
        self.rank = rank
        self.started_at = time.perf_counter() if started_at is None else started_at
        self.percent = 0
        self.detail = f"0/{self.total}"
        self.last_logged_bucket = 0
        self._log()

    def update(self, completed: int, *, percent_limit: int = 100) -> None:
        self.detail = f"{completed}/{self.total}"
        self.percent = min(
            percent_limit, int(100 * min(completed, self.total) / self.total)
        )
        bucket = self.percent // 10
        if bucket > self.last_logged_bucket and self.percent < 100:
            self.last_logged_bucket = bucket
            self._log()

    def update_percent(self, percent: int, *, detail: str) -> None:
        self.detail = detail
        self.percent = min(99, max(self.percent, percent))
        bucket = self.percent // 10
        if bucket > self.last_logged_bucket:
            self.last_logged_bucket = bucket
            self._log()

    def finish(
        self, status: str, *, complete: bool = True, detail: str | None = None
    ) -> None:
        if complete:
            self.percent = 100
        if detail is not None:
            self.detail = detail
        self._log(status=status)

    def _log(self, status: str = "running") -> None:
        if self.rank != 0:
            return
        width = 20
        filled = width * self.percent // 100
        bar = "#" * filled + "-" * (width - filled)
        _LOG.info(
            "TPSP profile group=%s stage=%s [%s] %3d%% %s elapsed=%.1fs status=%s",
            self.group_name,
            self.stage,
            bar,
            self.percent,
            self.detail,
            time.perf_counter() - self.started_at,
            status,
        )


# TPSP v1 supports only standard LayerNorm and RMSNorm.
TPSPNormType = Literal["layer_norm", "rms_norm"]


class TPSPBackend(ABC):
    tpsp_max_microchunk_tokens_per_rank: int | None = None
    supported_norm_types: frozenset[TPSPNormType] = frozenset()

    def __init__(self, group_name: str, device: torch.device) -> None:
        self.group_name = group_name
        self.device = device
        self._closed = False

    @abstractmethod
    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> bool:
        """Initialize backend resources once, before configuring ops groups."""

    @abstractmethod
    def close(self, context: Any | None = None) -> None: ...

    @abstractmethod
    def profile(
        self,
        ops_groups: Sequence[TPSPOpsGroup],
        max_batched_tokens: int,
        *,
        microchunk_tokens: int | None = None,
        threshold_tokens: int | None = None,
    ) -> TPSPContext | None:
        """Configure named ops groups on an already-open backend.

        Args:
            ops_groups: Named projection and normalization pairs to profile.
            max_batched_tokens: Maximum scheduler token count to profile.
            microchunk_tokens: Optional global microchunk override.
            threshold_tokens: Optional global threshold override.

        Returns:
            The TPSP context, or None when unsupported.

        """

    @abstractmethod
    def tpsp_matmul_rs_norm_ag(
        self,
        projection_context: object,
        x: torch.Tensor,
        projection: nn.Module,
        residual: torch.Tensor,
        norm: nn.Module,
        *,
        config: object | None = None,
        norm_type: TPSPNormType = "rms_norm",
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def set_config(self, handle: object, config: object) -> None: ...

    def prepare_weights(self, projections: Sequence[nn.Module]) -> None:
        """Prepare backend-specific weights after model weights are loaded."""
        return None

    def get_tuning_configs(
        self,
        handle: object,
        microchunk_tokens_per_rank: int,
        group_name: str | None = None,
        *,
        compute_engine_counts: Sequence[int] | None = None,
        compute_engine_step: int = _COARSE_COMPUTE_ENGINE_STEP,
    ) -> Sequence[object]:
        """Return microchunk configs for the requested compute-engine counts."""
        return (microchunk_tokens_per_rank,)

    def get_compute_engine_count(self, config: object) -> int | None:
        """Return the compute-engine count represented by a tuning config."""
        count = getattr(config, "compute_engine_count", None)
        return count if type(count) is int else None

    def get_default_compute_engine_count(self) -> int | None:
        """Return the backend's default-selection sentinel, if it has one."""
        return None

    @abstractmethod
    def _create_context(self) -> object:
        """Create independent ops configuration using an already-open backend."""


@dataclass
class _ProjectionInputs:
    """Shared inputs for comparing a projection with its fused implementation."""

    hidden_states: torch.Tensor
    residual: torch.Tensor
    local_residual: torch.Tensor
    original_residual: torch.Tensor

    def restore_residual(self) -> None:
        self.residual.copy_(self.original_residual)


@dataclass(frozen=True)
class TPSPMeasurement:
    tokens: int
    conventional_ms: float
    tpsp_ms: float
    benefit_ms: float


@dataclass(frozen=True)
class TPSPScanResult:
    tp_size: int
    hidden_size: int
    max_batched_tokens: int
    status: str
    reason: str
    threshold_tokens: int | None = None
    config: object | None = None
    measurements: tuple[TPSPMeasurement, ...] = ()
    candidates: tuple[tuple[object, float], ...] = ()
    input_width: int | None = None
    finalists: tuple[tuple[object, float], ...] = ()
    selected_coarse_rank: int | None = None
    selected_seed_rank: int | None = None
    refined_candidates: tuple[tuple[int, object, float], ...] = ()

    @property
    def enabled(self) -> bool:
        return self.status == "enabled"


@dataclass(frozen=True)
class TPSPProfile:
    """Shared TPSP activation decision."""

    enabled: bool
    threshold_tokens: int | None
    max_batched_tokens: int
    reason: str = ""

    def is_active(self, num_tokens: int) -> bool:
        if not self.enabled:
            return False
        if self.threshold_tokens is None:
            raise RuntimeError("TPSP has an invalid enabled profile")
        if not 1 <= num_tokens <= self.max_batched_tokens:
            raise ValueError("current_batched_tokens must be within the profiled range")
        return num_tokens >= self.threshold_tokens


@dataclass(frozen=True)
class TPSPOpsGroup:
    """A consecutive projection and normalization pair."""

    name: str
    projection: nn.Module
    norm: nn.Module


@dataclass(frozen=True)
class _CompiledOpsGroup:
    ops: TPSPOpsGroup
    conventional: Callable[[_ProjectionInputs], torch.Tensor]


@dataclass(frozen=True)
class TPSPContext:
    """Shared backend and named per-group handles."""

    profile: TPSPProfile
    backend: Any
    handles: dict[str, object]


class UnsupportedTPSPNormError(ValueError):
    """Normalization not supported by the Transformers TPSP path."""


class UnsupportedTPSPModelError(ValueError):
    """Transformers model structure not supported by the TPSP path."""


def get_tpsp_norm_type(norm: nn.Module) -> TPSPNormType:
    """Identify known normalization implementations without guessing by attributes."""
    from vllm.model_executor.layers.layernorm import (
        LayerNorm,
        RMSNorm,
    )
    from vllm.model_executor.models.transformers.fusers.rms_norm import (
        TPAwareRMSNorm,
    )

    norm_cls = type(norm)
    if norm_cls in (nn.LayerNorm, LayerNorm):
        norm_type: TPSPNormType = "layer_norm"
    elif norm_cls in (RMSNorm, TPAwareRMSNorm):
        norm_type = "rms_norm"
        if norm.variance_size_override is not None:
            raise UnsupportedTPSPNormError(
                "TPSP RMSNorm does not support variance_size_override"
            )
    else:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )

    eps = (
        getattr(norm, "eps", None)
        if norm_type == "layer_norm"
        else getattr(norm, "variance_epsilon", None)
    )
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    bias = getattr(norm, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.ndim != 1
        or bias.shape != weight.shape
    ):
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if norm_type != "layer_norm" and bias is not None:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if not isinstance(eps, (float, int)):
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if eps <= 0:
        raise ValueError("TPSP norm_eps must be positive")
    return norm_type


def _unsupported_tpsp_norm_types(
    backend: TPSPBackend, ops_groups: Sequence[TPSPOpsGroup]
) -> frozenset[TPSPNormType]:
    norm_types = {get_tpsp_norm_type(entry.norm) for entry in ops_groups}
    return frozenset(norm_types - backend.supported_norm_types)


def tpsp_shard_residual(residual: torch.Tensor) -> torch.Tensor:
    """Return the current TP rank's padded shard of a full residual."""
    from vllm.distributed.parallel_state import get_tp_group

    if residual.ndim != 2:
        raise ValueError("TPSP full residual must be a 2D tensor")
    group = get_tp_group()
    rows = math.ceil(residual.size(0) / group.world_size)
    start = group.rank_in_group * rows
    count = min(rows, max(0, residual.size(0) - start))
    if count == rows:
        return residual[start : start + rows].contiguous()
    shard = residual.new_zeros((rows, residual.size(1)))
    shard[:count] = residual[start : start + count]
    return shard


def _group_parameter(value: Any, group_name: str, name: str) -> Any:
    if isinstance(value, Mapping):
        if group_name not in value:
            raise ValueError(f"{name} does not include TPSP group {group_name!r}")
        return value[group_name]
    return value


def _inclusive_range(minimum: int, maximum: int, step: int) -> tuple[int, ...]:
    values = tuple(range(minimum, maximum + 1, step))
    if not values or values[-1] != maximum:
        values += (maximum,)
    return values


def profile_tpsp(
    backend: Any,
    ops_groups: Sequence[TPSPOpsGroup],
    max_batched_tokens: int,
    *,
    microchunk_tokens: int | None = None,
    threshold_tokens: int | None = None,
    compute_engine_count_min: int | Mapping[str, int] | None = None,
    compute_engine_count_max: int | Mapping[str, int] | None = None,
    compute_engine_step_coarse: int = _COARSE_COMPUTE_ENGINE_STEP,
    compute_engine_step_fine: int = _FINE_COMPUTE_ENGINE_STEP,
    microchunk_tokens_per_rank_min_max: (
        tuple[int, int] | Mapping[str, tuple[int, int]] | None
    ) = None,
    microchunk_step_coarse_per_rank: int = _COARSE_MICROCHUNK_STEP_PER_RANK,
    microchunk_step_fine_per_rank: int = _FINE_MICROCHUNK_STEP_PER_RANK,
) -> TPSPContext | None:
    """Profile chunk/SM configurations and threshold, honoring token overrides.

    1. Connect each group's projection and norm into a runnable path.
    2. Open a handle, probe chunks, and jointly refine each group's best region.
    3. Find one threshold from the sum of separately timed group paths.
    4. Return a shared profile with the named handles, or use regular forward
       if any scan is inconclusive or finds no benefit.

    Args:
        backend: Backend that opens contexts and evaluates tuning candidates.
        ops_groups: Named projection and normalization pairs to profile.
        max_batched_tokens: Maximum scheduler token count to profile.
        microchunk_tokens: Fixed global token size, or None to scan.
        threshold_tokens: Fixed global threshold, or None to scan.
        compute_engine_count_min: Per-group minimum SM count, or a shared minimum.
        compute_engine_count_max: Per-group maximum SM count, or a shared maximum.
        compute_engine_step_coarse: Step for coarse SM candidates.
        compute_engine_step_fine: Step around finalist SM counts.
        microchunk_tokens_per_rank_min_max: Per-group per-rank microchunk bounds.
        microchunk_step_coarse_per_rank: Step for coarse candidates.
        microchunk_step_fine_per_rank: Step for local refinement.

    Returns:
        The selected TPSP context, or None when profiling is inconclusive.

    """
    if backend is None:
        raise ValueError("TPSP profiling requires a backend")
    if type(max_batched_tokens) is not int or max_batched_tokens <= 0:
        raise ValueError("TPSP max_batched_tokens must be a positive integer")
    for name, value in (
        ("microchunk_tokens", microchunk_tokens),
        ("threshold_tokens", threshold_tokens),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"TPSP {name} must be a positive integer")
    if microchunk_tokens is not None:
        microchunk_tokens = min(microchunk_tokens, max_batched_tokens)
    if threshold_tokens is not None:
        threshold_tokens = min(threshold_tokens, max_batched_tokens)

    # Step 1: assemble runnable ops groups.
    compiled = _compile_ops_groups(ops_groups)

    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()

    def disabled(reason: str) -> TPSPContext | None:
        _LOG.warning("TPSP using regular forward: %s", reason)
        return None

    if unsupported := _unsupported_tpsp_norm_types(backend, ops_groups):
        return disabled(
            f"backend does not support norm types {', '.join(sorted(unsupported))}"
        )

    scan_max_batched_tokens = max_batched_tokens

    started = time.perf_counter()
    enabled = False
    threshold = None
    handles: dict[str, object] = {}
    try:
        if scan_max_batched_tokens < 1:
            return disabled("max_batched_tokens must be positive")

        # Step 2: select a microchunk size for each group.
        if reason := _scan_group_chunks(
            backend,
            compiled,
            group,
            max_batched_tokens,
            handles,
            microchunk_tokens=microchunk_tokens,
            scan_configs=microchunk_tokens is None or threshold_tokens is None,
            compute_engine_count_min=compute_engine_count_min,
            compute_engine_count_max=compute_engine_count_max,
            compute_engine_step_coarse=compute_engine_step_coarse,
            compute_engine_step_fine=compute_engine_step_fine,
            microchunk_tokens_per_rank_min_max=microchunk_tokens_per_rank_min_max,
            microchunk_step_coarse_per_rank=microchunk_step_coarse_per_rank,
            microchunk_step_fine_per_rank=microchunk_step_fine_per_rank,
        ):
            return disabled(reason)

        # Step 3: scan one threshold using the sum of per-group timings.
        if group.rank_in_group == 0:
            _LOG.info(
                "TPSP threshold scan max_batched_tokens=%d "
                "(scheduler max_batched_tokens=%d)",
                scan_max_batched_tokens,
                max_batched_tokens,
            )
        if threshold_tokens is None:
            measurement = _scan_threshold(
                backend, compiled, handles, scan_max_batched_tokens
            )
        else:
            measurement = TPSPScanResult(
                group.world_size,
                compiled[0].ops.norm.weight.numel(),
                max_batched_tokens,
                "enabled",
                "fixed threshold",
                threshold_tokens=threshold_tokens,
            )
        if not measurement.enabled:
            return disabled(
                f"projection profile {measurement.status}: {measurement.reason}"
            )

        # Step 4: return the shared profile and named handles.
        threshold = measurement.threshold_tokens
        context = _build_tpsp_context(backend, handles, measurement, max_batched_tokens)
        enabled = True
        return context
    finally:
        if group.rank_in_group == 0:
            _LOG.info(
                "TPSP projection scan: elapsed=%.2fs threshold=%s enabled=%s",
                time.perf_counter() - started,
                threshold,
                enabled,
            )
        if not enabled:
            for handle in handles.values():
                backend.close(handle)


def _projection_inputs(
    tokens: int,
    projection: nn.Module,
    hidden_size: int,
    tp_size: int,
    rank: int,
    device: torch.device,
) -> _ProjectionInputs:
    input_width = _projection_input_width(projection)
    generator = torch.Generator(device=device).manual_seed(1831 + rank)
    a = torch.randn(
        tokens,
        input_width,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    residual = torch.randn(
        tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=torch.Generator(device=device).manual_seed(407),
    )
    rows = math.ceil(tokens / tp_size)
    padded = residual.new_zeros((tp_size * rows, hidden_size))
    padded[:tokens].copy_(residual)
    return _ProjectionInputs(
        hidden_states=a,
        residual=residual,
        local_residual=padded.narrow(0, rank * rows, rows).contiguous(),
        original_residual=residual.clone(),
    )


def _projection_input_width(projection: nn.Module) -> int:
    input_width = getattr(projection, "input_size_per_partition", None)
    if type(input_width) is int and input_width > 0:
        return input_width
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("TPSP projections must have a two-dimensional weight")
    input_width = weight.shape[1]
    if input_width <= 0:
        raise ValueError("TPSP projection input width must be positive")
    return input_width


def _run_conventional_projection(
    projection: nn.Module,
    norm: nn.Module,
    inputs: _ProjectionInputs,
    layer_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    projected = projection(inputs.hidden_states)
    full = projected[0] if isinstance(projected, tuple) else projected
    if layer_scale is not None:
        scale = layer_scale.to(dtype=full.dtype).reshape(-1) + 1
        full = full * scale
    # Instance wrappers hide whether the norm accepts a residual.
    forward = getattr(type(norm), "forward_native", type(norm).forward)
    try:
        inspect.signature(forward).bind(norm, full, inputs.residual)
    except TypeError:
        inputs.residual.add_(full)
        normalized = norm(inputs.residual)
    else:
        normalized = norm(full, inputs.residual)
    return normalized[0] if isinstance(normalized, tuple) else normalized


def _run_fused_projection(
    backend: Any,
    projection_context: object,
    projection: nn.Module,
    norm: nn.Module,
    inputs: _ProjectionInputs,
    microchunk_config: object | None = None,
) -> torch.Tensor:
    return backend.tpsp_matmul_rs_norm_ag(
        projection_context,
        inputs.hidden_states,
        projection,
        inputs.local_residual,
        norm,
        norm_type=get_tpsp_norm_type(norm),
        config=microchunk_config,
    )[0]


def _compile_ops_groups(
    ops_groups: Sequence[TPSPOpsGroup],
) -> tuple[_CompiledOpsGroup, ...]:
    """Connect each group's projection output to its normalization input."""
    if not ops_groups or any(not entry.name for entry in ops_groups):
        raise ValueError("TPSP profiling requires named ops groups")
    if len({entry.name for entry in ops_groups}) != len(ops_groups):
        raise ValueError("TPSP ops group names must be unique")
    first_norm_weight = getattr(ops_groups[0].norm, "weight", None)
    if not isinstance(first_norm_weight, torch.Tensor):
        raise ValueError("TPSP normalizations require a weight tensor")
    hidden_size = first_norm_weight.numel()
    if any(
        not isinstance(getattr(entry.norm, "weight", None), torch.Tensor)
        or entry.norm.weight.numel() != hidden_size
        for entry in ops_groups
    ):
        raise ValueError("TPSP ops groups require matching norm hidden sizes")
    for entry in ops_groups:
        _projection_input_width(entry.projection)
        norm_type = get_tpsp_norm_type(entry.norm)
        norm_eps = (
            entry.norm.eps if norm_type == "layer_norm" else entry.norm.variance_epsilon
        )
        if norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
    return tuple(
        _CompiledOpsGroup(
            entry,
            partial(_run_conventional_projection, entry.projection, entry.norm),
        )
        for entry in ops_groups
    )


def _scan_group_chunks(
    backend: Any,
    groups: tuple[_CompiledOpsGroup, ...],
    tp_group: Any,
    max_batched_tokens: int,
    handles: dict[str, object],
    *,
    microchunk_tokens: int | None,
    scan_configs: bool,
    compute_engine_count_min: int | Mapping[str, int] | None,
    compute_engine_count_max: int | Mapping[str, int] | None,
    compute_engine_step_coarse: int,
    compute_engine_step_fine: int,
    microchunk_tokens_per_rank_min_max: (
        tuple[int, int] | Mapping[str, tuple[int, int]] | None
    ),
    microchunk_step_coarse_per_rank: int,
    microchunk_step_fine_per_rank: int,
) -> str | None:
    """Select independent chunk/SM configurations per group."""
    if backend._closed:
        raise RuntimeError("TPSP backend is closed")
    hidden_size = groups[0].ops.norm.weight.numel()
    if (
        tp_group.world_size < 2
        or hidden_size <= 0
        or hidden_size % tp_group.world_size
        or max_batched_tokens < 1
    ):
        return (
            "requires supported TP, divisible hidden size and positive "
            "max_batched_tokens"
        )
    for compiled in groups:
        entry = compiled.ops
        handle = backend._create_context()
        handles[entry.name] = handle
        bounds = _group_parameter(
            microchunk_tokens_per_rank_min_max,
            entry.name,
            "microchunk_tokens_per_rank_min_max",
        )
        if microchunk_tokens is not None:
            fixed_chunk = _microchunk_tokens_per_rank(
                backend, microchunk_tokens, tp_group.world_size
            )
            bounds = (fixed_chunk, fixed_chunk)
        if not scan_configs:
            assert microchunk_tokens is not None
            configs = backend.get_tuning_configs(
                handle,
                fixed_chunk,
                group_name=entry.name,
                compute_engine_counts=(0,),
            )
            if not configs:
                return f"no fixed TPSP config available for {entry.name}"
            config = configs[0]
        else:
            candidate = _scan_microchunk(
                backend,
                handle,
                projection=entry.projection,
                norm=entry.norm,
                tp_size=tp_group.world_size,
                max_batched_tokens=max_batched_tokens,
                group_name=entry.name,
                compute_engine_count_min=_group_parameter(
                    compute_engine_count_min,
                    entry.name,
                    "compute_engine_count_min",
                ),
                compute_engine_count_max=_group_parameter(
                    compute_engine_count_max,
                    entry.name,
                    "compute_engine_count_max",
                ),
                compute_engine_step_coarse=compute_engine_step_coarse,
                compute_engine_step_fine=compute_engine_step_fine,
                microchunk_tokens_per_rank_min_max=bounds,
                microchunk_step_coarse_per_rank=microchunk_step_coarse_per_rank,
                microchunk_step_fine_per_rank=microchunk_step_fine_per_rank,
            )
            if candidate.status != "candidate" or candidate.config is None:
                return (
                    f"{entry.name} microchunk selection {candidate.status}: "
                    f"{candidate.reason}"
                )
            config = candidate.config
        backend.set_config(handle, config)
        if tp_group.rank_in_group == 0:
            _LOG.info("TPSP selected group=%s config=%s", entry.name, config)
    return None


def _microchunk_tokens_per_rank(
    backend: TPSPBackend, microchunk_tokens: int, tp_size: int
) -> int:
    if type(microchunk_tokens) is not int or microchunk_tokens <= 0:
        raise ValueError("TPSP microchunk tokens must be a positive integer")
    if type(tp_size) is not int or tp_size <= 0:
        raise ValueError("TPSP TP size must be a positive integer")
    tokens_per_rank = math.ceil(microchunk_tokens / tp_size)
    max_tokens_per_rank = backend.tpsp_max_microchunk_tokens_per_rank
    if max_tokens_per_rank is not None:
        tokens_per_rank = min(tokens_per_rank, max_tokens_per_rank)
    return tokens_per_rank


def _build_tpsp_context(
    backend: Any,
    handles: dict[str, object],
    measurement: TPSPScanResult,
    max_batched_tokens: int,
) -> TPSPContext:
    """Build the shared profile and named handles after a successful scan."""
    threshold = measurement.threshold_tokens
    if (
        not measurement.enabled
        or threshold is None
        or not 1 <= threshold <= max_batched_tokens
    ):
        raise RuntimeError("TPSP has an invalid enabled profile")
    return TPSPContext(
        TPSPProfile(True, threshold, max_batched_tokens), backend, handles
    )


def _beneficial(measurement: TPSPMeasurement) -> bool:
    return measurement.benefit_ms > 0.01 * measurement.conventional_ms


@torch.inference_mode()
def _scan_threshold(
    backend: Any,
    ops_groups: Sequence[_CompiledOpsGroup],
    handles: dict[str, object],
    max_batched_tokens: int,
) -> TPSPScanResult:
    """Compare the sum of the group timings with their normal paths."""
    group = c10d._resolve_process_group(backend.group_name)
    tp_size = dist.get_world_size(group)
    hidden_size = ops_groups[0].ops.norm.weight.numel()
    rank = dist.get_rank(group)
    device = backend.device
    threshold_started = time.perf_counter()
    entries = tuple((handles[entry.ops.name], entry) for entry in ops_groups)

    def measure(
        handle: object,
        compiled: _CompiledOpsGroup,
        inputs: _ProjectionInputs,
        fused: bool,
    ) -> float:
        if not fused:
            inputs.restore_residual()
        dist.barrier(group=group)
        torch.accelerator.synchronize(device)
        start = time.perf_counter()
        if fused:
            result = _run_fused_projection(
                backend, handle, compiled.ops.projection, compiled.ops.norm, inputs
            )
        else:
            result = compiled.conventional(inputs)
        torch.accelerator.synchronize(device)
        elapsed = torch.tensor(
            [(time.perf_counter() - start) * 1000], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        del result
        return elapsed.item()

    def measure_size(tokens: int) -> TPSPMeasurement:
        data = tuple(
            _projection_inputs(
                tokens, compiled.ops.projection, hidden_size, tp_size, rank, device
            )
            for _, compiled in entries
        )
        for (handle, compiled), inputs in zip(entries, data):
            measure(handle, compiled, inputs, False)
            measure(handle, compiled, inputs, True)
        conventional = sum(
            measure(handle, compiled, inputs, False)
            for (handle, compiled), inputs in zip(entries, data)
        )
        fused = [
            sum(
                measure(handle, compiled, inputs, True)
                for (handle, compiled), inputs in zip(entries, data)
            )
            for _ in range(_TRIALS)
        ]
        median_tpsp = statistics.median(fused)
        return TPSPMeasurement(
            tokens, conventional, median_tpsp, conventional - median_tpsp
        )

    sizes = _inclusive_range(128, max_batched_tokens, _COARSE_THRESHOLD_STEP)
    threshold_progress = _TPSPProgress(
        ",".join(entry.ops.name for entry in ops_groups),
        "threshold",
        len(sizes),
        rank,
        started_at=threshold_started,
    )
    measurements_by_size: dict[int, TPSPMeasurement] = {}
    threshold = None
    streak = 0
    streak_start = None
    refine_start = None
    previous_size = None
    for coarse_steps, tokens in enumerate(sizes, start=1):
        result = measure_size(tokens)
        measurements_by_size[tokens] = result
        threshold_progress.update_percent(
            70 * coarse_steps // len(sizes),
            detail=f"coarse points={coarse_steps}/{len(sizes)}",
        )
        if _beneficial(result):
            if streak == 0:
                streak_start = tokens
                refine_start = previous_size if previous_size is not None else tokens
            streak += 1
        else:
            streak = 0
            streak_start = None
            refine_start = None
        if streak == 3:
            break
        previous_size = tokens

    threshold_progress.update_percent(70, detail="coarse phase complete")
    if streak == 3 and streak_start is not None and refine_start is not None:
        fine_end = min(max_batched_tokens, streak_start + 2 * _FINE_THRESHOLD_STEP)
        fine_sizes = _inclusive_range(refine_start, fine_end, _FINE_THRESHOLD_STEP)
        fine_streak = 0
        previous_fine_size = None
        for index, tokens in enumerate(fine_sizes, start=1):
            fine_result = measurements_by_size.get(tokens)
            if fine_result is None:
                fine_result = measure_size(tokens)
                measurements_by_size[tokens] = fine_result
            threshold_progress.update_percent(
                70 + 25 * index // len(fine_sizes),
                detail=f"fine points={index}/{len(fine_sizes)}",
            )
            if previous_fine_size is not None and (
                tokens - previous_fine_size != _FINE_THRESHOLD_STEP
            ):
                fine_streak = 0
            fine_streak = fine_streak + 1 if _beneficial(fine_result) else 0
            if fine_streak == 3:
                threshold = tokens - 2 * _FINE_THRESHOLD_STEP
                break
            previous_fine_size = tokens
    else:
        threshold_progress.update_percent(95, detail="fine refinement not needed")

    maximum = measurements_by_size.get(max_batched_tokens)
    if maximum is None:
        maximum = measure_size(max_batched_tokens)
        measurements_by_size[max_batched_tokens] = maximum
    if threshold is not None and not _beneficial(maximum):
        threshold = None
    if threshold is None and maximum.benefit_ms > 0:
        threshold = max_batched_tokens
    status = "enabled" if threshold is not None else "disabled"
    reason = "" if threshold is not None else "no sustained summed TPSP benefit"
    threshold_progress.finish(
        status, detail=f"measured sizes={len(measurements_by_size)}"
    )
    profile = TPSPScanResult(
        tp_size,
        hidden_size,
        max_batched_tokens,
        status,
        reason,
        threshold_tokens=threshold,
        measurements=tuple(
            measurements_by_size[tokens] for tokens in sorted(measurements_by_size)
        ),
    )
    if rank == 0:
        _LOG.info("TPSP projection startup profile: %s", profile)
    return profile


def _scan_microchunk(
    backend: Any,
    projection_context: object,
    projection: nn.Module,
    norm: nn.Module,
    tp_size: int,
    max_batched_tokens: int,
    group_name: str | None = None,
    compute_engine_count_min: int | None = None,
    compute_engine_count_max: int | None = None,
    compute_engine_step_coarse: int = _COARSE_COMPUTE_ENGINE_STEP,
    compute_engine_step_fine: int = _FINE_COMPUTE_ENGINE_STEP,
    microchunk_tokens_per_rank_min_max: tuple[int, int] | None = None,
    microchunk_step_coarse_per_rank: int = _COARSE_MICROCHUNK_STEP_PER_RANK,
    microchunk_step_fine_per_rank: int = _FINE_MICROCHUNK_STEP_PER_RANK,
) -> TPSPScanResult:
    """Probe chunks, jointly refine a local grid, and check candidate parity.

    Returns:
        TPSPScanResult with timings and the selected configuration in ``config``.
        The shared group profile determines the token threshold.

    """
    device = backend.device
    hidden_size = norm.weight.numel()
    input_width = _projection_input_width(projection)
    if any(
        type(step) is not int or step <= 0
        for step in (
            compute_engine_step_coarse,
            compute_engine_step_fine,
            microchunk_step_coarse_per_rank,
            microchunk_step_fine_per_rank,
        )
    ):
        raise ValueError("TPSP scan steps must be positive integers")
    if (
        compute_engine_step_fine > compute_engine_step_coarse
        or microchunk_step_fine_per_rank > microchunk_step_coarse_per_rank
    ):
        raise ValueError("Fine TPSP scan steps cannot exceed coarse steps")
    if (compute_engine_count_min is None) != (compute_engine_count_max is None):
        raise ValueError(
            "compute_engine_count_min and compute_engine_count_max must be set together"
        )
    if compute_engine_count_min is not None and (
        type(compute_engine_count_min) is not int
        or type(compute_engine_count_max) is not int
        or compute_engine_count_min < 0
        or compute_engine_count_min > compute_engine_count_max
    ):
        raise ValueError("Invalid compute-engine count range")
    group = c10d._resolve_process_group(backend.group_name)
    if dist.get_world_size(group) != tp_size:
        raise ValueError("group_name and tp_size disagree")
    rank = dist.get_rank(group)

    def make_inputs(tokens: int) -> _ProjectionInputs:
        return _projection_inputs(
            tokens, projection, hidden_size, tp_size, rank, device
        )

    def run_fused(inputs: _ProjectionInputs, config: object) -> torch.Tensor:
        return _run_fused_projection(
            backend, projection_context, projection, norm, inputs, config
        )

    def time_config_ms(inputs: _ProjectionInputs, config: object) -> float:
        dist.barrier(group=group)
        torch.accelerator.synchronize(device)
        start = time.perf_counter()
        result = run_fused(inputs, config)
        torch.accelerator.synchronize(device)
        elapsed = torch.tensor(
            [(time.perf_counter() - start) * 1000], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        value = elapsed.item()
        del result
        return value

    def inconclusive(reason: str, candidate_results=()) -> TPSPScanResult:
        if rank == 0:
            _LOG.warning("TPSP startup profile: status=inconclusive reason=%s", reason)
        return TPSPScanResult(
            tp_size,
            hidden_size,
            max_batched_tokens,
            "inconclusive",
            reason,
            candidates=tuple(candidate_results),
            input_width=input_width,
        )

    # Stage 1 screens the selected chunk/count candidates at full load.
    screening_inputs = make_inputs(max_batched_tokens)
    supported_max_microchunk_tokens_per_rank = math.ceil(max_batched_tokens / tp_size)
    cap_tokens_per_rank = backend.tpsp_max_microchunk_tokens_per_rank
    if cap_tokens_per_rank is not None:
        supported_max_microchunk_tokens_per_rank = min(
            supported_max_microchunk_tokens_per_rank, cap_tokens_per_rank
        )
    min_microchunk_tokens_per_rank, max_microchunk_tokens_per_rank = (
        microchunk_tokens_per_rank_min_max
        if microchunk_tokens_per_rank_min_max is not None
        else (1, supported_max_microchunk_tokens_per_rank)
    )
    if (
        type(min_microchunk_tokens_per_rank) is not int
        or type(max_microchunk_tokens_per_rank) is not int
        or min_microchunk_tokens_per_rank < 1
        or max_microchunk_tokens_per_rank > supported_max_microchunk_tokens_per_rank
        or min_microchunk_tokens_per_rank > max_microchunk_tokens_per_rank
    ):
        raise ValueError("Invalid TPSP microchunk token range")
    probe_count: int | None = None
    sparse_microchunks = _inclusive_range(
        min_microchunk_tokens_per_rank,
        max_microchunk_tokens_per_rank,
        microchunk_step_coarse_per_rank,
    )
    if compute_engine_count_min is not None:
        assert compute_engine_count_max is not None
        positive_min = max(1, compute_engine_count_min)
        positive_counts = (
            _inclusive_range(
                positive_min,
                compute_engine_count_max,
                compute_engine_step_coarse,
            )
            if positive_min <= compute_engine_count_max
            else ()
        )
        if positive_counts:
            midpoint = (positive_counts[0] + positive_counts[-1]) / 2
            probe_count = min(
                positive_counts,
                key=lambda count: (abs(count - midpoint), -count),
            )
            coarse_microchunks = _inclusive_range(
                min_microchunk_tokens_per_rank,
                max_microchunk_tokens_per_rank,
                microchunk_step_fine_per_rank,
            )
        else:
            probe_count = None
            coarse_microchunks = sparse_microchunks
    else:
        probe_count = None
        coarse_microchunks = sparse_microchunks
    joint_probe_scan = probe_count is not None
    coarse_results: list[tuple[object, float, int]] = []

    def coarse_targets() -> tuple[int, ...] | None:
        if compute_engine_count_min is None:
            return None
        assert compute_engine_count_max is not None
        targets = _inclusive_range(
            compute_engine_count_min,
            compute_engine_count_max,
            compute_engine_step_coarse,
        )
        default_count = backend.get_default_compute_engine_count()
        if default_count is not None:
            targets = tuple(dict.fromkeys((default_count, *targets)))
        return targets

    def fine_targets(
        center: int | None, available: Sequence[int]
    ) -> tuple[int, ...] | None:
        if center is None:
            return None
        if center == backend.get_default_compute_engine_count():
            return (center,)
        positive = [count for count in available if count > 0]
        lower = (
            compute_engine_count_min
            if compute_engine_count_min is not None
            else min(positive, default=center)
        )
        upper = (
            compute_engine_count_max
            if compute_engine_count_max is not None
            else max(positive, default=center)
        )
        minimum = max(
            lower, center - _FINE_NEIGHBORHOOD_RADIUS * compute_engine_step_fine
        )
        maximum = min(
            upper, center + _FINE_NEIGHBORHOOD_RADIUS * compute_engine_step_fine
        )
        targets: tuple[int, ...] = _inclusive_range(
            minimum, maximum, compute_engine_step_fine
        )
        if center not in targets:
            targets = tuple(sorted((*targets, center)))
        return targets

    def local_microchunks(center: int) -> tuple[int, ...]:
        radius = _FINE_NEIGHBORHOOD_RADIUS * microchunk_step_fine_per_rank
        minimum = max(min_microchunk_tokens_per_rank, center - radius)
        maximum = min(max_microchunk_tokens_per_rank, center + radius)
        microchunks = _inclusive_range(minimum, maximum, microchunk_step_fine_per_rank)
        if center not in microchunks:
            microchunks = tuple(sorted((*microchunks, center)))
        return microchunks

    def score_config_ms(
        config: object, inputs: _ProjectionInputs, timings_count: int
    ) -> float:
        time_config_ms(inputs, config)
        timings = []
        for _ in range(timings_count):
            timings.append(time_config_ms(inputs, config))
        return statistics.median(timings)

    coarse_started = time.perf_counter()
    coarse_configs = []
    for microchunk_tokens_per_rank in coarse_microchunks:
        targets = coarse_targets()
        if joint_probe_scan:
            assert probe_count is not None
            targets = (probe_count,)
            default_count = backend.get_default_compute_engine_count()
            if (
                default_count is not None
                and microchunk_tokens_per_rank in sparse_microchunks
            ):
                targets = tuple(dict.fromkeys((default_count, *targets)))
        configs = backend.get_tuning_configs(
            projection_context,
            microchunk_tokens_per_rank,
            group_name=group_name,
            compute_engine_counts=targets,
            compute_engine_step=compute_engine_step_coarse,
        )
        if not configs:
            raise ValueError("TPSP backend returned no tuning configurations")
        coarse_configs.append((microchunk_tokens_per_rank, configs))
    coarse_progress = _TPSPProgress(
        group_name or backend.group_name,
        "coarse",
        sum(len(configs) for _, configs in coarse_configs),
        rank,
        started_at=coarse_started,
    )
    for microchunk_tokens_per_rank, configs in coarse_configs:
        for config in configs:
            score = score_config_ms(config, screening_inputs, _COARSE_TIMINGS)
            coarse_results.append((config, score, microchunk_tokens_per_rank))
            coarse_progress.update(len(coarse_results))
    coarse_progress.finish("complete")
    coarse_ranked_results = tuple(sorted(coarse_results, key=lambda result: result[1]))
    coarse_top = coarse_ranked_results[:_FINALIST_COUNT]
    coarse_candidates = tuple((config, score) for config, score, _ in coarse_top)
    probe_seed = None
    default_seed = None
    if joint_probe_scan:
        assert probe_count is not None
        probe_seed = min(
            (
                result
                for result in coarse_results
                if backend.get_compute_engine_count(result[0]) == probe_count
            ),
            key=lambda result: result[1],
            default=None,
        )
        if probe_seed is None:
            if rank == 0:
                _LOG.info(
                    "TPSP midpoint probe unavailable group=%s; "
                    "using finalist refinement",
                    group_name,
                )
            joint_probe_scan = False
        else:
            default_count = backend.get_default_compute_engine_count()
            if default_count is not None and default_count != probe_count:
                default_seed = min(
                    (
                        result
                        for result in coarse_results
                        if backend.get_compute_engine_count(result[0]) == default_count
                    ),
                    key=lambda result: result[1],
                    default=None,
                )
    available_engine_counts = tuple(
        dict.fromkeys(
            count
            for config, _, _ in coarse_results
            if (count := backend.get_compute_engine_count(config)) is not None
        )
    )
    fine_measurements: dict[
        tuple[int, int | None], tuple[tuple[object, float], ...]
    ] = {}
    if joint_probe_scan:
        assert probe_seed is not None
        probe_config, _, probe_microchunk_tokens_per_rank = probe_seed
        probe_counts = fine_targets(
            backend.get_compute_engine_count(probe_config), available_engine_counts
        )
        fine_total = len(local_microchunks(probe_microchunk_tokens_per_rank)) * len(
            probe_counts if probe_counts is not None else (None,)
        )
        fine_total += int(default_seed is not None)
    else:
        fine_total = sum(
            len(local_microchunks(seed_microchunk_tokens_per_rank))
            + len(
                fine_targets(
                    backend.get_compute_engine_count(seed_config),
                    available_engine_counts,
                )
                or (None,)
            )
            for seed_config, _, seed_microchunk_tokens_per_rank in coarse_top
        )
    fine_progress = _TPSPProgress(
        group_name or backend.group_name,
        "fine",
        fine_total,
        rank,
    )
    fine_completed = 0

    def advance_fine_progress() -> None:
        nonlocal fine_completed
        fine_completed += 1
        fine_progress.update(fine_completed, percent_limit=99)

    def measure_pair(
        microchunk_tokens_per_rank: int, count: int | None, inputs: _ProjectionInputs
    ) -> tuple[tuple[object, float], ...]:
        pair = (microchunk_tokens_per_rank, count)
        if pair in fine_measurements:
            return fine_measurements[pair]
        configs = backend.get_tuning_configs(
            projection_context,
            microchunk_tokens_per_rank,
            group_name=group_name,
            compute_engine_counts=(count,) if count is not None else None,
            compute_engine_step=compute_engine_step_fine,
        )
        if not configs:
            raise ValueError("TPSP backend returned no fine tuning configurations")
        results = []
        for config in configs:
            prior = None
            for measured in fine_measurements.values():
                prior = next(
                    (
                        score
                        for measured_config, score in measured
                        if measured_config == config
                    ),
                    None,
                )
                if prior is not None:
                    break
            score = prior
            if score is None:
                score = score_config_ms(config, inputs, _FINE_TIMINGS)
            results.append((config, score))
        fine_measurements[pair] = tuple(results)
        return fine_measurements[pair]

    refined_candidates: list[tuple[int, object, float]] = []
    if joint_probe_scan:
        assert probe_seed is not None
        seed_config, _, seed_microchunk_tokens_per_rank = probe_seed
        seed_count = backend.get_compute_engine_count(seed_config)
        counts = fine_targets(seed_count, available_engine_counts)
        joint_results: list[tuple[int, object, float]] = []
        for microchunk_tokens_per_rank in local_microchunks(
            seed_microchunk_tokens_per_rank
        ):
            for count in counts if counts is not None else (None,):
                measured = measure_pair(
                    microchunk_tokens_per_rank, count, screening_inputs
                )
                joint_results.extend(
                    (microchunk_tokens_per_rank, config, score)
                    for config, score in measured
                )
                advance_fine_progress()
        _, candidate, candidate_score = min(joint_results, key=lambda result: result[2])
        refined_candidates.append((1, candidate, candidate_score))

        if default_seed is not None:
            default_config, _, default_microchunk_tokens_per_rank = default_seed
            default_count = backend.get_default_compute_engine_count()
            assert default_count is not None
            default_results = measure_pair(
                default_microchunk_tokens_per_rank,
                default_count,
                screening_inputs,
            )
            advance_fine_progress()
            candidate, candidate_score = min(
                default_results, key=lambda result: result[1]
            )
            refined_candidates.append((2, candidate, candidate_score))
    else:
        for seed_rank, (seed_config, _, seed_microchunk_tokens_per_rank) in enumerate(
            coarse_top, start=1
        ):
            seed_count = backend.get_compute_engine_count(seed_config)
            microchunk_results: list[tuple[int, object, float]] = []
            for microchunk_tokens_per_rank in local_microchunks(
                seed_microchunk_tokens_per_rank
            ):
                measured = measure_pair(
                    microchunk_tokens_per_rank, seed_count, screening_inputs
                )
                microchunk_results.extend(
                    (microchunk_tokens_per_rank, config, score)
                    for config, score in measured
                )
                advance_fine_progress()
            selected_microchunk_tokens_per_rank = min(
                microchunk_results, key=lambda result: result[2]
            )[0]

            counts = fine_targets(seed_count, available_engine_counts)
            count_results: list[tuple[object, float]] = []
            for count in counts if counts is not None else (None,):
                measured = measure_pair(
                    selected_microchunk_tokens_per_rank, count, screening_inputs
                )
                count_results.extend(measured)
                advance_fine_progress()
            candidate, candidate_score = min(
                count_results, key=lambda result: result[1]
            )
            refined_candidates.append((seed_rank, candidate, candidate_score))

    if not refined_candidates:
        raise ValueError("TPSP coarse scan returned no finalist candidates")
    ranked_candidates: list[tuple[int, object, float]] = []
    for result in sorted(refined_candidates, key=lambda candidate: candidate[2]):
        if not any(config == result[1] for _, config, _ in ranked_candidates):
            ranked_candidates.append(result)
    selected_config: object | None = None
    selected_score: float | None = None
    selected_seed_rank: int | None = None
    parity_inputs = min(max_batched_tokens, tp_size * 5 + 1)
    for seed_rank, candidate_config, score in ranked_candidates:
        check_inputs = make_inputs(parity_inputs)
        conventional = _run_conventional_projection(projection, norm, check_inputs)
        native = run_fused(check_inputs, candidate_config)
        torch.accelerator.synchronize(device)
        local_failure = False
        try:
            torch.testing.assert_close(native, conventional, rtol=0.02, atol=0.05)
        except AssertionError:
            local_failure = True
        del check_inputs, conventional, native
        failed = torch.tensor([int(local_failure)], dtype=torch.int32, device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        if failed.item():
            if rank == 0:
                _LOG.warning(
                    "TPSP parity failed group=%s seed_rank=%d config=%s; "
                    "trying next finalist",
                    group_name,
                    seed_rank,
                    candidate_config,
                )
            continue
        selected_config = candidate_config
        selected_score = score
        selected_seed_rank = seed_rank
        break
    if selected_config is None or selected_score is None:
        fine_progress.finish("inconclusive")
        return inconclusive(
            "all refined candidates failed numerical parity",
            tuple((config, score) for _, config, score in refined_candidates),
        )
    fine_progress.finish("complete")
    selected_coarse_rank = next(
        (
            coarse_rank
            for coarse_rank, (coarse_config, _, _) in enumerate(
                coarse_ranked_results, start=1
            )
            if coarse_config == selected_config
        ),
        None,
    )
    if rank == 0:
        _LOG.info(
            "TPSP selected group=%s config=%s latency=%.3fms "
            "coarse_rank=%s seed_rank=%d",
            group_name,
            selected_config,
            selected_score,
            selected_coarse_rank,
            selected_seed_rank,
        )
    del screening_inputs
    return TPSPScanResult(
        tp_size,
        hidden_size,
        max_batched_tokens,
        "candidate",
        "",
        config=selected_config,
        candidates=coarse_candidates,
        input_width=input_width,
        finalists=tuple(
            (config, score) for _, config, score in ranked_candidates[:_FINALIST_COUNT]
        ),
        selected_coarse_rank=selected_coarse_rank,
        selected_seed_rank=selected_seed_rank,
        refined_candidates=tuple(refined_candidates),
    )
