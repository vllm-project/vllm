# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 TPSP backends, execution contexts, and profiling utilities."""

from __future__ import annotations

import logging
import math
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import distributed_c10d as c10d

_LOG = logging.getLogger(__name__)
_TRIALS = 3
_COARSE_CHUNK_STEP = 256
_FINE_CHUNK_STEP = 64
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


class TPSPBackend(ABC):
    tpsp_chunk_granularity: int
    tpsp_max_microchunk_rows: int | None = None
    tpsp_max_microchunk_tokens: int | None = None

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
    ) -> Any | None: ...

    def profile(
        self,
        ops_groups: Sequence[TPSPOpsGroup],
        max_batched_tokens: int,
        *,
        compute_engine_count_min: int | Mapping[str, int] | None = None,
        compute_engine_count_max: int | Mapping[str, int] | None = None,
        compute_engine_step_coarse: int = _COARSE_COMPUTE_ENGINE_STEP,
        compute_engine_step_fine: int = _FINE_COMPUTE_ENGINE_STEP,
        chunk_min_max: tuple[int, int] | Mapping[str, tuple[int, int]] | None = None,
        chunk_step_coarse: int = _COARSE_CHUNK_STEP,
        chunk_step_fine: int = _FINE_CHUNK_STEP,
    ) -> TPSPContext | None:
        """Select independent chunks and a shared threshold for ops groups.

        Args:
            ops_groups: Named projection and normalization pairs to profile.
            max_batched_tokens: Maximum scheduler token count to profile.
            compute_engine_count_min: Per-group minimum count, or a shared
                minimum. A backend may add its default-selection candidate.
            compute_engine_count_max: Per-group maximum count, or a shared
                maximum.
            compute_engine_step_coarse: Step for the full-grid count scan.
            compute_engine_step_fine: Step around finalist counts.
            chunk_min_max: Per-group chunk bounds, or shared row bounds.
            chunk_step_coarse: Step for default-count chunk candidates.
            chunk_step_fine: Step for the full chunk probe and local refinement.

        Returns:
            The selected TPSP context, or None when profiling is inconclusive.

        """
        return profile_tpsp(
            self,
            ops_groups,
            max_batched_tokens,
            compute_engine_count_min=compute_engine_count_min,
            compute_engine_count_max=compute_engine_count_max,
            compute_engine_step_coarse=compute_engine_step_coarse,
            compute_engine_step_fine=compute_engine_step_fine,
            chunk_min_max=chunk_min_max,
            chunk_step_coarse=chunk_step_coarse,
            chunk_step_fine=chunk_step_fine,
        )

    @abstractmethod
    def set_config(self, handle: object, config: object) -> None: ...

    def get_tuning_configs(
        self,
        handle: object,
        chunk: int,
        group_name: str | None = None,
        *,
        compute_engine_counts: Sequence[int] | None = None,
        compute_engine_step: int = _COARSE_COMPUTE_ENGINE_STEP,
    ) -> Sequence[object]:
        """Return chunk configs for the requested compute-engine counts."""
        return (chunk,)

    def get_compute_engine_count(self, config: object) -> int | None:
        """Return the compute-engine count represented by a tuning config."""
        count = getattr(config, "compute_engine_count", None)
        return count if type(count) is int else None

    def get_default_compute_engine_count(self) -> int | None:
        """Return the backend's default-selection sentinel, if it has one."""
        return None

    def get_max_microchunk_rows(self, tp_size: int) -> int | None:
        if self.tpsp_max_microchunk_tokens is not None:
            return math.ceil(self.tpsp_max_microchunk_tokens / tp_size)
        return self.tpsp_max_microchunk_rows

    @abstractmethod
    def fused_gemm_rs_norm_ag(
        self,
        projection_context: object,
        x: torch.Tensor,
        projection: nn.Module,
        residual: torch.Tensor,
        norm: nn.Module,
        *,
        config: object | None = None,
        norm_type: str = "rms_norm",
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def close(self, context: Any | None = None) -> None: ...


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
    compute_engine_count_min: int | Mapping[str, int] | None = None,
    compute_engine_count_max: int | Mapping[str, int] | None = None,
    compute_engine_step_coarse: int = _COARSE_COMPUTE_ENGINE_STEP,
    compute_engine_step_fine: int = _FINE_COMPUTE_ENGINE_STEP,
    chunk_min_max: tuple[int, int] | Mapping[str, tuple[int, int]] | None = None,
    chunk_step_coarse: int = _COARSE_CHUNK_STEP,
    chunk_step_fine: int = _FINE_CHUNK_STEP,
) -> TPSPContext | None:
    """Profile named ops groups in four steps.

    1. Connect each group's projection and norm into a runnable path.
    2. Open a handle, probe chunks, and jointly refine each group's best region.
    3. Find one threshold from the sum of separately timed group paths.
    4. Return a shared profile with the named handles, or use regular forward
       if any scan is inconclusive or finds no benefit.

    Args:
        backend: Backend that opens contexts and evaluates tuning candidates.
        ops_groups: Named projection and normalization pairs to profile.
        max_batched_tokens: Maximum scheduler token count to profile.
        compute_engine_count_min: Per-group minimum count, or a shared
            minimum. A backend may add its default-selection candidate.
        compute_engine_count_max: Per-group maximum count, or a shared maximum.
        compute_engine_step_coarse: Step for the full-grid count scan.
        compute_engine_step_fine: Step around finalist counts.
        chunk_min_max: Per-group chunk bounds, or shared row bounds.
        chunk_step_coarse: Step for default-count chunk candidates.
        chunk_step_fine: Step for the full chunk probe and local refinement.

    Returns:
        The selected TPSP context, or None when profiling is inconclusive.

    """
    if backend is None:
        raise ValueError("TPSP profiling requires a backend")

    # Step 1: assemble runnable ops groups.
    compiled = _compile_ops_groups(ops_groups)

    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()

    def disabled(reason: str) -> TPSPContext | None:
        _LOG.warning("TPSP using regular forward: %s", reason)
        return None

    scan_max_batched_tokens = max_batched_tokens - max_batched_tokens % group.world_size

    started = time.perf_counter()
    enabled = False
    threshold = None
    handles: dict[str, object] = {}
    try:
        if scan_max_batched_tokens < group.world_size:
            return disabled("no TP-aligned token count within max_batched_tokens")

        # Step 2: select a chunk for each group.
        if reason := _scan_group_chunks(
            backend,
            compiled,
            group,
            max_batched_tokens,
            handles,
            compute_engine_count_min=compute_engine_count_min,
            compute_engine_count_max=compute_engine_count_max,
            compute_engine_step_coarse=compute_engine_step_coarse,
            compute_engine_step_fine=compute_engine_step_fine,
            chunk_min_max=chunk_min_max,
            chunk_step_coarse=chunk_step_coarse,
            chunk_step_fine=chunk_step_fine,
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
        measurement = _scan_threshold(
            backend, compiled, handles, scan_max_batched_tokens
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
            backend.close()


def _projection_inputs(
    tokens: int,
    projection: nn.Module,
    hidden_size: int,
    tp_size: int,
    rank: int,
    device: torch.device,
) -> _ProjectionInputs:
    generator = torch.Generator(device=device).manual_seed(1831 + rank)
    a = torch.randn(
        tokens,
        projection.input_size_per_partition,
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


def _run_conventional_projection(
    projection: nn.Module,
    norm: nn.Module,
    inputs: _ProjectionInputs,
) -> torch.Tensor:
    full, _ = projection(inputs.hidden_states)
    full, _ = norm(full, inputs.residual)
    return full


def _run_fused_projection(
    backend: Any,
    projection_context: object,
    projection: nn.Module,
    norm: nn.Module,
    inputs: _ProjectionInputs,
    chunk: object | None = None,
) -> torch.Tensor:
    return backend.fused_gemm_rs_norm_ag(
        projection_context,
        inputs.hidden_states,
        projection,
        inputs.local_residual,
        norm,
        config=chunk,
    )[0]


def _compile_ops_groups(
    ops_groups: Sequence[TPSPOpsGroup],
) -> tuple[_CompiledOpsGroup, ...]:
    """Connect each group's projection output to its normalization input."""
    if not ops_groups or any(not entry.name for entry in ops_groups):
        raise ValueError("TPSP profiling requires named ops groups")
    if len({entry.name for entry in ops_groups}) != len(ops_groups):
        raise ValueError("TPSP ops group names must be unique")
    hidden_size = ops_groups[0].norm.weight.numel()
    if any(entry.norm.weight.numel() != hidden_size for entry in ops_groups):
        raise ValueError("TPSP ops groups require matching norm hidden sizes")
    for entry in ops_groups:
        if entry.projection.input_size_per_partition <= 0:
            raise ValueError("input_width must be positive")
        if entry.norm.variance_epsilon <= 0:
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
    compute_engine_count_min: int | Mapping[str, int] | None,
    compute_engine_count_max: int | Mapping[str, int] | None,
    compute_engine_step_coarse: int,
    compute_engine_step_fine: int,
    chunk_min_max: tuple[int, int] | Mapping[str, tuple[int, int]] | None,
    chunk_step_coarse: int,
    chunk_step_fine: int,
) -> str | None:
    """Select an independent chunk per group, retaining opened handles."""
    if backend._closed:
        raise RuntimeError("TPSP backend is closed")
    granularity = backend.tpsp_chunk_granularity
    if type(granularity) is not int or granularity <= 0:
        raise ValueError("TPSP chunk granularity must be a positive integer")
    parameter = groups[0].ops.projection.weight
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
        handle = backend.open(
            dtype=parameter.dtype,
            tp_size=tp_group.world_size,
            hidden_size=hidden_size,
            max_batched_tokens=max_batched_tokens,
            group_name=tp_group.device_group.group_name,
            device=parameter.device,
        )
        if handle is None:
            return f"fused backend unavailable for {entry.name}"
        handles[entry.name] = handle
        candidate = _scan_chunk(
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
            chunk_min_max=_group_parameter(chunk_min_max, entry.name, "chunk_min_max"),
            chunk_step_coarse=chunk_step_coarse,
            chunk_step_fine=chunk_step_fine,
        )
        if candidate.status != "candidate" or candidate.config is None:
            return (
                f"{entry.name} chunk selection {candidate.status}: {candidate.reason}"
            )
        config = candidate.config
        backend.set_config(handle, config)
        if tp_group.rank_in_group == 0:
            _LOG.info(
                "TPSP selected group=%s config=%s coarse_rank=%s seed_rank=%s",
                entry.name,
                config,
                candidate.selected_coarse_rank,
                candidate.selected_seed_rank,
            )
    return None


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


def _scan_chunk(
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
    chunk_min_max: tuple[int, int] | None = None,
    chunk_step_coarse: int = _COARSE_CHUNK_STEP,
    chunk_step_fine: int = _FINE_CHUNK_STEP,
) -> TPSPScanResult:
    """Probe chunks, jointly refine a local grid, and check candidate parity.

    Returns:
        TPSPScanResult with timings and the selected configuration in ``config``.
        The shared group profile determines the token threshold.

    """
    device = backend.device
    chunk_granularity = backend.tpsp_chunk_granularity
    hidden_size = norm.weight.numel()
    input_width = projection.input_size_per_partition
    if any(
        type(step) is not int or step <= 0
        for step in (
            compute_engine_step_coarse,
            compute_engine_step_fine,
            chunk_step_coarse,
            chunk_step_fine,
        )
    ):
        raise ValueError("TPSP scan steps must be positive integers")
    if (
        compute_engine_step_fine > compute_engine_step_coarse
        or chunk_step_fine > chunk_step_coarse
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
    shard_rows = math.ceil(max_batched_tokens / tp_size)

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
    supported_max_chunk_rows = (
        math.ceil(shard_rows / chunk_granularity) * chunk_granularity
    )
    cap_rows = backend.get_max_microchunk_rows(tp_size)
    if cap_rows is not None:
        cap_rows = (
            cap_rows // chunk_granularity * chunk_granularity
            if cap_rows >= chunk_granularity
            else cap_rows
        )
        supported_max_chunk_rows = min(supported_max_chunk_rows, cap_rows)
    supported_min_chunk_rows = min(chunk_granularity, supported_max_chunk_rows)
    min_chunk_rows, max_chunk_rows = (
        chunk_min_max
        if chunk_min_max is not None
        else (supported_min_chunk_rows, supported_max_chunk_rows)
    )
    if (
        type(min_chunk_rows) is not int
        or type(max_chunk_rows) is not int
        or min_chunk_rows < supported_min_chunk_rows
        or max_chunk_rows > supported_max_chunk_rows
        or min_chunk_rows > max_chunk_rows
        or (
            supported_max_chunk_rows >= chunk_granularity
            and (
                min_chunk_rows % chunk_granularity or max_chunk_rows % chunk_granularity
            )
        )
    ):
        raise ValueError("Invalid TPSP chunk range")
    probe_count: int | None = None
    if supported_max_chunk_rows >= chunk_granularity and (
        chunk_step_coarse % chunk_granularity or chunk_step_fine % chunk_granularity
    ):
        raise ValueError("Chunk scan steps must align with chunk granularity")
    sparse_chunks = _inclusive_range(min_chunk_rows, max_chunk_rows, chunk_step_coarse)
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
            coarse_chunks = _inclusive_range(
                min_chunk_rows, max_chunk_rows, chunk_step_fine
            )
        else:
            probe_count = None
            coarse_chunks = sparse_chunks
    else:
        probe_count = None
        coarse_chunks = sparse_chunks
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

    def local_chunks(center: int) -> tuple[int, ...]:
        radius = _FINE_NEIGHBORHOOD_RADIUS * chunk_step_fine
        minimum = max(min_chunk_rows, center - radius)
        maximum = min(max_chunk_rows, center + radius)
        chunks = _inclusive_range(minimum, maximum, chunk_step_fine)
        if center not in chunks:
            chunks = tuple(sorted((*chunks, center)))
        return chunks

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
    for chunk in coarse_chunks:
        targets = coarse_targets()
        if joint_probe_scan:
            assert probe_count is not None
            targets = (probe_count,)
            default_count = backend.get_default_compute_engine_count()
            if default_count is not None and chunk in sparse_chunks:
                targets = tuple(dict.fromkeys((default_count, *targets)))
        configs = backend.get_tuning_configs(
            projection_context,
            chunk,
            group_name=group_name,
            compute_engine_counts=targets,
            compute_engine_step=compute_engine_step_coarse,
        )
        if not configs:
            raise ValueError("TPSP backend returned no tuning configurations")
        coarse_configs.append((chunk, configs))
    coarse_progress = _TPSPProgress(
        group_name or backend.group_name,
        "coarse",
        sum(len(configs) for _, configs in coarse_configs),
        rank,
        started_at=coarse_started,
    )
    for chunk, configs in coarse_configs:
        for config in configs:
            score = score_config_ms(config, screening_inputs, _COARSE_TIMINGS)
            coarse_results.append((config, score, chunk))
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
        probe_config, _, probe_chunk = probe_seed
        probe_counts = fine_targets(
            backend.get_compute_engine_count(probe_config), available_engine_counts
        )
        fine_total = len(local_chunks(probe_chunk)) * len(
            probe_counts if probe_counts is not None else (None,)
        )
        fine_total += int(default_seed is not None)
    else:
        fine_total = sum(
            len(local_chunks(seed_chunk))
            + len(
                fine_targets(
                    backend.get_compute_engine_count(seed_config),
                    available_engine_counts,
                )
                or (None,)
            )
            for seed_config, _, seed_chunk in coarse_top
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
        chunk: int, count: int | None, inputs: _ProjectionInputs
    ) -> tuple[tuple[object, float], ...]:
        pair = (chunk, count)
        if pair in fine_measurements:
            return fine_measurements[pair]
        configs = backend.get_tuning_configs(
            projection_context,
            chunk,
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
        seed_config, _, seed_chunk = probe_seed
        seed_count = backend.get_compute_engine_count(seed_config)
        counts = fine_targets(seed_count, available_engine_counts)
        joint_results: list[tuple[int, object, float]] = []
        for chunk in local_chunks(seed_chunk):
            for count in counts if counts is not None else (None,):
                measured = measure_pair(chunk, count, screening_inputs)
                joint_results.extend(
                    (chunk, config, score) for config, score in measured
                )
                advance_fine_progress()
        _, candidate, candidate_score = min(joint_results, key=lambda result: result[2])
        refined_candidates.append((1, candidate, candidate_score))

        if default_seed is not None:
            default_config, _, default_chunk = default_seed
            default_count = backend.get_default_compute_engine_count()
            assert default_count is not None
            default_results = measure_pair(
                default_chunk, default_count, screening_inputs
            )
            advance_fine_progress()
            candidate, candidate_score = min(
                default_results, key=lambda result: result[1]
            )
            refined_candidates.append((2, candidate, candidate_score))
    else:
        for seed_rank, (seed_config, _, seed_chunk) in enumerate(coarse_top, start=1):
            seed_count = backend.get_compute_engine_count(seed_config)
            chunk_results: list[tuple[int, object, float]] = []
            for chunk in local_chunks(seed_chunk):
                measured = measure_pair(chunk, seed_count, screening_inputs)
                chunk_results.extend(
                    (chunk, config, score) for config, score in measured
                )
                advance_fine_progress()
            selected_chunk = min(chunk_results, key=lambda result: result[2])[0]

            counts = fine_targets(seed_count, available_engine_counts)
            count_results: list[tuple[object, float]] = []
            for count in counts if counts is not None else (None,):
                measured = measure_pair(selected_chunk, count, screening_inputs)
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
