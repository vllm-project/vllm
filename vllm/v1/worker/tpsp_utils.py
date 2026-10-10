# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 TPSP execution contexts and shared profiling utilities."""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import distributed_c10d as c10d

_LOG = logging.getLogger(__name__)
_EPS = 1e-5
_TRIALS = 5
_SCREEN_TRIALS = 5


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
    lower_benefit_ms: float


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
    candidates: tuple[tuple[int, float], ...] = ()
    input_width: int | None = None
    finalists: tuple[tuple[int, float], ...] = ()

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
    padded = torch.zeros(
        tp_size * rows, hidden_size, dtype=torch.bfloat16, device=device
    )
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


def profile_tpsp(
    backend: Any,
    ops_groups: Sequence[TPSPOpsGroup],
    max_batched_tokens: int,
) -> TPSPContext | None:
    """Choose a chunk for each group, then one shared token threshold.

    Each group gets its own backend context and chunk scan. The threshold
    scan compares the sum of the fused groups against their normal paths.
    If any scan is inconclusive or finds no benefit, use the normal path.
    """
    if backend is None:
        raise ValueError("TPSP profiling requires a backend")
    if not ops_groups or any(not entry.name for entry in ops_groups):
        raise ValueError("TPSP profiling requires named ops groups")
    if len({entry.name for entry in ops_groups}) != len(ops_groups):
        raise ValueError("TPSP ops group names must be unique")

    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    hidden_size = ops_groups[0].norm.weight.numel()
    if any(entry.norm.weight.numel() != hidden_size for entry in ops_groups):
        raise ValueError("TPSP ops groups require matching norm hidden sizes")

    def disabled(reason: str) -> TPSPContext | None:
        _LOG.warning("TPSP using regular forward: %s", reason)
        return None

    parameter = ops_groups[0].projection.weight
    started = time.perf_counter()
    enabled = False
    threshold = None
    handles: dict[str, object] = {}
    try:
        for entry in ops_groups:
            handle = backend.open(
                dtype=parameter.dtype,
                tp_size=group.world_size,
                hidden_size=hidden_size,
                max_batched_tokens=max_batched_tokens,
                group_name=group.device_group.group_name,
                device=parameter.device,
            )
            if handle is None:
                return disabled(f"fused backend unavailable for {entry.name}")
            handles[entry.name] = handle
            candidate = backend.profile_projection(
                handle,
                projection=entry.projection,
                norm=entry.norm,
                tp_size=group.world_size,
                hidden_size=hidden_size,
                input_width=entry.projection.input_size_per_partition,
                max_batched_tokens=max_batched_tokens,
                norm_eps=entry.norm.variance_epsilon,
                time_budget_s=240.0,
            )
            if candidate.status != "candidate" or candidate.config is None:
                return disabled(
                    f"{entry.name} chunk selection "
                    f"{candidate.status}: {candidate.reason}"
                )

        measurement = scan_threshold(backend, ops_groups, handles, max_batched_tokens)
        if not measurement.enabled:
            return disabled(
                f"projection profile {measurement.status}: {measurement.reason}"
            )
        threshold = measurement.threshold_tokens
        if threshold is None or not 1 <= threshold <= max_batched_tokens:
            raise RuntimeError("TPSP has an invalid enabled profile")
        enabled = True
        return TPSPContext(
            TPSPProfile(True, threshold, max_batched_tokens),
            backend,
            handles,
        )
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


def _beneficial(measurement: TPSPMeasurement) -> bool:
    return measurement.lower_benefit_ms > max(0.05, 0.02 * measurement.conventional_ms)


@torch.inference_mode()
def scan_threshold(
    backend: Any,
    ops_groups: Sequence[TPSPOpsGroup],
    handles: dict[str, object],
    max_batched_tokens: int,
) -> TPSPScanResult:
    """Compare the sum of the group timings with their normal paths."""
    group = c10d._resolve_process_group(backend.group_name)
    tp_size = dist.get_world_size(group)
    hidden_size = ops_groups[0].norm.weight.numel()
    rank = dist.get_rank(group)
    device = backend.device
    entries = tuple(
        (handles[entry.name], entry.projection, entry.norm) for entry in ops_groups
    )
    deadline = time.monotonic() + 240.0

    def expired() -> bool:
        flag = torch.tensor(
            [time.monotonic() >= deadline], dtype=torch.int32, device=device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        return bool(flag.item())

    def measure(
        handle: object,
        projection: nn.Module,
        norm: nn.Module,
        inputs: _ProjectionInputs,
        fused: bool,
    ) -> float:
        if not fused:
            inputs.restore_residual()
        dist.barrier(group=group)
        torch.accelerator.synchronize(device)
        start = time.perf_counter()
        if fused:
            result = _run_fused_projection(backend, handle, projection, norm, inputs)
        else:
            result = _run_conventional_projection(projection, norm, inputs)
        torch.accelerator.synchronize(device)
        elapsed = torch.tensor(
            [(time.perf_counter() - start) * 1000], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        del result
        return elapsed.item()

    def measure_size(tokens: int) -> TPSPMeasurement | None:
        data = tuple(
            _projection_inputs(tokens, projection, hidden_size, tp_size, rank, device)
            for handle, projection, norm in entries
        )
        for (handle, projection, norm), inputs in zip(entries, data):
            measure(handle, projection, norm, inputs, False)
            measure(handle, projection, norm, inputs, True)
        conventional: list[float] = []
        fused: list[float] = []
        for trial in range(_TRIALS):
            order = (False, True) if trial % 2 == 0 else (True, False)
            timings = {False: 0.0, True: 0.0}
            for (handle, projection, norm), inputs in zip(entries, data):
                for enabled in order:
                    timings[enabled] += measure(
                        handle, projection, norm, inputs, enabled
                    )
            conventional.append(timings[False])
            fused.append(timings[True])
            if expired():
                return None
        savings = [normal - sp for normal, sp in zip(conventional, fused)]
        lower = statistics.mean(savings) - 2.776 * statistics.stdev(
            savings
        ) / math.sqrt(_TRIALS)
        return TPSPMeasurement(
            tokens, statistics.median(conventional), statistics.median(fused), lower
        )

    sizes = list(range(128, max_batched_tokens + 1, 128))
    if not sizes or sizes[-1] != max_batched_tokens:
        sizes.append(max_batched_tokens)
    measurements: list[TPSPMeasurement] = []
    threshold = None
    streak = 0
    for tokens in sizes:
        result = measure_size(tokens)
        if result is None:
            break
        measurements.append(result)
        streak = streak + 1 if _beneficial(result) else 0
        if streak == min(3, len(sizes)):
            threshold = measurements[-streak].tokens
            break
    if threshold is not None and sizes[-1] != tokens:
        maximum = measure_size(sizes[-1])
        if maximum is None:
            threshold = None
        else:
            measurements.append(maximum)
            if not _beneficial(maximum):
                threshold = None
    if (
        threshold is None
        and measurements
        and measurements[-1].tokens == max_batched_tokens
        and measurements[-1].lower_benefit_ms > 0
    ):
        threshold = max_batched_tokens
    status = "enabled" if threshold is not None else "disabled"
    reason = "" if threshold is not None else "no sustained summed TPSP benefit"
    if expired():
        status, reason, threshold = (
            "inconclusive",
            "projection profiling time budget exceeded",
            None,
        )
    profile = TPSPScanResult(
        tp_size,
        hidden_size,
        max_batched_tokens,
        status,
        reason,
        threshold_tokens=threshold,
        measurements=tuple(measurements),
    )
    if rank == 0:
        _LOG.info("TPSP projection startup profile: %s", profile)
    return profile


def scan_chunk(
    backend: Any,
    projection_context: object,
    projection: nn.Module,
    norm: nn.Module,
    tp_size: int,
    hidden_size: int,
    input_width: int,
    max_batched_tokens: int,
    time_budget_s: float,
    norm_eps: float = _EPS,
) -> TPSPScanResult:
    """Screen chunks, retest the finalists, and check numerical parity.

    Returns:
        TPSPScanResult with candidate timings and the selected chunk in ``config``.
        The paired projection profile determines the token threshold.

    """
    if backend._closed:
        raise RuntimeError("TPSP backend is closed")
    if projection_context is None:
        raise ValueError("TPSP requires a projection context")
    device = backend.device
    group_name = backend.group_name
    chunk_granularity = backend.tpsp_chunk_granularity
    if type(chunk_granularity) is not int or chunk_granularity <= 0:
        raise ValueError("TPSP chunk granularity must be a positive integer")
    if time_budget_s <= 0:
        raise ValueError("time_budget_s must be positive")

    if (
        tp_size < 2
        or hidden_size <= 0
        or hidden_size % tp_size
        or max_batched_tokens < 1
    ):
        return TPSPScanResult(
            tp_size,
            hidden_size,
            max_batched_tokens,
            "unsupported",
            "requires supported TP, divisible hidden size and positive "
            "max_batched_tokens",
        )
    if input_width <= 0:
        raise ValueError("input_width must be positive")
    if norm_eps <= 0:
        raise ValueError("norm_eps must be positive")
    group = c10d._resolve_process_group(group_name)
    if dist.get_world_size(group) != tp_size:
        raise ValueError("group_name and tp_size disagree")
    rank = dist.get_rank(group)
    shard_rows = math.ceil(max_batched_tokens / tp_size)
    deadline = time.monotonic() + time_budget_s

    def expired() -> bool:
        flag = torch.tensor(
            [time.monotonic() >= deadline], dtype=torch.int32, device=device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        return bool(flag.item())

    def make_inputs(tokens: int) -> _ProjectionInputs:
        return _projection_inputs(
            tokens, projection, hidden_size, tp_size, rank, device
        )

    def run_fused(inputs: _ProjectionInputs, chunk: int) -> torch.Tensor:
        return _run_fused_projection(
            backend, projection_context, projection, norm, inputs, chunk
        )

    def time_chunk_ms(inputs: _ProjectionInputs, chunk: int) -> float:
        dist.barrier(group=group)
        torch.accelerator.synchronize(device)
        start = time.perf_counter()
        result = run_fused(inputs, chunk)
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

    # Stage 1: screen micro-chunk sizes for this projection at maximum load.
    screening_inputs = make_inputs(max_batched_tokens)

    def score_chunk_ms(chunk: int, inputs: _ProjectionInputs) -> float | None:
        """Warm up, then score the second-fastest of five slowest-rank trials."""
        time_chunk_ms(inputs, chunk)
        if expired():
            return None
        timings = []
        for _ in range(_SCREEN_TRIALS):
            timings.append(time_chunk_ms(inputs, chunk))
            if expired():
                return None
        return sorted(timings)[1]

    candidate_results: list[tuple[int, float]] = []
    best = float("inf")
    without_improvement = 0
    max_chunk_rows = math.ceil(shard_rows / chunk_granularity) * chunk_granularity
    if backend.tpsp_max_microchunk_tokens is not None:
        cap_rows = math.ceil(backend.tpsp_max_microchunk_tokens / tp_size)
        max_chunk_rows = min(
            max_chunk_rows,
            cap_rows // chunk_granularity * chunk_granularity
            if cap_rows >= chunk_granularity
            else cap_rows,
        )
    for chunk in range(
        min(chunk_granularity, max_chunk_rows),
        max_chunk_rows + 1,
        chunk_granularity,
    ):
        score = score_chunk_ms(chunk, screening_inputs)
        if score is None:
            return inconclusive("screening time budget exceeded")
        candidate_results.append((chunk, score))
        if score < best:
            best = score
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement == 32:
            break
    # Stage 2: retest the two fastest chunks before committing to one.
    finalists = sorted(candidate_results, key=lambda result: result[1])[:2]
    finalist_results: list[tuple[int, float]] = []
    for finalist, _ in finalists:
        score = score_chunk_ms(finalist, screening_inputs)
        if score is None:
            return inconclusive(
                "finalist time budget exceeded", candidate_results=candidate_results
            )
        finalist_results.append((finalist, score))
    candidate = min(finalist_results, key=lambda item: item[1])[0]
    del screening_inputs

    # Stage 3: verify numerical parity before passing this chunk to the pair.
    check_inputs = make_inputs(min(max_batched_tokens, tp_size * 5 + 1))
    conventional = _run_conventional_projection(projection, norm, check_inputs)
    native = run_fused(check_inputs, candidate)
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(native, conventional, rtol=0.02, atol=0.05)
    del check_inputs, conventional, native
    return TPSPScanResult(
        tp_size,
        hidden_size,
        max_batched_tokens,
        "candidate",
        "",
        config=candidate,
        candidates=tuple(candidate_results),
        input_width=input_width,
        finalists=tuple(finalist_results),
    )
