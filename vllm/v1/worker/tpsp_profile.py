# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling for BF16 TP projection / residual RMSNorm chains."""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed import distributed_c10d as c10d

from vllm.platforms.interface import TPSPBackend

_LOG = logging.getLogger(__name__)
_EPS = 1e-5
_TRIALS = 5
_SCREEN_TRIALS = 5


def get_tpsp_backend(group_name: str, device: torch.device) -> TPSPBackend | None:
    from vllm.platforms import current_platform

    backend_cls = current_platform.get_tpsp_backend_cls()
    if backend_cls is None:
        return None
    return backend_cls(group_name, device)


@dataclass(frozen=True)
class SPMeasurement:
    tokens: int
    conventional_ms: float
    sp_ms: float
    lower_benefit_ms: float


@dataclass(frozen=True)
class SPProfile:
    tp_size: int
    hidden_size: int
    max_batched_tokens: int
    status: str
    reason: str
    threshold_tokens: int | None = None
    config: object | None = None
    measurements: tuple[SPMeasurement, ...] = ()
    candidates: tuple[tuple[int, float], ...] = ()
    input_width: int | None = None
    finalists: tuple[tuple[int, float], ...] = ()

    @property
    def enabled(self) -> bool:
        return self.status == "enabled"


class TPSPProjection(nn.Module):
    """A model-owned projection plan, populated after weights are loaded."""

    def __init__(
        self,
        input_width: int,
        hidden_size: int,
        norm_eps: float,
        tp_size: int,
        group_name: str,
    ) -> None:
        super().__init__()
        self.input_width = input_width
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        self.tp_size = tp_size
        self.group_name = group_name
        self.profile: SPProfile | None = None
        self.backend: TPSPBackend | None = None
        self.context: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.profile is not None and self.profile.enabled


def _tpsp_projections(model: nn.Module) -> dict[str, TPSPProjection]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TPSPProjection)
    }


def close_tpsp_projections(model: nn.Module) -> None:
    projections = _tpsp_projections(model)
    backends: dict[int, TPSPBackend] = {}
    for projection in projections.values():
        backend = projection.backend
        if backend is not None:
            backends[id(backend)] = backend
            if projection.context is not None:
                backend.close(projection.context)
        projection.context = None
        projection.backend = None
        if profile := projection.profile:
            projection.profile = SPProfile(
                profile.tp_size,
                profile.hidden_size,
                profile.max_batched_tokens,
                "disabled",
                "TPSP closed",
                input_width=profile.input_width,
            )
    for backend in backends.values():
        backend.close()


def initialize_tpsp(model: nn.Module, max_batched_tokens: int) -> bool:
    """Open TPSP backends and profile projections after loading weights."""
    projections = _tpsp_projections(model)
    if not projections:
        return False
    if all(projection.profile is not None for projection in projections.values()):
        return True
    if any(projection.profile is not None for projection in projections.values()):
        raise RuntimeError("TPSP projections were only partially profiled")
    parameter = next(model.parameters())
    groups: dict[tuple[str, int], dict[str, TPSPProjection]] = {}
    for name, projection in projections.items():
        groups.setdefault((projection.group_name, projection.tp_size), {})[name] = (
            projection
        )
    try:
        for (group_name, tp_size), members in groups.items():
            backend = get_tpsp_backend(group_name, parameter.device)
            for projection in members.values():
                projection.backend = backend
                context = (
                    backend.open(
                        dtype=parameter.dtype,
                        tp_size=tp_size,
                        hidden_size=projection.hidden_size,
                        max_batched_tokens=max_batched_tokens,
                        group_name=group_name,
                        device=parameter.device,
                    )
                    if backend is not None
                    else None
                )
                if context is not None:
                    assert backend is not None
                    projection.context = context
                    projection.profile = backend.profile(
                        tp_size=tp_size,
                        hidden_size=projection.hidden_size,
                        input_width=projection.input_width,
                        max_batched_tokens=max_batched_tokens,
                        norm_eps=projection.norm_eps,
                        time_budget_s=240.0,
                        context=context,
                    )
                else:
                    projection.profile = SPProfile(
                        tp_size,
                        projection.hidden_size,
                        max_batched_tokens,
                        "unsupported",
                        "no fused backend on this device"
                        if backend is None
                        else "fused backend unavailable for this projection",
                        input_width=projection.input_width,
                    )
                if not projection.enabled and projection.context is not None:
                    assert backend is not None
                    backend.close(projection.context)
                    projection.context = None
            if not any(projection.enabled for projection in members.values()):
                if backend is not None:
                    backend.close()
                for projection in members.values():
                    projection.backend = None
            inactive = [
                f"{name}={projection.profile.status}"
                + (
                    f" ({projection.profile.reason})"
                    if projection.profile.reason
                    else ""
                )
                for name, projection in members.items()
                if projection.profile is not None and not projection.enabled
            ]
            if inactive:
                _LOG.warning(
                    "TPSP using regular projection for %s: %s",
                    "all plans" if len(inactive) == len(members) else "inactive plans",
                    ", ".join(inactive),
                )
    except Exception:
        close_tpsp_projections(model)
        raise
    return True


def select_sp_config(profile: SPProfile, current_batched_tokens: int) -> bool:
    """Use the fixed startup choice only for batches within its measured range."""
    if not 1 <= current_batched_tokens <= profile.max_batched_tokens:
        raise ValueError("current_batched_tokens must be within the profiled range")
    return (
        profile.enabled
        and profile.threshold_tokens is not None
        and current_batched_tokens >= profile.threshold_tokens
    )


def _beneficial(measurement: SPMeasurement) -> bool:
    return measurement.lower_benefit_ms > max(0.05, 0.02 * measurement.conventional_ms)


def profile_sp_config(
    backend: TPSPBackend,
    tp_size: int,
    hidden_size: int,
    input_width: int,
    max_batched_tokens: int,
    time_budget_s: float,
    norm_eps: float = _EPS,
    context: Any | None = None,
) -> SPProfile:
    """Time chunks through the backend, check correctness, and find a threshold.

    Screen chunks, retest the two finalists, and compare the winner with the
    conventional path in 128-token steps. Keep the existing confidence-bound
    and maximum-size checks when choosing the threshold.

    Returns:
        SPProfile with candidate timings and per-size measurements. An enabled
        profile contains the selected chunk in ``config`` and the first
        beneficial size in ``threshold_tokens``; other statuses have no config.

    """
    if backend._closed:
        raise RuntimeError("TPSP backend is closed")
    context = backend._profile_context(context)
    device = backend.device
    group_name = backend.group_name
    chunk_granularity = backend.ops.tpsp_chunk_granularity
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
        return SPProfile(
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

    def inputs(tokens: int):
        generator = torch.Generator(device=device).manual_seed(1831 + rank)
        a = torch.randn(
            tokens,
            input_width,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        b = torch.randn(
            input_width,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ) / math.sqrt(input_width)
        weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
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
        return (
            (a, b, b.T.contiguous()),
            weight,
            residual,
            padded.narrow(0, rank * rows, rows).contiguous(),
            residual.clone(),
        )

    def project(data, candidate):
        (a, b, linear_weight), weight, residual, local_residual, _ = data
        if candidate is not None:
            return backend.fused_gemm_rs_norm_ag(
                a,
                b,
                weight,
                local_residual,
                norm_eps,
                candidate,
                context=context,
            )[2]
        partial = F.linear(a, linear_weight)
        full = partial.clone()
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=group)
        torch.ops._C.fused_add_rms_norm(full, residual, weight, norm_eps)
        return full

    def measure(data, candidate) -> float:
        dist.barrier(group=group)
        if candidate is None:
            data[2].copy_(data[4])
        torch.accelerator.synchronize(device)
        start = time.perf_counter()
        result = project(data, candidate)
        torch.accelerator.synchronize(device)
        elapsed = torch.tensor(
            [(time.perf_counter() - start) * 1000], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        value = elapsed.item()
        del result
        return value

    def inconclusive(reason: str, measurements=(), candidate_results=()) -> SPProfile:
        if rank == 0:
            _LOG.warning("TPSP startup profile: status=inconclusive reason=%s", reason)
        return SPProfile(
            tp_size,
            hidden_size,
            max_batched_tokens,
            "inconclusive",
            reason,
            measurements=tuple(measurements),
            candidates=tuple(candidate_results),
            input_width=input_width,
        )

    data = inputs(max_batched_tokens)
    samples: dict[int, list[float]] = {}

    def screen_chunk(candidate: int, results: dict[int, list[float]]) -> float | None:
        results[candidate] = []
        measure(data, candidate)
        if expired():
            return None
        for _ in range(_SCREEN_TRIALS):
            results[candidate].append(measure(data, candidate))
            if expired():
                return None
        score = torch.tensor(
            [sorted(results[candidate])[1]], dtype=torch.float64, device=device
        )
        dist.all_reduce(score, op=dist.ReduceOp.MAX, group=group)
        return float(score.item())

    candidates: list[int] = []
    best = float("inf")
    without_improvement = 0
    for chunk in range(
        chunk_granularity,
        math.ceil(shard_rows / chunk_granularity) * chunk_granularity + 1,
        chunk_granularity,
    ):
        score = screen_chunk(chunk, samples)
        if score is None:
            return inconclusive("screening time budget exceeded")
        candidates.append(chunk)
        if score < best:
            best = score
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement == 32:
            break
    scores = torch.tensor(
        [sorted(samples[candidate])[1] for candidate in candidates],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(scores, op=dist.ReduceOp.MAX, group=group)
    candidate_results = tuple(
        (candidate, float(score))
        for candidate, score in zip(candidates, scores.tolist())
    )
    finalists = sorted(candidate_results, key=lambda result: result[1])[:2]
    retested: dict[int, list[float]] = {}
    for finalist, _ in finalists:
        if screen_chunk(finalist, retested) is None:
            return inconclusive(
                "finalist time budget exceeded", candidate_results=candidate_results
            )
    final_scores = torch.tensor(
        [sorted(retested[finalist])[1] for finalist, _ in finalists],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(final_scores, op=dist.ReduceOp.MAX, group=group)
    finalist_results = tuple(
        (finalist, float(score))
        for (finalist, _), score in zip(finalists, final_scores.tolist())
    )
    candidate = min(finalist_results, key=lambda item: item[1])[0]
    data = None

    check_data = inputs(min(max_batched_tokens, tp_size * 5 + 1))
    conventional = project(check_data, None)
    native = project(check_data, candidate)
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(native, conventional, rtol=0.02, atol=0.05)
    del check_data, conventional, native

    def measure_size(tokens: int) -> SPMeasurement | None:
        data = inputs(tokens)
        measure(data, None)
        measure(data, candidate)
        paired: list[float] = []
        conventional: list[float] = []
        sp: list[float] = []
        for trial in range(_TRIALS):
            order = (None, candidate) if trial % 2 == 0 else (candidate, None)
            times = {item: measure(data, item) for item in order}
            conventional.append(times[None])
            sp.append(times[candidate])
            paired.append(times[None] - times[candidate])
            if expired():
                return None
        lower = statistics.mean(paired) - 2.776 * statistics.stdev(paired) / math.sqrt(
            _TRIALS
        )
        return SPMeasurement(
            tokens, statistics.median(conventional), statistics.median(sp), lower
        )

    measurements: list[SPMeasurement] = []
    threshold = None
    sizes = list(range(128, max_batched_tokens + 1, 128))
    if not sizes or sizes[-1] != max_batched_tokens:
        sizes.append(max_batched_tokens)
    streak = 0
    for tokens in sizes:
        result = measure_size(tokens)
        if result is None:
            return inconclusive(
                "measurement time budget exceeded", measurements, candidate_results
            )
        measurements.append(result)
        streak = streak + 1 if _beneficial(result) else 0
        if streak == min(3, len(sizes)):
            threshold = measurements[-streak].tokens
            break
    if threshold is not None and sizes[-1] != tokens:
        maximum = measure_size(sizes[-1])
        if maximum is None:
            return inconclusive(
                "maximum-size measurement time budget exceeded",
                measurements,
                candidate_results,
            )
        measurements.append(maximum)
        if not _beneficial(maximum):
            threshold = None
    if (
        threshold is None
        and measurements[-1].tokens == max_batched_tokens
        and measurements[-1].lower_benefit_ms > 0
    ):
        threshold = max_batched_tokens
    status = "enabled" if threshold is not None else "disabled"
    reason = (
        ""
        if threshold is not None
        else "no sustained benefit over the conventional projection"
    )
    profile = SPProfile(
        tp_size,
        hidden_size,
        max_batched_tokens,
        status,
        reason,
        threshold_tokens=threshold,
        config=candidate if threshold else None,
        measurements=tuple(measurements),
        candidates=tuple(candidate_results),
        input_width=input_width,
        finalists=finalist_results,
    )
    if rank == 0:
        _LOG.warning("TPSP startup profile: norm_eps=%s %s", norm_eps, profile)
    return profile
