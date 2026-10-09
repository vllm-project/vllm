# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling for BF16 TP projection / residual RMSNorm chains."""

from __future__ import annotations

import logging
import math
import os
import statistics
import time
from collections.abc import Callable
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


@dataclass(frozen=True)
class ChunkConfig:
    microchunk_tokens: int
    comm_mode: str = "nccl"


@dataclass(frozen=True)
class TPSPProjectionContext:
    transport: Any
    config: object


class ProfilingTPSPBackend(TPSPBackend):
    """Profile a fused projection with platform-owned resources."""

    ops: Any = None
    profiles_transport_modes = False

    def __init__(self, group_name: str, device: torch.device):
        self.group_name = group_name
        self.device = device
        self.chunk_granularity = getattr(self.ops, "tpsp_chunk_granularity", 64)
        if type(self.chunk_granularity) is not int or self.chunk_granularity <= 0:
            raise ValueError("TPSP chunk granularity must be a positive integer")
        self._closed = False

    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> Any | None:
        return None

    @classmethod
    def _valid_open(
        cls,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> bool:
        if (
            dtype != torch.bfloat16
            or tp_size < 2
            or hidden_size <= 0
            or hidden_size % tp_size
            or max_batched_tokens <= 0
        ):
            _LOG.warning(
                "TPSP unavailable for dtype=%s tp_size=%s hidden_size=%s",
                dtype,
                tp_size,
                hidden_size,
            )
            return False
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::fused_add_rms_norm", device.type.upper()
        ):
            _LOG.warning("TPSP unavailable: %s fused_add_rms_norm is missing", device)
            return False
        group = c10d._resolve_process_group(group_name)
        if dist.get_world_size(group) != tp_size:
            _LOG.warning("TPSP unavailable: group_name and tp_size disagree")
            return False
        return True

    def _profile_context(self, context: Any | None) -> Any | None:
        return context

    def profile(
        self,
        *,
        tp_size: int,
        hidden_size: int,
        input_width: int,
        max_batched_tokens: int,
        norm_eps: float,
        sharded_residual: bool,
        time_budget_s: float,
        context: Any | None = None,
    ) -> SPProfile:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        context = self._profile_context(context)
        return profile_sp_config(
            self,
            tp_size=tp_size,
            hidden_size=hidden_size,
            input_width=input_width,
            max_batched_tokens=max_batched_tokens,
            group_name=self.group_name,
            norm_eps=norm_eps,
            sharded_residual=sharded_residual,
            time_budget_s=time_budget_s,
            context=context,
        )

    def device_synchronize(self) -> None:
        getattr(torch, self.device.type).synchronize(self.device)

    def close(self, context: Any | None = None) -> None:
        if not self._closed:
            self._closed = True


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
    pool_mb: int | None = None
    measurements: tuple[SPMeasurement, ...] = ()
    candidates: tuple[tuple[ChunkConfig, float], ...] = ()
    input_width: int | None = None
    norm_eps: float = _EPS
    finalists: tuple[tuple[ChunkConfig, float], ...] = ()
    gather_sharded_residual: bool = False

    @property
    def enabled(self) -> bool:
        return self.status == "enabled"


@dataclass(frozen=True)
class TPSPShape:
    input_width: int
    hidden_size: int
    norm_eps: float
    sharded_residual: bool


class TPSPProjection(nn.Module):
    """A model-owned projection plan, populated after weights are loaded."""

    def __init__(self, shape: TPSPShape, tp_size: int, group_name: str) -> None:
        super().__init__()
        self.shape = shape
        self.tp_size = tp_size
        self.group_name = group_name
        self.profile: SPProfile | None = None
        self.backend: TPSPBackend | None = None
        self.context: Any | None = None

    @property
    def active(self) -> bool:
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
                norm_eps=profile.norm_eps,
                gather_sharded_residual=profile.gather_sharded_residual,
            )
    for backend in backends.values():
        backend.close()


def profile_tpsp_projections(model: nn.Module, max_batched_tokens: int) -> bool:
    """Profile TPSP modules after loading weights, before memory profiling."""
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
                shape = projection.shape
                context = (
                    backend.open(
                        dtype=parameter.dtype,
                        tp_size=tp_size,
                        hidden_size=shape.hidden_size,
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
                        hidden_size=shape.hidden_size,
                        input_width=shape.input_width,
                        max_batched_tokens=max_batched_tokens,
                        norm_eps=shape.norm_eps,
                        sharded_residual=shape.sharded_residual,
                        time_budget_s=240.0,
                        context=context,
                    )
                else:
                    projection.profile = SPProfile(
                        tp_size,
                        shape.hidden_size,
                        max_batched_tokens,
                        "unsupported",
                        "no fused backend on this device"
                        if backend is None
                        else "neither NCCL nor P2P is usable",
                        input_width=shape.input_width,
                        norm_eps=shape.norm_eps,
                        gather_sharded_residual=shape.sharded_residual,
                    )
                if not projection.active and projection.context is not None:
                    assert backend is not None
                    backend.close(projection.context)
                    projection.context = None
            if not any(projection.active for projection in members.values()):
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
                if projection.profile is not None and not projection.active
            ]
            if len(inactive) == len(members):
                _LOG.warning(
                    "TPSP using standard forward; projection plans: %s",
                    ", ".join(inactive),
                )
            elif inactive:
                _LOG.warning(
                    "TPSP using regular projection for inactive plans: %s",
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


def _token_sizes(max_tokens: int) -> list[int]:
    sizes = list(range(128, max_tokens + 1, 128))
    if not sizes or sizes[-1] != max_tokens:
        sizes.append(max_tokens)
    return sizes


def _beneficial(measurement: SPMeasurement) -> bool:
    return measurement.lower_benefit_ms > max(0.05, 0.02 * measurement.conventional_ms)


def _screen_score(samples: list[float]) -> float:
    return sorted(samples)[1]


def _search_chunks(
    shard_rows: int, score: Callable[[int], float | None], step: int = 64
) -> list[int] | None:
    chunks = []
    best = float("inf")
    without_improvement = 0
    for chunk in range(step, math.ceil(shard_rows / step) * step + 1, step):
        value = score(chunk)
        if value is None:
            return None
        chunks.append(chunk)
        if value < best:
            best = value
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement == 32:
            break
    return chunks


def profile_sp_config(
    backend: ProfilingTPSPBackend,
    tp_size: int,
    hidden_size: int,
    input_width: int,
    max_batched_tokens: int,
    group_name: str,
    time_budget_s: float,
    norm_eps: float = _EPS,
    sharded_residual: bool = False,
    context: Any | None = None,
) -> SPProfile:
    """Scan chunks with the fused op's default communication configuration."""
    if time_budget_s <= 0:
        raise ValueError("time_budget_s must be positive")

    def unsupported(reason: str) -> SPProfile:
        return SPProfile(
            tp_size, hidden_size, max_batched_tokens, "unsupported", reason
        )

    if (
        tp_size < 2
        or (backend.device.type != "cuda" and tp_size > 8)
        or hidden_size <= 0
        or hidden_size % tp_size
        or max_batched_tokens < 1
    ):
        return unsupported(
            "requires supported TP, divisible hidden size and positive "
            "max_batched_tokens"
        )
    if input_width <= 0:
        raise ValueError("input_width must be positive")
    if norm_eps <= 0:
        raise ValueError("norm_eps must be positive")
    group = c10d._resolve_process_group(group_name)
    if dist.get_world_size(group) != tp_size:
        raise ValueError("group_name and tp_size disagree")
    device = backend.device
    if device.type == "cuda" and context is None:
        raise ValueError("CUDA TPSP requires a projection context")
    rank = dist.get_rank(group)
    shard_rows = math.ceil(max_batched_tokens / tp_size)
    pool_mb = None
    if device.type == "xpu":
        output_mb = math.ceil(max_batched_tokens * hidden_size * 2 / 2**20)
        minimum_pool_mb = math.ceil((2 * output_mb + 256) / 512) * 512
        configured_pool = os.environ.setdefault(
            "ASYNC_TP_OUTPUT_POOL_MB", str(minimum_pool_mb)
        )
        pool_mb = int(configured_pool)
        if pool_mb < minimum_pool_mb:
            raise RuntimeError(
                f"ASYNC_TP_OUTPUT_POOL_MB={pool_mb} is below the profile minimum "
                f"{minimum_pool_mb}"
            )
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
        ) / (math.sqrt(input_width) if device.type == "cuda" else 8)
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
            residual.clone() if device.type == "cuda" else residual,
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
        # CUDA must be compared with regular TP Llama, which keeps a full residual.
        if sharded_residual and device.type != "cuda":
            rows = local_residual.size(0)
            gathered_residual = torch.empty(
                (tp_size * rows, hidden_size), device=device, dtype=a.dtype
            )
            dist.all_gather_into_tensor(gathered_residual, local_residual, group=group)
            full_residual = gathered_residual[: a.size(0)].contiguous()
        else:
            full_residual = residual if device.type == "cuda" else residual.clone()
        torch.ops._C.fused_add_rms_norm(full, full_residual, weight, norm_eps)
        return full

    def measure(data, candidate) -> float:
        dist.barrier(group=group)
        if device.type == "cuda" and candidate is None:
            data[2].copy_(data[4])
        backend.device_synchronize()
        start = time.perf_counter()
        result = project(data, candidate)
        backend.device_synchronize()
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
            pool_mb=pool_mb,
            measurements=tuple(measurements),
            candidates=tuple(candidate_results),
            input_width=input_width,
            norm_eps=norm_eps,
            gather_sharded_residual=sharded_residual,
        )

    data = inputs(max_batched_tokens)
    samples: dict[ChunkConfig, list[float]] = {}

    def screen_chunk(
        candidate: ChunkConfig, results: dict[ChunkConfig, list[float]]
    ) -> float | None:
        results[candidate] = []
        measure(data, candidate)
        if expired():
            return None
        for _ in range(_SCREEN_TRIALS):
            results[candidate].append(measure(data, candidate))
            if expired():
                return None
        score = torch.tensor(
            [_screen_score(results[candidate])], dtype=torch.float64, device=device
        )
        dist.all_reduce(score, op=dist.ReduceOp.MAX, group=group)
        return float(score.item())

    modes = (
        [mode for mode in ("nccl", "p2p") if mode in context.modes]
        if context is not None and backend.profiles_transport_modes
        else ["nccl"]
    )
    forced_mode = os.environ.get("VLLM_TPSP_COMM_MODE")
    if forced_mode is not None:
        if forced_mode not in modes:
            raise ValueError(
                f"VLLM_TPSP_COMM_MODE={forced_mode!r} is not an available "
                f"TPSP transport: {modes}"
            )
        modes = [forced_mode]
    candidates: list[ChunkConfig] = []
    for mode in modes:

        def score_chunk(chunk: int, mode: str = mode) -> float | None:
            return screen_chunk(ChunkConfig(chunk, mode), samples)

        chunks = _search_chunks(
            shard_rows,
            score_chunk,
            backend.chunk_granularity,
        )
        if chunks is None:
            return inconclusive("screening time budget exceeded")
        candidates.extend(ChunkConfig(chunk, mode) for chunk in chunks)
    scores = torch.tensor(
        [_screen_score(samples[candidate]) for candidate in candidates],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(scores, op=dist.ReduceOp.MAX, group=group)
    candidate_results = tuple(
        (candidate, float(score))
        for candidate, score in zip(candidates, scores.tolist())
    )
    finalists = [
        item
        for mode in modes
        for item in sorted(
            (result for result in candidate_results if result[0].comm_mode == mode),
            key=lambda result: result[1],
        )[:2]
    ]
    retested: dict[ChunkConfig, list[float]] = {}
    for candidate, _ in finalists:
        if screen_chunk(candidate, retested) is None:
            return inconclusive(
                "finalist time budget exceeded", candidate_results=candidate_results
            )
    final_scores = torch.tensor(
        [_screen_score(retested[candidate]) for candidate, _ in finalists],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(final_scores, op=dist.ReduceOp.MAX, group=group)
    finalist_results = tuple(
        (candidate, float(score))
        for (candidate, _), score in zip(finalists, final_scores.tolist())
    )
    candidate = min(finalist_results, key=lambda item: item[1])[0]
    data = None

    check_data = inputs(min(max_batched_tokens, tp_size * 5 + 1))
    conventional = project(check_data, None)
    native = project(check_data, candidate)
    backend.device_synchronize()
    if device.type == "cuda":
        torch.testing.assert_close(native, conventional, rtol=0.02, atol=0.05)
    else:
        torch.testing.assert_close(native, conventional, rtol=0.01, atol=0.02)
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
    sizes = _token_sizes(max_batched_tokens)
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
    if device.type == "cuda" and threshold is not None and sizes[-1] != tokens:
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
        device.type == "cuda"
        and threshold is None
        and measurements[-1].tokens == max_batched_tokens
        and measurements[-1].lower_benefit_ms > 0
    ):
        threshold = max_batched_tokens
    status = "enabled" if threshold is not None else "disabled"
    reason = (
        ""
        if threshold is not None
        else (
            "no sustained benefit over the conventional projection"
            if device.type == "cuda"
            else "no sustained benefit with 128-token steps"
        )
    )
    profile = SPProfile(
        tp_size,
        hidden_size,
        max_batched_tokens,
        status,
        reason,
        threshold,
        (
            TPSPProjectionContext(context, candidate)
            if context is not None and status == "enabled"
            else candidate
            if status == "enabled"
            else None
        ),
        pool_mb,
        tuple(measurements),
        candidate_results,
        input_width,
        norm_eps,
        finalist_results,
        sharded_residual,
    )
    if rank == 0:
        _LOG.warning(
            "TPSP startup profile: status=%s threshold=%s chunk=%s "
            "input_width=%s norm_eps=%s sharded_residual=%s pool_mb=%s "
            "candidates=%s finalists=%s measurements=%s reason=%s",
            status,
            threshold,
            candidate,
            input_width,
            norm_eps,
            sharded_residual,
            pool_mb,
            candidate_results,
            finalist_results,
            measurements,
            reason,
        )
    return profile
