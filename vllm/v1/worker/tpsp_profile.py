# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling for BF16 TP projection / residual RMSNorm chains."""

import importlib
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
from torch.distributed import distributed_c10d as c10d

_LOG = logging.getLogger(__name__)
_EPS = 1e-5
_TRIALS = 5
_SCREEN_TRIALS = 5


@dataclass(frozen=True)
class ChunkConfig:
    microchunk_tokens: int


class TPSPBackend:
    """Own the fused operator and its platform-specific profile configuration.

    A native ``profile_tpsp_config`` returns an object with
    ``threshold_tokens`` and opaque ``config`` attributes. It requires a
    matching native ``fused_matmul_reduce_scatter_norm_all_gather_profiled``
    entry point. When the native profiler is absent, vLLM scans chunks
    through the standard fused op.
    """

    def __init__(self, ops: Any, group_name: str, device: torch.device):
        self.ops = ops
        self.group_name = group_name
        self.device = device
        self.chunk_granularity = getattr(ops, "tpsp_chunk_granularity", 64)
        if type(self.chunk_granularity) is not int or self.chunk_granularity <= 0:
            raise ValueError("TPSP chunk granularity must be a positive integer")
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        group_name: str,
        device: torch.device,
    ) -> "TPSPBackend | None":
        if (
            dtype != torch.bfloat16
            or not 2 <= tp_size <= 8
            or hidden_size <= 0
            or hidden_size % tp_size
        ):
            _LOG.warning(
                "TPSP unavailable for dtype=%s tp_size=%s hidden_size=%s",
                dtype,
                tp_size,
                hidden_size,
            )
            return None
        try:
            ops = importlib.import_module("deep_symm.async_tp")
        except ModuleNotFoundError as exc:
            if exc.name not in ("deep_symm", "deep_symm.async_tp"):
                raise
            _LOG.warning("TPSP unavailable: %s", exc)
            return None
        if device.type not in getattr(ops, "tpsp_supported_devices", ("xpu",)):
            _LOG.warning("TPSP fused projection does not support %s", device.type)
            return None
        if not hasattr(ops._C, "fused_matmul_reduce_scatter_norm_all_gather"):
            _LOG.warning("TPSP native fused projection is unavailable")
            return None
        if device.type == "xpu":
            import vllm_xpu_kernels._C  # noqa: F401

        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::fused_add_rms_norm", device.type.upper()
        ):
            raise RuntimeError(f"vLLM {device.type} fused_add_rms_norm is unavailable")
        group = c10d._resolve_process_group(group_name)
        if dist.get_world_size(group) != tp_size:
            raise ValueError("TPSP group_name and tp_size disagree")
        return cls(ops, group_name, device)

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
    ) -> "SPProfile":
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        kwargs = dict(
            tp_size=tp_size,
            hidden_size=hidden_size,
            input_width=input_width,
            max_batched_tokens=max_batched_tokens,
            group_name=self.group_name,
            norm_eps=norm_eps,
            sharded_residual=sharded_residual,
            time_budget_s=time_budget_s,
        )
        external = getattr(self.ops._C, "profile_tpsp_config", None)
        if external is not None:
            if not hasattr(
                self.ops._C,
                "fused_matmul_reduce_scatter_norm_all_gather_profiled",
            ):
                raise RuntimeError(
                    "TPSP external profiler requires a profiled fused op"
                )
            result = external(**kwargs)
            threshold = result.threshold_tokens
            config = result.config
            if threshold is not None and (
                type(threshold) is not int
                or not 1 <= threshold <= max_batched_tokens
                or config is None
            ):
                raise ValueError("TPSP external profiler returned an invalid plan")
            if threshold is None and config is not None:
                raise ValueError("TPSP disabled plan must not contain a config")
            group = c10d._resolve_process_group(self.group_name)
            shared = torch.tensor(
                [threshold or 0, threshold or 0], dtype=torch.int64, device=self.device
            )
            dist.all_reduce(shared[:1], op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(shared[1:], op=dist.ReduceOp.MAX, group=group)
            if shared[0] != shared[1]:
                raise ValueError(
                    "TPSP external profiler returned inconsistent thresholds"
                )
            return SPProfile(
                tp_size,
                hidden_size,
                max_batched_tokens,
                "enabled" if threshold is not None else "disabled",
                "",
                threshold_tokens=threshold,
                config=config,
                input_width=input_width,
                norm_eps=norm_eps,
                gather_sharded_residual=sharded_residual,
            )
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
        )

    def fused(
        self,
        a,
        b,
        weight,
        residual,
        eps: float,
        config: object,
        *,
        synchronize: bool = True,
    ):
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if isinstance(config, ChunkConfig):
            result = self.ops.fused_matmul_reduce_scatter_norm_all_gather(
                a,
                b,
                weight,
                None,
                self.group_name,
                eps=eps,
                norm_type="rms_norm",
                residual=residual,
                microchunk_tokens=config.microchunk_tokens,
            )
        else:
            result = self.ops._C.fused_matmul_reduce_scatter_norm_all_gather_profiled(
                a, b, weight, residual, self.group_name, eps=eps, config=config
            )
        if synchronize:
            self.device_synchronize()
        return result

    def device_synchronize(self) -> None:
        getattr(torch, self.device.type).synchronize(self.device)

    def close(self) -> None:
        if not self._closed:
            close = getattr(self.ops, "close_tpsp", None)
            if close is not None:
                close(self.group_name)
            else:
                _LOG.warning(
                    "TPSP backend has no close_tpsp API; native pools may remain "
                    "allocated until process exit"
                )
            self._closed = True


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
    candidates: tuple[tuple[int, float], ...] = ()
    input_width: int | None = None
    norm_eps: float = _EPS
    finalists: tuple[tuple[int, float], ...] = ()
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


class TPSPProfileSession:
    """Profile named projection shapes without depending on a model class."""

    def __init__(
        self, shapes: dict[str, TPSPShape], tp_size: int, group_name: str
    ) -> None:
        self.shapes = shapes
        self.tp_size = tp_size
        self.group_name = group_name
        self.profiles: dict[str, SPProfile] | None = None
        self.backend: TPSPBackend | None = None

    def profile(self, max_batched_tokens: int, parameter: torch.nn.Parameter) -> None:
        if self.profiles is not None:
            return
        if not self.shapes:
            raise ValueError("TPSP requires at least one projection shape")
        backend = TPSPBackend.open(
            dtype=parameter.dtype,
            tp_size=self.tp_size,
            hidden_size=next(iter(self.shapes.values())).hidden_size,
            group_name=self.group_name,
            device=parameter.device,
        )
        self.backend = backend
        if backend is None:
            self.profiles = {
                name: SPProfile(
                    self.tp_size,
                    shape.hidden_size,
                    max_batched_tokens,
                    "unsupported",
                    "no fused backend on this device",
                    input_width=shape.input_width,
                    norm_eps=shape.norm_eps,
                    gather_sharded_residual=shape.sharded_residual,
                )
                for name, shape in self.shapes.items()
            }
            return
        try:
            self.profiles = {
                name: backend.profile(
                    tp_size=self.tp_size,
                    hidden_size=shape.hidden_size,
                    input_width=shape.input_width,
                    max_batched_tokens=max_batched_tokens,
                    norm_eps=shape.norm_eps,
                    sharded_residual=shape.sharded_residual,
                    time_budget_s=240.0,
                )
                for name, shape in self.shapes.items()
            }
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None
        if self.profiles is not None:
            self.profiles = {
                name: SPProfile(
                    profile.tp_size,
                    profile.hidden_size,
                    profile.max_batched_tokens,
                    "disabled",
                    "TPSP closed",
                    input_width=profile.input_width,
                    norm_eps=profile.norm_eps,
                    gather_sharded_residual=profile.gather_sharded_residual,
                )
                for name, profile in self.profiles.items()
            }


def profile_registered_tpsp(model: torch.nn.Module, max_batched_tokens: int) -> None:
    """Profile projection shapes registered by an opt-in model adapter."""
    session = getattr(model, "tpsp_profile", None)
    if session is not None:
        if not isinstance(session, TPSPProfileSession):
            raise TypeError("tpsp_profile must be a TPSPProfileSession")
        session.profile(max_batched_tokens, next(model.parameters()))


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


def _top_chunks(chunks: list[int], scores: dict[int, float]) -> list[int]:
    return sorted(chunks, key=scores.__getitem__)[:2]


def profile_sp_config(
    backend: TPSPBackend,
    tp_size: int,
    hidden_size: int,
    input_width: int,
    max_batched_tokens: int,
    group_name: str,
    time_budget_s: float,
    norm_eps: float = _EPS,
    sharded_residual: bool = False,
) -> SPProfile:
    """Scan chunks with the fused op's default communication configuration."""
    if time_budget_s <= 0:
        raise ValueError("time_budget_s must be positive")

    def unsupported(reason: str) -> SPProfile:
        return SPProfile(
            tp_size, hidden_size, max_batched_tokens, "unsupported", reason
        )

    if (
        not 2 <= tp_size <= 8
        or hidden_size <= 0
        or hidden_size % tp_size
        or max_batched_tokens < 1
    ):
        return unsupported(
            "requires TP in [2, 8], divisible hidden size and positive "
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
        b = (
            torch.randn(
                input_width,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            / 8
        )
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
        )

    def run(data, candidate):
        (a, b, linear_weight), weight, residual, local_residual = data
        if candidate is not None:
            return backend.fused(
                a, b, weight, local_residual, norm_eps, candidate, synchronize=False
            )[2]
        partial = F.linear(a, linear_weight)
        full = partial.clone()
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=group)
        if sharded_residual:
            rows = local_residual.size(0)
            gathered_residual = torch.empty(
                (tp_size * rows, hidden_size), device=device, dtype=a.dtype
            )
            dist.all_gather_into_tensor(gathered_residual, local_residual, group=group)
            full_residual = gathered_residual[: a.size(0)].contiguous()
        else:
            full_residual = residual.clone()
        torch.ops._C.fused_add_rms_norm(full, full_residual, weight, norm_eps)
        return full

    def measure(data, candidate) -> float:
        dist.barrier(group=group)
        backend.device_synchronize()
        start = time.perf_counter()
        result = run(data, candidate)
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
    samples: dict[int, list[float]] = {}

    def screen_chunk(chunk: int, results: dict[int, list[float]]) -> float | None:
        candidate = ChunkConfig(chunk)
        results[chunk] = []
        measure(data, candidate)
        if expired():
            return None
        for _ in range(_SCREEN_TRIALS):
            results[chunk].append(measure(data, candidate))
            if expired():
                return None
        score = torch.tensor(
            [_screen_score(results[chunk])], dtype=torch.float64, device=device
        )
        dist.all_reduce(score, op=dist.ReduceOp.MAX, group=group)
        return float(score.item())

    chunks = _search_chunks(
        shard_rows,
        lambda chunk: screen_chunk(chunk, samples),
        backend.chunk_granularity,
    )
    if chunks is None:
        return inconclusive("screening time budget exceeded")
    scores = torch.tensor(
        [_screen_score(samples[chunk]) for chunk in chunks],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(scores, op=dist.ReduceOp.MAX, group=group)
    candidate_results = tuple(
        (chunk, float(score)) for chunk, score in zip(chunks, scores.tolist())
    )
    screened_scores = dict(candidate_results)
    finalists = _top_chunks(chunks, screened_scores)
    retested: dict[int, list[float]] = {}
    for chunk in finalists:
        if screen_chunk(chunk, retested) is None:
            return inconclusive(
                "finalist time budget exceeded", candidate_results=candidate_results
            )
    final_scores = torch.tensor(
        [_screen_score(retested[chunk]) for chunk in finalists],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(final_scores, op=dist.ReduceOp.MAX, group=group)
    finalist_results = tuple(
        (chunk, float(score)) for chunk, score in zip(finalists, final_scores.tolist())
    )
    candidate = ChunkConfig(min(finalist_results, key=lambda item: item[1])[0])
    data = None

    check_data = inputs(min(max_batched_tokens, tp_size * 5 + 1))
    conventional = run(check_data, None)
    native = run(check_data, candidate)
    backend.device_synchronize()
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
    status = "enabled" if threshold is not None else "disabled"
    reason = (
        "" if threshold is not None else "no sustained benefit with 128-token steps"
    )
    profile = SPProfile(
        tp_size,
        hidden_size,
        max_batched_tokens,
        status,
        reason,
        threshold,
        candidate if status == "enabled" else None,
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
            candidate.microchunk_tokens,
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
