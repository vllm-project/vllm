# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling for BF16 TP projection / residual RMSNorm chains."""

from __future__ import annotations

import importlib
import logging
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import distributed_c10d as c10d

if TYPE_CHECKING:
    from vllm.v1.worker.tpsp_cuda import CudaTPSPContext

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


class TPSPBackend:
    """Profile a fused projection with platform-owned resources.

    A native ``profile_tpsp_config`` returns an object with
    ``threshold_tokens`` and opaque ``config`` attributes. It requires a
    matching native ``fused_matmul_reduce_scatter_norm_all_gather_profiled``
    entry point. When the native profiler is absent, vLLM scans chunks
    through the standard fused op, checking both NCCL and P2P when available.
    Set ``VLLM_TPSP_COMM_MODE`` to ``nccl`` or ``p2p`` to restrict the sweep.
    """

    requires_projection_context = False
    profiles_transport_modes = False
    synchronize_after_fused = True
    supports_projection_bias = False

    @classmethod
    def create(cls, group_name: str, device: torch.device) -> TPSPBackend | None:
        return cls(None, group_name, device)

    def __init__(
        self,
        ops: Any,
        group_name: str,
        device: torch.device,
    ):
        self.ops = ops
        self.group_name = group_name
        self.device = device
        self.chunk_granularity = getattr(ops, "tpsp_chunk_granularity", 64)
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
        external = getattr(getattr(self.ops, "_C", None), "profile_tpsp_config", None)
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
                config=(
                    TPSPProjectionContext(context, config)
                    if threshold is not None and context is not None
                    else config
                ),
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
            context=context,
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
        norm_type: str = "rms_norm",
        projection_bias: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        context: Any | None = None,
    ):
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if isinstance(config, TPSPProjectionContext):
            if context is not None:
                raise ValueError("TPSP context was supplied twice")
            context, config = config.transport, config.config
        context = self._profile_context(context)
        if isinstance(config, ChunkConfig):
            result = self.run(
                a,
                b,
                weight,
                residual,
                eps,
                config,
                norm_type=norm_type,
                projection_bias=projection_bias,
                norm_bias=norm_bias,
                context=context,
            )
        else:
            if (
                norm_type != "rms_norm"
                or norm_bias is not None
                or projection_bias is not None
            ):
                raise ValueError(
                    "TPSP profiled configuration only supports RMSNorm without bias"
                )
            result = self.ops._C.fused_matmul_reduce_scatter_norm_all_gather_profiled(
                a, b, weight, residual, self.group_name, eps=eps, config=config
            )
        if synchronize and self.synchronize_after_fused:
            self.device_synchronize()
        return result

    def run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float,
        config: ChunkConfig,
        *,
        norm_type: str = "rms_norm",
        projection_bias: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        context: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def device_synchronize(self) -> None:
        getattr(torch, self.device.type).synchronize(self.device)

    def close(self, context: Any | None = None) -> None:
        if not self._closed:
            self._closed = True


def get_tpsp_backend(group_name: str, device: torch.device) -> TPSPBackend | None:
    from vllm.platforms import current_platform

    backend_path = current_platform.get_tpsp_backend_cls()
    if not backend_path:
        return None
    module, _, name = backend_path.rpartition(".")
    backend_cls = getattr(importlib.import_module(module), name)
    return backend_cls.create(group_name, device)


class XPUTPSPBackend(TPSPBackend):
    requires_projection_context = True

    @classmethod
    def create(
        cls,
        group_name: str,
        device: torch.device,
    ) -> XPUTPSPBackend | None:
        if device.type != "xpu":
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
        import vllm_xpu_kernels._C  # noqa: F401

        return cls(ops, group_name, device)

    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> str | None:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if device != self.device or group_name != self.group_name or tp_size > 8:
            return None
        if not self._valid_open(
            dtype=dtype,
            tp_size=tp_size,
            hidden_size=hidden_size,
            max_batched_tokens=max_batched_tokens,
            group_name=group_name,
            device=device,
        ):
            return None
        return group_name

    def _profile_context(self, context: Any | None) -> str:
        if context != self.group_name:
            raise ValueError("TPSP context belongs to another backend")
        return context

    def run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float,
        config: ChunkConfig,
        *,
        norm_type: str = "rms_norm",
        projection_bias: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        context: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context != self.group_name:
            raise ValueError("TPSP context belongs to another backend")
        if (
            norm_type != "rms_norm"
            or projection_bias is not None
            or norm_bias is not None
        ):
            raise ValueError("This TPSP backend does not support bias or LayerNorm")
        return self.ops.fused_matmul_reduce_scatter_norm_all_gather(
            a,
            b,
            weight,
            None,
            self.group_name,
            eps=eps,
            norm_type=norm_type,
            residual=residual,
            microchunk_tokens=config.microchunk_tokens,
        )

    def close(self, context: Any | None = None) -> None:
        if context is not None:
            self._profile_context(context)
        if not self._closed:
            close = getattr(self.ops, "close_tpsp", None)
            if close is not None:
                close(self.group_name)
            else:
                _LOG.warning(
                    "TPSP backend has no close_tpsp API; native pools may remain "
                    "allocated until process exit"
                )
            super().close(context)


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
        self.contexts: dict[str, Any] = {}

    def profile(self, max_batched_tokens: int, parameter: torch.nn.Parameter) -> None:
        if self.profiles is not None:
            return
        if not self.shapes:
            raise ValueError("TPSP requires at least one projection shape")
        backend = get_tpsp_backend(self.group_name, parameter.device)
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
            self._log_fallback()
            return
        try:
            profiles = {}
            for name, shape in self.shapes.items():
                context = backend.open(
                    dtype=parameter.dtype,
                    tp_size=self.tp_size,
                    hidden_size=shape.hidden_size,
                    max_batched_tokens=max_batched_tokens,
                    group_name=self.group_name,
                    device=parameter.device,
                )
                if context is None:
                    profiles[name] = SPProfile(
                        self.tp_size,
                        shape.hidden_size,
                        max_batched_tokens,
                        "unsupported",
                        "neither NCCL nor P2P is usable",
                        input_width=shape.input_width,
                        norm_eps=shape.norm_eps,
                        gather_sharded_residual=shape.sharded_residual,
                    )
                    continue
                self.contexts[name] = context
                profiles[name] = backend.profile(
                    tp_size=self.tp_size,
                    hidden_size=shape.hidden_size,
                    input_width=shape.input_width,
                    max_batched_tokens=max_batched_tokens,
                    norm_eps=shape.norm_eps,
                    sharded_residual=shape.sharded_residual,
                    time_budget_s=240.0,
                    context=context,
                )
            self.profiles = profiles
            if not all(profile.enabled for profile in profiles.values()):
                self._close_backend()
                self._log_fallback()
        except Exception:
            self.close()
            raise

    def _log_fallback(self) -> None:
        assert self.profiles is not None
        details = ", ".join(
            f"{name}={profile.status}"
            + (f" ({profile.reason})" if profile.reason else "")
            for name, profile in self.profiles.items()
        )
        _LOG.warning("TPSP using standard forward; projection plans: %s", details)

    def _close_backend(self) -> None:
        if self.backend is not None:
            for context in self.contexts.values():
                self.backend.close(context)
            self.backend.close()
            self.backend = None
            self.contexts.clear()

    def close(self) -> None:
        self._close_backend()
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
    context: CudaTPSPContext | None = None,
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

    def run(data, candidate):
        (a, b, linear_weight), weight, residual, local_residual, _ = data
        if candidate is not None:
            return backend.fused(
                a,
                b,
                weight,
                local_residual,
                norm_eps,
                candidate,
                synchronize=False,
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
    conventional = run(check_data, None)
    native = run(check_data, candidate)
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
