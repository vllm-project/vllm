# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 TP projection plans and startup profiling.

Stage 1 screens chunk sizes for each projection; stage 2 retests finalists.
Stage 3 checks each chunk's numerical result. Stage 4 checks the fused pair's
result, and stage 5 measures the pair to choose one token threshold.
"""

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
    """A projection's chunk configuration and rank-local backend context."""

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
        self.config: object = None
        self.backend: TPSPBackend | None = None
        self.context: Any | None = None

    def fused_gemm_norm(
        self,
        x: torch.Tensor,
        projection: nn.Module,
        residual: torch.Tensor,
        norm: nn.Module,
        residual_is_sharded: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backend = self.backend
        if backend is None:
            raise RuntimeError("TP/SP backend is unavailable")
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        rows = math.ceil(x.size(0) / group.world_size)
        if residual_is_sharded:
            if residual.shape != (rows, self.hidden_size):
                raise RuntimeError("TP/SP residual shard has an unexpected shape")
            local_residual = residual
        else:
            if residual.shape != (x.size(0), self.hidden_size):
                raise RuntimeError("TP/SP full residual has an unexpected shape")
            start = group.rank_in_group * rows
            local_residual = residual.new_zeros((rows, self.hidden_size))
            count = min(rows, max(0, x.size(0) - start))
            if count:
                local_residual[:count] = residual[start : start + count]

        weight = projection.weight
        key = (weight.data_ptr(), weight._version)
        cached = getattr(projection, "_tpsp_transposed_weight", None)
        if cached is None or cached[0] != key:
            cached = (key, weight.T.contiguous())
            projection._tpsp_transposed_weight = cached
        reduced, _, gathered = backend.fused_gemm_rs_norm_ag(
            x.contiguous(),
            cached[1],
            norm.weight,
            local_residual,
            norm.variance_epsilon,
            self.config,
            projection_bias=projection.bias,
            context=self.context,
        )
        return gathered, reduced


@dataclass(frozen=True)
class TPSPProfile:
    enabled: bool
    threshold_tokens: int | None
    max_batched_tokens: int
    o_proj: TPSPProjection
    down_proj: TPSPProjection
    reason: str = ""

    def is_active(self, num_tokens: int) -> bool:
        if not self.enabled:
            return False
        if self.threshold_tokens is None:
            raise RuntimeError("TP/SP has an invalid enabled profile")
        if not 1 <= num_tokens <= self.max_batched_tokens:
            raise ValueError("current_batched_tokens must be within the profiled range")
        return num_tokens >= self.threshold_tokens


def _clear_tpsp_weight_caches(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "_tpsp_transposed_weight"):
            del module._tpsp_transposed_weight


def profile_tpsp(model: nn.Module, max_batched_tokens: int) -> TPSPProfile:
    """Open rank-local contexts and profile both projections as one unit."""
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    first_layer = model.layers[0]
    hidden_size = model.config.hidden_size
    group_name = group.device_group.group_name
    o_plan = TPSPProjection(
        first_layer.self_attn.o_proj.input_size_per_partition,
        hidden_size,
        first_layer.post_attention_layernorm.variance_epsilon,
        group.world_size,
        group_name,
    )
    down_plan = TPSPProjection(
        first_layer.mlp.down_proj.input_size_per_partition,
        hidden_size,
        model.norm.variance_epsilon,
        group.world_size,
        group_name,
    )
    plans = (o_plan, down_plan)

    def disabled(reason: str) -> TPSPProfile:
        _LOG.warning("TPSP using regular forward: %s", reason)
        return TPSPProfile(False, None, max_batched_tokens, o_plan, down_plan, reason)

    for layer in model.layers:
        if (
            layer.self_attn.o_proj.input_size_per_partition != o_plan.input_width
            or layer.mlp.down_proj.input_size_per_partition != down_plan.input_width
        ):
            raise RuntimeError("TP/SP profile does not match the Llama projection")

    parameter = first_layer.self_attn.o_proj.weight
    backend = get_tpsp_backend(group_name, parameter.device)
    if backend is None:
        return disabled("no fused backend on this device")
    enabled = False
    try:
        for plan in plans:
            plan.backend = backend
            plan.context = backend.open(
                dtype=parameter.dtype,
                tp_size=group.world_size,
                hidden_size=hidden_size,
                max_batched_tokens=max_batched_tokens,
                group_name=group_name,
                device=parameter.device,
            )
            if plan.context is None:
                return disabled("fused backend unavailable for this projection")
            candidate = backend.profile(
                tp_size=group.world_size,
                hidden_size=hidden_size,
                input_width=plan.input_width,
                max_batched_tokens=max_batched_tokens,
                norm_eps=plan.norm_eps,
                time_budget_s=240.0,
                context=plan.context,
                config_only=True,
            )
            if candidate.status != "candidate" or candidate.config is None:
                return disabled(
                    f"chunk selection {candidate.status}: {candidate.reason}"
                )
            plan.config = candidate.config

        measurement = profile_tpsp_pair(model, o_plan, down_plan, max_batched_tokens)
        if not measurement.enabled:
            return disabled(
                f"paired profile {measurement.status}: {measurement.reason}"
            )
        threshold = measurement.threshold_tokens
        if threshold is None or not 1 <= threshold <= max_batched_tokens:
            raise RuntimeError("TP/SP Llama has an invalid enabled profile")
        enabled = True
        return TPSPProfile(True, threshold, max_batched_tokens, o_plan, down_plan)
    finally:
        if not enabled:
            for plan in plans:
                if plan.context is not None:
                    backend.close(plan.context)
                plan.context = None
                plan.backend = None
                plan.config = None
            backend.close()
            _clear_tpsp_weight_caches(model)


def _beneficial(measurement: SPMeasurement) -> bool:
    return measurement.lower_benefit_ms > max(0.05, 0.02 * measurement.conventional_ms)


@torch.inference_mode()
def profile_tpsp_pair(
    model: nn.Module,
    o_plan: TPSPProjection,
    down_plan: TPSPProjection,
    max_batched_tokens: int,
) -> SPProfile:
    """Measure the attention-output-to-next-norm path with both projections."""
    from vllm.distributed.parallel_state import get_tp_group

    layers = model.layers
    layer_idx = min(1, len(layers) - 1)
    layer = layers[layer_idx]
    next_norm = (
        layers[layer_idx + 1].input_layernorm
        if layer_idx + 1 < len(layers)
        else model.norm
    )
    group = get_tp_group()
    device_group = group.device_group
    rank = group.rank_in_group
    tp_size = group.world_size
    device = layer.self_attn.o_proj.weight.device
    dtype = layer.self_attn.o_proj.weight.dtype
    hidden_size = layer.hidden_size
    width = layer.self_attn.o_proj.input_size_per_partition
    deadline = time.monotonic() + 240.0

    def inputs(
        tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_output = torch.randn(
            (tokens, width),
            dtype=dtype,
            device=device,
            generator=torch.Generator(device=device).manual_seed(1831 + rank),
        )
        full_residual = torch.randn(
            (tokens, hidden_size),
            dtype=dtype,
            device=device,
            generator=torch.Generator(device=device).manual_seed(407),
        )
        rows = math.ceil(tokens / tp_size)
        local_residual = full_residual.new_zeros((rows, hidden_size))
        start = rank * rows
        count = min(rows, max(0, tokens - start))
        if count:
            local_residual[:count] = full_residual[start : start + count]
        return attn_output, full_residual, local_residual, full_residual.clone()

    def project(
        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], fused: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, full_residual, local_residual, baseline_residual = data
        if fused:
            hidden, residual = o_plan.fused_gemm_norm(
                attn_output,
                layer.self_attn.o_proj,
                local_residual if layer_idx else full_residual,
                layer.post_attention_layernorm,
                bool(layer_idx),
            )
            mlp_input = layer.mlp(hidden, tpsp_active=True)
            return down_plan.fused_gemm_norm(
                mlp_input, layer.mlp.down_proj, residual, next_norm, True
            )
        hidden, _ = layer.self_attn.o_proj(attn_output)
        hidden, residual = layer.post_attention_layernorm(hidden, baseline_residual)
        hidden = layer.mlp(hidden)
        return next_norm(hidden, residual)

    def expired() -> bool:
        flag = torch.tensor(
            [time.monotonic() >= deadline], dtype=torch.int32, device=device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=device_group)
        return bool(flag.item())

    def measure(
        data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], fused: bool
    ) -> float:
        if not fused:
            data[3].copy_(data[1])
        dist.barrier(group=device_group)
        torch.accelerator.synchronize(device)
        start = time.perf_counter()
        project(data, fused)
        torch.accelerator.synchronize(device)
        elapsed = torch.tensor(
            [(time.perf_counter() - start) * 1000], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
        return elapsed.item()

    # Stage 4: check the complete o -> MLP -> down chain against the normal path.
    check_data = inputs(min(max_batched_tokens, tp_size * 5 + 1))
    conventional, conventional_residual = project(check_data, False)
    native, native_residual = project(check_data, True)
    padded = native_residual.new_empty((tp_size * native_residual.size(0), hidden_size))
    dist.all_gather_into_tensor(
        padded, native_residual.contiguous(), group=device_group
    )
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(native, conventional, rtol=0.03, atol=0.1)
    torch.testing.assert_close(
        padded[: check_data[0].size(0)], conventional_residual, rtol=0.03, atol=0.1
    )
    del check_data, conventional, conventional_residual, native, native_residual

    # Stage 5: select one token threshold for both projections together.
    measurements: list[SPMeasurement] = []
    sizes = list(range(128, max_batched_tokens + 1, 128))
    if not sizes or sizes[-1] != max_batched_tokens:
        sizes.append(max_batched_tokens)

    def measure_size(tokens: int) -> SPMeasurement | None:
        data = inputs(tokens)
        measure(data, False)
        measure(data, True)
        paired: list[float] = []
        conventional: list[float] = []
        fused: list[float] = []
        for trial in range(_TRIALS):
            order = (False, True) if trial % 2 == 0 else (True, False)
            times = {item: measure(data, item) for item in order}
            conventional.append(times[False])
            fused.append(times[True])
            paired.append(times[False] - times[True])
            if expired():
                return None
        lower = statistics.mean(paired) - 2.776 * statistics.stdev(paired) / math.sqrt(
            _TRIALS
        )
        return SPMeasurement(
            tokens, statistics.median(conventional), statistics.median(fused), lower
        )

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
    reason = "" if threshold is not None else "no sustained paired TPSP benefit"
    if expired():
        status, reason, threshold = (
            "inconclusive",
            "paired profiling time budget exceeded",
            None,
        )
    profile = SPProfile(
        tp_size,
        hidden_size,
        max_batched_tokens,
        status,
        reason,
        threshold_tokens=threshold,
        measurements=tuple(measurements),
    )
    if rank == 0:
        _LOG.warning("TPSP paired startup profile: %s", profile)
    return profile


def profile_sp_config(
    backend: TPSPBackend,
    tp_size: int,
    hidden_size: int,
    input_width: int,
    max_batched_tokens: int,
    time_budget_s: float,
    norm_eps: float = _EPS,
    context: Any | None = None,
    config_only: bool = False,
) -> SPProfile:
    """Time chunks through the backend, check correctness, and find a threshold.

    Screen chunks and retest the two finalists. Unless ``config_only`` is set,
    compare the winner with the conventional path in 128-token steps, retaining
    the existing confidence-bound and maximum-size checks.

    Returns:
        SPProfile with candidate timings and per-size measurements. An enabled
        profile contains the selected chunk in ``config`` and the first
        beneficial size in ``threshold_tokens``. With ``config_only``, return
        the selected chunk without a threshold for the paired profile to use.

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

    # Stage 1: screen micro-chunk sizes for this projection at maximum load.
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
    # Stage 2: retest the two fastest chunks before committing to one.
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

    # Stage 3: verify numerical parity before passing this chunk to the pair.
    check_data = inputs(min(max_batched_tokens, tp_size * 5 + 1))
    conventional = project(check_data, None)
    native = project(check_data, candidate)
    torch.accelerator.synchronize(device)
    torch.testing.assert_close(native, conventional, rtol=0.02, atol=0.05)
    del check_data, conventional, native
    if config_only:
        return SPProfile(
            tp_size,
            hidden_size,
            max_batched_tokens,
            "candidate",
            "",
            config=candidate,
            candidates=tuple(candidate_results),
            input_width=input_width,
            finalists=finalist_results,
        )

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
