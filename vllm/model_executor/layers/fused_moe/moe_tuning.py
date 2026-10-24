# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable warmup autotuning helpers for the fused MoE Triton kernel."""

from __future__ import annotations

import functools
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    get_default_config,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.platforms import current_platform
from vllm.triton_utils import triton

logger = init_logger(__name__)

Config = dict[str, Any]
ConfigKey = tuple[tuple[str, int], ...]

# Same M grid as benchmarks/kernels/benchmark_moe.py, which produced the bundled
# config tables.
DEFAULT_M_BUCKETS = (
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096,
)  # fmt: skip

# Configs timed per M bucket, starting points included. Every new config costs
# a Triton compile, which dominates the tuning time.
MAX_CONFIGS_PER_BUCKET = 48

# Random configs timed with the starting points so the local search does not
# only explore around the default.
_RANDOM_STARTS = 8

# Routing draws the timed calls cycle through. At small M the number of active
# experts varies between draws, and one more wave of thread blocks can make a
# config 30% slower, so a single draw can crown a config that is usually slow.
_ROUTING_DRAWS = 10

# A candidate whose first timed replay is this much slower than the best so
# far cannot win, so its full timing is skipped.
_EARLY_EXIT_RATIO = 1.5

# Alternating timings of the sweep winner and the default before keeping the
# winner.
_HEAD_TO_HEAD_ROUNDS = 2

# Speedup over the default a winner needs in the side-by-side timing. Random
# routing activates more experts than real models do at small M, which can
# reverse rankings that differ by a few percent.
_MIN_SPEEDUP = 1.05

_TILE_AXES: dict[str, tuple[int, ...]] = {
    "BLOCK_SIZE_M": (16, 32, 64, 128, 256),
    "BLOCK_SIZE_N": (16, 32, 64, 128, 256),
    "BLOCK_SIZE_K": (32, 64, 128, 256),
    "GROUP_SIZE_M": (1, 4, 8, 16, 32),
    "num_warps": (1, 2, 4, 8),
    "num_stages": (1, 2, 3, 4, 5),
}

# Backend launch options searched when the installed Triton keeps every value
# on this GPU.
_BACKEND_OPTION_VALUES: dict[str, tuple[int, ...]] = {
    "waves_per_eu": (0, 1, 2, 4),
    "matrix_instr_nonkdim": (16, 32),
    "kpack": (1, 2),
}


@dataclass(frozen=True)
class FusedMoETuningCase:
    M: int
    w1_shape: tuple[int, int, int]
    w2_shape: tuple[int, int, int]
    w1_stride: tuple[int, int, int]
    w2_stride: tuple[int, int, int]
    top_k: int
    dtype: torch.dtype
    use_fp8_w8a8: bool
    # The kernel pairs per-token activation scales with per-channel weight
    # scales, so one flag selects both.
    per_act_token_quant: bool
    block_shape: tuple[int, int] | None
    activation: MoEActivation
    device: torch.device

    @property
    def E(self) -> int:
        return self.w2_shape[0]

    @property
    def N(self) -> int:
        return self.w2_shape[2]

    @property
    def K(self) -> int:
        return self.w1_shape[2]

    @property
    def config_dtype(self) -> str | None:
        return "fp8_w8a8" if self.use_fp8_w8a8 else None

    @property
    def block_shape_list(self) -> list[int] | None:
        return list(self.block_shape) if self.block_shape is not None else None


@dataclass
class FusedMoEInputs:
    hidden_states: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    # (topk_weights, topk_ids) per routing draw.
    routings: list[tuple[torch.Tensor, torch.Tensor]]
    quant_config: FusedMoEQuantConfig | None


@dataclass(frozen=True)
class TritonBackend:
    """What the installed Triton reports about the current GPU."""

    warp_size: int
    max_shared_mem: int
    # Searched backend option values, and the value Triton uses when unset.
    options: dict[str, tuple[int, ...]]
    option_defaults: dict[str, int]


@functools.cache
def _triton_backend(device_index: int) -> TritonBackend:
    driver = triton.runtime.driver.active
    target = driver.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    defaults = backend.parse_options({})

    def keeps(name: str, value: int) -> bool:
        try:
            return getattr(backend.parse_options({name: value}), name) == value
        except Exception:
            return False

    options = {
        name: tuple(sorted({getattr(defaults, name), *values}))
        for name, values in _BACKEND_OPTION_VALUES.items()
        if all(keeps(name, v) for v in values)
    }
    properties = driver.utils.get_device_properties(device_index)
    return TritonBackend(
        warp_size=target.warp_size,
        max_shared_mem=properties["max_shared_mem"],
        options=options,
        option_defaults={name: getattr(defaults, name) for name in options},
    )


def triton_backend() -> TritonBackend:
    return _triton_backend(torch.accelerator.current_device_index())


def valid_m_buckets(max_num_batched_tokens: int) -> list[int]:
    """Offline M grid clipped to the server's token limit.

    The limit itself is added when it falls inside the grid. Larger limits use
    the grid's largest entry through the runtime's nearest-M lookup.
    """
    if max_num_batched_tokens <= 0:
        return []
    buckets = [m for m in DEFAULT_M_BUCKETS if m <= max_num_batched_tokens]
    if max_num_batched_tokens < DEFAULT_M_BUCKETS[-1]:
        buckets.append(max_num_batched_tokens)
    return sorted(set(buckets))


def _key(config: Config) -> ConfigKey:
    return tuple(sorted(config.items()))


def _divides(a: int, b: int) -> bool:
    return a % b == 0 or b % a == 0


def normalize_config(
    config: Config, block_shape: tuple[int, int] | None
) -> Config | None:
    """The config as the kernel launches it, or None if the kernel cannot run it.

    SPLIT_K is always forced to 1. Block quantization clamps BLOCK_SIZE_K to the
    quant block and needs tiles that divide or are divided by the quant block.
    """
    config = {k: v for k, v in config.items() if k != "SPLIT_K"}
    if block_shape is not None:
        block_n, block_k = block_shape
        config["BLOCK_SIZE_K"] = min(config["BLOCK_SIZE_K"], block_n, block_k)
        if not (
            _divides(config["BLOCK_SIZE_N"], block_n)
            and _divides(config["BLOCK_SIZE_K"], block_k)
        ):
            return None
    return config


def default_config(case: FusedMoETuningCase) -> Config:
    config = get_default_config(
        case.M,
        case.E,
        case.N,
        case.K,
        case.top_k,
        case.config_dtype,
        case.block_shape_list,
    )
    normalized = normalize_config(config, case.block_shape)
    assert normalized is not None
    return normalized


def search_axes(case: FusedMoETuningCase) -> dict[str, tuple[int, ...]]:
    backend = triton_backend()
    # An M tile much larger than an expert's share of the rows is padding.
    rows_per_expert = triton.cdiv(case.M * case.top_k, case.E)
    max_block_m = max(16, 2 * triton.next_power_of_2(rows_per_expert))
    return {
        **_TILE_AXES,
        "BLOCK_SIZE_M": tuple(
            m for m in _TILE_AXES["BLOCK_SIZE_M"] if m <= max_block_m
        ),
        **backend.options,
    }


def fits_hardware(config: Config, case: FusedMoETuningCase) -> bool:
    """Whether the config's tiles fit the GPU's registers and shared memory."""
    backend = triton_backend()
    block_m, block_n = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]
    threads = config["num_warps"] * backend.warp_size
    # Each thread holds block_m * block_n / threads fp32 accumulators.
    if not threads <= block_m * block_n <= 256 * threads:
        return False
    if config.get("matrix_instr_nonkdim") == 32 and min(block_m, block_n) < 32:
        return False
    element_size = 1 if case.use_fp8_w8a8 else case.dtype.itemsize
    tile_bytes = (block_m + block_n) * config["BLOCK_SIZE_K"] * element_size
    # Pipelining keeps num_stages - 1 copies of the input tiles in shared
    # memory. A config that still runs out fails to compile and is skipped.
    return tile_bytes * max(1, config["num_stages"] - 1) <= backend.max_shared_mem


def _complete(
    config: Config, axes: dict[str, tuple[int, ...]], case: FusedMoETuningCase
) -> Config | None:
    """The config restricted to the searched keys, with backend options it lacks
    set to Triton's defaults."""
    defaults = triton_backend().option_defaults
    out = {k: v for k, v in config.items() if k in axes}
    for name in axes:
        if name not in out and name in defaults:
            out[name] = defaults[name]
    if any(name not in out for name in axes):
        return None
    return normalize_config(out, case.block_shape)


def _canonical(config: Config) -> Config:
    """The config without backend options left at Triton's defaults."""
    defaults = triton_backend().option_defaults
    return {k: v for k, v in config.items() if k not in defaults or v != defaults[k]}


def _random_configs(
    axes: dict[str, tuple[int, ...]],
    case: FusedMoETuningCase,
    count: int,
    rng: random.Random,
) -> list[Config]:
    configs: list[Config] = []
    for _ in range(50 * count):
        if len(configs) == count:
            break
        config = normalize_config(
            {name: rng.choice(values) for name, values in axes.items()},
            case.block_shape,
        )
        if config is not None and fits_hardware(config, case):
            configs.append(config)
    return configs


def _neighbors(
    config: Config, axes: dict[str, tuple[int, ...]], case: FusedMoETuningCase
) -> Iterator[Config]:
    """Configs one step away along a single axis."""
    for name, values in axes.items():
        if config.get(name) not in values:
            continue
        i = values.index(config[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                neighbor = normalize_config(
                    {**config, name: values[j]}, case.block_shape
                )
                if neighbor is not None and fits_hardware(neighbor, case):
                    yield neighbor


def starting_configs(
    case: FusedMoETuningCase, seeds: Sequence[Config] = ()
) -> list[Config]:
    """The default config, then the seeds the kernel can run, without repeats."""
    starts: dict[ConfigKey, Config] = {}
    for config in [
        default_config(case),
        *(normalize_config(seed, case.block_shape) for seed in seeds),
    ]:
        if config is not None:
            starts.setdefault(_key(config), config)
    return list(starts.values())


def _storage_numel(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    return 1 + sum((size - 1) * step for size, step in zip(shape, stride))


def moe_scratch_bytes(case: FusedMoETuningCase) -> int:
    """Peak scratch memory of one call on weights in the live layout, with
    headroom for the graph pool."""
    weight_bytes = 1 if case.use_fp8_w8a8 else case.dtype.itemsize
    weights = weight_bytes * (
        _storage_numel(case.w1_shape, case.w1_stride)
        + _storage_numel(case.w2_shape, case.w2_stride)
    )
    rows = case.M * case.top_k
    caches = rows * (max(case.w1_shape[1], case.K) + case.N)
    outputs = 4 * case.M * case.K
    return weights + 2 * case.dtype.itemsize * (caches + outputs)


def _rand_weight(
    case: FusedMoETuningCase,
    shape: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> torch.Tensor:
    dtype = current_platform.fp8_dtype() if case.use_fp8_w8a8 else case.dtype
    w = torch.empty_strided(shape, stride, dtype=dtype, device=case.device)
    # One expert at a time keeps the staging buffer small for large models.
    for expert in range(shape[0]):
        w[expert].copy_(
            torch.randn(shape[1:], dtype=case.dtype, device=case.device) / 15
        )
    return w


def _weight_scale(case: FusedMoETuningCase, rows: int, cols: int) -> torch.Tensor:
    if case.block_shape is not None:
        block_n, block_k = case.block_shape
        shape: tuple[int, ...] = (
            case.E,
            triton.cdiv(rows, block_n),
            triton.cdiv(cols, block_k),
        )
    elif case.per_act_token_quant:
        shape = (case.E, rows)
    else:
        shape = (case.E,)
    return torch.ones(shape, dtype=torch.float32, device=case.device)


def make_moe_inputs(case: FusedMoETuningCase) -> FusedMoEInputs:
    w1 = _rand_weight(case, case.w1_shape, case.w1_stride)
    w2 = _rand_weight(case, case.w2_shape, case.w2_stride)
    quant_config = None
    if case.use_fp8_w8a8:
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=_weight_scale(case, case.w1_shape[1], case.K),
            w2_scale=_weight_scale(case, case.K, case.N),
            per_act_token_quant=case.per_act_token_quant,
            per_out_ch_quant=case.per_act_token_quant,
            block_shape=case.block_shape_list,
        )
    hidden_states = (
        torch.randn(case.M, case.K, dtype=case.dtype, device=case.device) / 10
    )
    routings = []
    for _ in range(_ROUTING_DRAWS):
        gating = torch.randn(case.M, case.E, dtype=torch.float32, device=case.device)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, gating, case.top_k, renormalize=True
        )
        routings.append((topk_weights, topk_ids))
    return FusedMoEInputs(hidden_states, w1, w2, routings, quant_config)


def run_moe(
    case: FusedMoETuningCase,
    inputs: FusedMoEInputs,
    config: Config,
    routing: int = 0,
) -> torch.Tensor:
    topk_weights, topk_ids = inputs.routings[routing]
    # override_config only restores the previous config when its body exits
    # normally, so a failing candidate is re-raised after the block.
    with override_config(config):
        try:
            return fused_experts(
                inputs.hidden_states,
                inputs.w1,
                inputs.w2,
                topk_weights,
                topk_ids,
                activation=case.activation,
                quant_config=inputs.quant_config,
            )
        except Exception as e:
            error = e
    raise error


def benchmark_moe_config(
    case: FusedMoETuningCase,
    config: Config,
    *,
    inputs: FusedMoEInputs | None = None,
    num_iters: int = 10,
    num_warmup: int = 5,
    graph_batch_size: int = 10,
    cutoff_us: float | None = None,
) -> float | None:
    """Benchmark one config on disposable buffers and return microseconds.

    If the first timed replay is slower than cutoff_us per call, that rough
    time is returned without the remaining replays.
    """
    try:
        if inputs is None:
            inputs = make_moe_inputs(case)

        for i in range(num_warmup):
            run_moe(case, inputs, config, i % len(inputs.routings))
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(graph_batch_size):
                run_moe(case, inputs, config, i % len(inputs.routings))
        torch.accelerator.synchronize()

        graph.replay()
        torch.accelerator.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latencies: list[float] = []
        for i in range(num_iters):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end))
            if i == 0 and cutoff_us is not None:
                first_us = latencies[0] / graph_batch_size * 1000
                if first_us > cutoff_us:
                    graph.reset()
                    return first_us
        graph.reset()
        return sum(latencies) / (num_iters * graph_batch_size) * 1000
    except Exception as e:
        if "OutOfResources" not in str(e):
            logger.debug("Fused MoE config %s failed for %s: %s", config, case, e)
        return None


def validate_moe_config(
    case: FusedMoETuningCase,
    config: Config,
    *,
    inputs: FusedMoEInputs | None = None,
    reference: torch.Tensor | None = None,
    rtol: float = 1e-2,
) -> bool:
    """Check a candidate against the default launch config on identical inputs."""
    if inputs is None:
        inputs = make_moe_inputs(case)
    if reference is None:
        reference = run_moe(case, inputs, default_config(case))
    out = run_moe(case, inputs, config)
    ref = reference.float()
    atol = rtol * ref.abs().max().item()
    return torch.allclose(out.float(), ref, rtol=rtol, atol=atol)


def tune_fused_moe_case(
    case: FusedMoETuningCase,
    *,
    seeds: Sequence[Config] = (),
    max_configs: int = MAX_CONFIGS_PER_BUCKET,
    num_iters: int = 10,
    num_warmup: int = 5,
) -> Config | None:
    """Return the fastest config found that reproduces the default config's output.

    The search times the starting points and a few random configs, then
    repeatedly times every unvisited neighbor of the fastest config not yet
    expanded, until max_configs configs have been timed.
    """
    rng = random.Random(case.M)
    inputs = make_moe_inputs(case)
    default = default_config(case)
    reference = run_moe(case, inputs, default)
    axes = search_axes(case)

    def time_config(config: Config, cutoff_us: float | None = None) -> float | None:
        return benchmark_moe_config(
            case,
            config,
            inputs=inputs,
            num_iters=num_iters,
            num_warmup=num_warmup,
            cutoff_us=cutoff_us,
        )

    timings: dict[ConfigKey, float | None] = {}
    best_us = float("inf")

    def time_new(configs: list[Config]) -> None:
        nonlocal best_us
        new = {_key(c): c for c in configs if _key(c) not in timings}
        batch = list(new.values())[: max_configs - len(timings)]
        for config in batch:
            elapsed_us = time_config(config, cutoff_us=best_us * _EARLY_EXIT_RATIO)
            timings[_key(config)] = elapsed_us
            if elapsed_us is not None:
                best_us = min(best_us, elapsed_us)

    def ranked() -> list[ConfigKey]:
        timed = ((us, k) for k, us in timings.items() if us is not None)
        return [k for _, k in sorted(timed)]

    starts = [_complete(c, axes, case) for c in starting_configs(case, seeds)]
    time_new(
        [c for c in starts if c is not None]
        + _random_configs(axes, case, _RANDOM_STARTS, rng)
    )
    expanded: set[ConfigKey] = set()
    while len(timings) < max_configs:
        center = next((k for k in ranked() if k not in expanded), None)
        if center is None:
            break
        expanded.add(center)
        neighbors = list(_neighbors(dict(center), axes, case))
        rng.shuffle(neighbors)
        time_new(neighbors)

    winner = next(
        (
            _canonical(dict(key))
            for key in ranked()
            if validate_moe_config(case, dict(key), inputs=inputs, reference=reference)
        ),
        None,
    )
    if winner is None:
        return None

    # Timings taken during the search drift by more than the gap between close
    # candidates, so the winner has to beat the default again side by side.
    winner_us = default_us = float("inf")
    for _ in range(_HEAD_TO_HEAD_ROUNDS):
        default_us = min(default_us, time_config(default) or float("inf"))
        if winner != default:
            winner_us = min(winner_us, time_config(winner) or float("inf"))
    if default_us < winner_us * _MIN_SPEEDUP:
        winner, winner_us = default, default_us

    logger.info(
        "Fused MoE tuned E=%d,N=%d,K=%d,M=%d to %s (%.2f us, default %.2f us, "
        "%d configs timed).",
        case.E,
        case.N,
        case.K,
        case.M,
        winner,
        winner_us,
        default_us,
        len(timings),
    )
    return winner
