# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable warmup autotuning helpers for the W8A8 block FP8 Triton GEMM."""

from __future__ import annotations

import functools
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import triton

from .fp8_utils import _default_w8a8_block_fp8_config, w8a8_triton_block_scaled_mm

logger = init_logger(__name__)

Config = dict[str, int]
ConfigKey = tuple[tuple[str, int], ...]
ConfigBenchmark = Callable[[Config, float | None], float | None]
BlockFp8Inputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# Same M grid as benchmarks/kernels/benchmark_w8a8_block_fp8.py, which produced
# the bundled config tables.
DEFAULT_M_BUCKETS = (
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096,
)  # fmt: skip

# Configs timed per M bucket, starting points included. Every new config costs
# a Triton compile, which dominates the tuning time.
MAX_CONFIGS_PER_BUCKET = 48

# Random configs timed with the starting points so the local search does not
# only explore around the default.
_RANDOM_STARTS = 8

# A candidate whose first timed replay is this much slower than the best so
# far cannot win, so its full timing is skipped.
_EARLY_EXIT_RATIO = 1.5

# Alternating timings of the search winner and the default before keeping the
# winner.
_HEAD_TO_HEAD_ROUNDS = 2

# Speedup over the default a winner needs in the side-by-side timing, so timing
# noise alone does not replace the default.
_MIN_SPEEDUP = 1.02

_TILE_AXES: dict[str, tuple[int, ...]] = {
    "BLOCK_SIZE_M": (16, 32, 64, 128, 256),
    "BLOCK_SIZE_N": (16, 32, 64, 128, 256),
    "BLOCK_SIZE_K": (32, 64, 128, 256),
    "GROUP_SIZE_M": (1, 4, 8, 16, 32, 64),
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
class BlockFp8TuningCase:
    M: int
    N: int
    K: int
    block_n: int
    block_k: int
    out_dtype: torch.dtype
    device: torch.device

    @property
    def block_size(self) -> list[int]:
        return [self.block_n, self.block_k]


@dataclass(frozen=True)
class TritonBackend:
    """What the installed Triton reports about the current GPU."""

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
        max_shared_mem=properties["max_shared_mem"],
        options=options,
        option_defaults={name: getattr(defaults, name) for name in options},
    )


def triton_backend() -> TritonBackend:
    return _triton_backend(torch.accelerator.current_device_index())


def valid_m_buckets(max_num_batched_tokens: int) -> list[int]:
    """Clip the M grid and include limits below its maximum."""
    if max_num_batched_tokens <= 0:
        return []
    buckets = [m for m in DEFAULT_M_BUCKETS if m <= max_num_batched_tokens]
    if max_num_batched_tokens < DEFAULT_M_BUCKETS[-1]:
        buckets.append(max_num_batched_tokens)
    return sorted(set(buckets))


def _key(config: Config) -> ConfigKey:
    return tuple(sorted(config.items()))


def normalize_config(config: Config, case: BlockFp8TuningCase) -> Config | None:
    """Normalize effective values and reject K tiles spanning scale blocks."""
    if case.block_k % config["BLOCK_SIZE_K"] != 0:
        return None
    # Group order is irrelevant when M fits one tile.
    if triton.cdiv(case.M, config["BLOCK_SIZE_M"]) == 1:
        return {**config, "GROUP_SIZE_M": 1}
    return config


def default_config(case: BlockFp8TuningCase) -> Config:
    config = normalize_config(_default_w8a8_block_fp8_config(case.block_size), case)
    assert config is not None
    return config


def search_axes(case: BlockFp8TuningCase) -> dict[str, tuple[int, ...]]:
    # An M tile taller than next_pow2(M) only masks out rows.
    max_block_m = max(16, triton.next_power_of_2(case.M))
    return {
        **_TILE_AXES,
        "BLOCK_SIZE_M": tuple(
            m for m in _TILE_AXES["BLOCK_SIZE_M"] if m <= max_block_m
        ),
        **triton_backend().options,
    }


def _fits_shared_memory(config: Config) -> bool:
    backend = triton_backend()
    tile_bytes = (config["BLOCK_SIZE_M"] + config["BLOCK_SIZE_N"]) * config[
        "BLOCK_SIZE_K"
    ]
    return tile_bytes * max(1, config["num_stages"] - 1) <= backend.max_shared_mem


def _complete(
    config: Config, axes: dict[str, tuple[int, ...]], case: BlockFp8TuningCase
) -> Config | None:
    """Restrict search keys and fill missing backend defaults."""
    defaults = triton_backend().option_defaults
    out = {k: v for k, v in config.items() if k in axes}
    for name in axes:
        if name not in out and name in defaults:
            out[name] = defaults[name]
    if any(name not in out for name in axes):
        return None
    return normalize_config(out, case)


def _canonical(config: Config) -> Config:
    """The config without backend options left at Triton's defaults."""
    defaults = triton_backend().option_defaults
    return {k: v for k, v in config.items() if k not in defaults or v != defaults[k]}


def _random_configs(
    axes: dict[str, tuple[int, ...]],
    case: BlockFp8TuningCase,
    count: int,
    rng: random.Random,
) -> list[Config]:
    configs: dict[ConfigKey, Config] = {}
    for _ in range(50 * count):
        if len(configs) == count:
            break
        config = normalize_config(
            {name: rng.choice(values) for name, values in axes.items()}, case
        )
        if config is not None and _fits_shared_memory(config):
            configs.setdefault(_key(config), config)
    return list(configs.values())


def _neighbors(
    config: Config, axes: dict[str, tuple[int, ...]], case: BlockFp8TuningCase
) -> Iterator[Config]:
    """Configs one step away along a single axis."""
    for name, values in axes.items():
        if config.get(name) not in values:
            continue
        i = values.index(config[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                neighbor = normalize_config({**config, name: values[j]}, case)
                if neighbor is not None and _fits_shared_memory(neighbor):
                    yield neighbor


def starting_configs(
    case: BlockFp8TuningCase, seeds: Sequence[Config] = ()
) -> list[Config]:
    """The default config, then the seeds the kernel can run, without repeats."""
    starts: dict[ConfigKey, Config] = {}
    for config in [
        default_config(case),
        *(normalize_config(seed, case) for seed in seeds),
    ]:
        if config is not None:
            starts.setdefault(_key(config), config)
    return list(starts.values())


def block_fp8_scratch_bytes(case: BlockFp8TuningCase) -> int:
    """Peak scratch memory: float32 staging for A and B, plus two outputs."""
    staging = 4 * (case.M * case.K + case.N * case.K)
    outputs = 2 * case.M * case.N * case.out_dtype.itemsize
    return staging + outputs


def make_block_fp8_inputs(case: BlockFp8TuningCase) -> BlockFp8Inputs:
    fp8_dtype = current_platform.fp8_dtype()
    fp8_info = torch.finfo(fp8_dtype)
    device = case.device

    def rand_fp8(rows: int, cols: int) -> torch.Tensor:
        x = torch.rand(rows, cols, dtype=torch.float32, device=device) - 0.5
        x = x * 2 * fp8_info.max
        return x.clamp(min=fp8_info.min, max=fp8_info.max).to(fp8_dtype)

    A = rand_fp8(case.M, case.K)
    B = rand_fp8(case.N, case.K)
    n_tiles = triton.cdiv(case.N, case.block_n)
    k_tiles = triton.cdiv(case.K, case.block_k)
    As = torch.rand(case.M, k_tiles, dtype=torch.float32, device=device) * 1e-2
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device) * 1e-2
    return A, B, As, Bs


def _run(
    case: BlockFp8TuningCase, inputs: BlockFp8Inputs, config: Config
) -> torch.Tensor:
    A, B, As, Bs = inputs
    return w8a8_triton_block_scaled_mm(
        A, B, As, Bs, case.block_size, case.out_dtype, config=config
    )


def benchmark_block_fp8_config(
    case: BlockFp8TuningCase,
    config: Config,
    *,
    inputs: BlockFp8Inputs | None = None,
    num_iters: int = 10,
    num_warmup: int = 5,
    graph_batch_size: int = 10,
    cutoff_us: float | None = None,
) -> float | None:
    """Benchmark a config, stopping early when its first replay exceeds the cutoff."""
    try:
        if inputs is None:
            inputs = make_block_fp8_inputs(case)

        for _ in range(num_warmup):
            _run(case, inputs, config)
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_batch_size):
                _run(case, inputs, config)
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
            logger.debug("Block FP8 config %s failed for %s: %s", config, case, e)
        return None


def validate_block_fp8_config(
    case: BlockFp8TuningCase,
    config: Config,
    *,
    inputs: BlockFp8Inputs | None = None,
    reference: torch.Tensor | None = None,
    rtol: float = 1e-2,
) -> bool:
    """Check a candidate against the default launch config on identical inputs."""
    if inputs is None:
        inputs = make_block_fp8_inputs(case)
    if reference is None:
        reference = _run(case, inputs, default_config(case))
    out = _run(case, inputs, config)
    ref = reference.float()
    atol = rtol * ref.abs().max().item()
    return torch.allclose(out.float(), ref, rtol=rtol, atol=atol)


def _ranked_configs(timings: dict[ConfigKey, float | None]) -> list[ConfigKey]:
    timed = ((elapsed, key) for key, elapsed in timings.items() if elapsed is not None)
    return [key for _, key in sorted(timed)]


def _benchmark_configs(
    configs: Sequence[Config],
    timings: dict[ConfigKey, float | None],
    *,
    max_configs: int,
    best_us: float,
    benchmark: ConfigBenchmark,
) -> float:
    new = {_key(config): config for config in configs if _key(config) not in timings}
    for key, config in list(new.items())[: max_configs - len(timings)]:
        elapsed_us = benchmark(config, best_us * _EARLY_EXIT_RATIO)
        timings[key] = elapsed_us
        if elapsed_us is not None:
            best_us = min(best_us, elapsed_us)
    return best_us


def _search_configs(
    case: BlockFp8TuningCase,
    axes: dict[str, tuple[int, ...]],
    seeds: Sequence[Config],
    max_configs: int,
    rng: random.Random,
    benchmark: ConfigBenchmark,
) -> dict[ConfigKey, float | None]:
    starts = [_complete(config, axes, case) for config in starting_configs(case, seeds)]
    initial = [config for config in starts if config is not None]
    initial += _random_configs(axes, case, _RANDOM_STARTS, rng)

    timings: dict[ConfigKey, float | None] = {}
    best_us = _benchmark_configs(
        initial,
        timings,
        max_configs=max_configs,
        best_us=float("inf"),
        benchmark=benchmark,
    )
    expanded: set[ConfigKey] = set()
    while len(timings) < max_configs:
        center = next(
            (key for key in _ranked_configs(timings) if key not in expanded), None
        )
        if center is None:
            break
        expanded.add(center)
        neighbors = list(_neighbors(dict(center), axes, case))
        rng.shuffle(neighbors)
        best_us = _benchmark_configs(
            neighbors,
            timings,
            max_configs=max_configs,
            best_us=best_us,
            benchmark=benchmark,
        )
    return timings


def tune_block_fp8_case(
    case: BlockFp8TuningCase,
    *,
    seeds: Sequence[Config] = (),
    max_configs: int = MAX_CONFIGS_PER_BUCKET,
    num_iters: int = 10,
    num_warmup: int = 5,
) -> Config | None:
    """Find a correct config with a bounded random and local search."""
    rng = random.Random(case.M)
    inputs = make_block_fp8_inputs(case)
    default = default_config(case)
    reference = _run(case, inputs, default)
    axes = search_axes(case)

    def time_config(config: Config, cutoff_us: float | None = None) -> float | None:
        return benchmark_block_fp8_config(
            case,
            config,
            inputs=inputs,
            num_iters=num_iters,
            num_warmup=num_warmup,
            cutoff_us=cutoff_us,
        )

    timings = _search_configs(case, axes, seeds, max_configs, rng, time_config)

    winner = next(
        (
            _canonical(dict(key))
            for key in _ranked_configs(timings)
            if validate_block_fp8_config(
                case, dict(key), inputs=inputs, reference=reference
            )
        ),
        None,
    )
    if winner is None:
        return None

    # Recheck against the default before accepting a close result.
    winner_us = default_us = float("inf")
    for _ in range(_HEAD_TO_HEAD_ROUNDS):
        default_us = min(default_us, time_config(default) or float("inf"))
        if winner != default:
            winner_us = min(winner_us, time_config(winner) or float("inf"))
    if default_us < winner_us * _MIN_SPEEDUP:
        winner, winner_us = default, default_us

    logger.info(
        "Block FP8 tuned N=%d,K=%d,M=%d to %s (%.2f us, default %.2f us, "
        "%d configs timed).",
        case.N,
        case.K,
        case.M,
        winner,
        winner_us,
        default_us,
        len(timings),
    )
    return winner
