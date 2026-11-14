# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable warmup autotuning helpers for the LoRA shrink and expand kernels."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import next_power_of_2

from .lora_expand_op import lora_expand
from .lora_kernel_metadata import LoRAKernelMeta
from .lora_shrink_op import lora_shrink
from .utils import (
    _LORA_A_PTR_DICT,
    _LORA_B_PTR_DICT,
    _default_lora_op_config,
    override_lora_op_config,
)

logger = init_logger(__name__)

Config = dict[str, int]
ConfigKey = tuple[tuple[str, int], ...]
ConfigBenchmark = Callable[[Config, float | None], float | None]

# Representative token-count buckets used by config lookup.
LORA_M_BUCKETS = (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# Maximum distinct configs timed per bucket, including starting points.
MAX_CONFIGS_PER_BUCKET = 12

SHRINK_BLOCK_M_CHOICES = [16, 32, 64, 128]
SHRINK_BLOCK_N_CHOICES = [16]
SHRINK_BLOCK_K_CHOICES = [64, 128]
SHRINK_SPLIT_K_CHOICES = [2, 4, 8, 16, 32]
EXPAND_BLOCK_M_CHOICES = [16, 32, 64]
EXPAND_BLOCK_N_CHOICES = [32, 64, 128, 256]
EXPAND_BLOCK_K = 16
EXPAND_NUM_WARPS_CHOICES = [2, 4, 8]

# Stop timing a candidate after one replay when it exceeds this ratio.
_EARLY_EXIT_RATIO = 1.5

_HEAD_TO_HEAD_ROUNDS = 2

# Required speedup over the default.
_MIN_SPEEDUP = 1.02

_SHRINK_FIXED: Config = {
    "num_warps": 4,
    "num_ctas": 1,
    "group_size_m": 8,
    "num_stages": 2,
}
_EXPAND_FIXED: Config = {"block_k": EXPAND_BLOCK_K, "num_ctas": 1, "num_stages": 2}


@dataclass(frozen=True)
class LoRATuningCase:
    op_type: str  # "shrink" or "expand"
    M: int
    # Adapter slots; kernel metadata adds a no-LoRA slot.
    max_loras: int
    rank: int
    # Input sizes for shrink, output sizes for expand.
    slice_sizes: tuple[int, ...]
    add_inputs: bool | None  # expand only
    dtype: torch.dtype
    device: torch.device

    @property
    def num_slices(self) -> int:
        return len(self.slice_sizes)

    @property
    def hidden_size(self) -> int:
        return max(self.slice_sizes)

    @property
    def num_active_loras(self) -> int:
        return min(self.max_loras, self.M)


@dataclass
class LoRAInputs:
    x: torch.Tensor
    weights: list[torch.Tensor]
    out: torch.Tensor
    out_init: torch.Tensor
    meta_args: tuple[torch.Tensor, ...]


def lora_m_buckets(max_tokens: int) -> list[int]:
    """Return representative M buckets, capped at the largest grid entry."""
    buckets: list[int] = []
    if max_tokens <= 0:
        return buckets
    for m in LORA_M_BUCKETS:
        buckets.append(m)
        if m >= max_tokens:
            break
    return buckets


def default_config(case: LoRATuningCase) -> Config:
    config = _default_lora_op_config(case.op_type, case.M, case.num_slices, case.rank)
    return {k: v for k, v in config.items() if v is not None}


def _key(config: Config) -> ConfigKey:
    return tuple(sorted(config.items()))


def _block_m_ceiling(case: LoRATuningCase) -> int:
    return max(16, next_power_of_2(case.M))


def search_axes(case: LoRATuningCase) -> dict[str, tuple[int, ...]]:
    """Return the values searched for each config field."""
    bm_ceiling = _block_m_ceiling(case)
    if case.op_type == "shrink":
        rank_block = min(64, max(16, next_power_of_2(case.rank)))
        return {
            "block_m": tuple(m for m in SHRINK_BLOCK_M_CHOICES if m <= bm_ceiling),
            "block_n": tuple(sorted({*SHRINK_BLOCK_N_CHOICES, rank_block})),
            "block_k": tuple(SHRINK_BLOCK_K_CHOICES),
            "split_k": tuple(SHRINK_SPLIT_K_CHOICES),
        }
    return {
        "block_m": tuple(m for m in EXPAND_BLOCK_M_CHOICES if m <= bm_ceiling),
        "block_n": tuple(EXPAND_BLOCK_N_CHOICES),
        "num_warps": tuple(EXPAND_NUM_WARPS_CHOICES),
    }


def normalize_config(config: Config, case: LoRATuningCase) -> Config | None:
    """Return the config if it fits the case."""
    if config["block_m"] > _block_m_ceiling(case):
        return None
    if (
        case.op_type == "shrink"
        and config["block_k"] * config["split_k"] > case.hidden_size
    ):
        return None
    return config


def _grid_configs(
    case: LoRATuningCase, axes: dict[str, tuple[int, ...]]
) -> list[Config]:
    fixed = _SHRINK_FIXED if case.op_type == "shrink" else _EXPAND_FIXED
    configs = (
        normalize_config({**dict(zip(axes, values)), **fixed}, case)
        for values in product(*axes.values())
    )
    return [config for config in configs if config is not None]


def _snap(
    config: Config,
    axes: dict[str, tuple[int, ...]],
    grid: Sequence[Config],
) -> Config | None:
    """The grid config closest to config, measured in powers of two."""

    def distance(grid_config: Config) -> float:
        return sum(abs(math.log2(grid_config[k] / config[k])) for k in axes)

    return min(grid, key=distance, default=None)


def _neighbors(
    config: Config, axes: dict[str, tuple[int, ...]], case: LoRATuningCase
) -> Iterator[Config]:
    """Grid configs one step away along a single axis."""
    fixed = _SHRINK_FIXED if case.op_type == "shrink" else _EXPAND_FIXED
    if (
        set(config) != {*axes, *fixed}
        or any(config.get(name) not in values for name, values in axes.items())
        or any(config.get(name) != value for name, value in fixed.items())
    ):
        return
    for name, values in axes.items():
        i = values.index(config[name])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                neighbor = normalize_config({**config, name: values[j]}, case)
                if neighbor is not None:
                    yield neighbor


def starting_configs(
    case: LoRATuningCase, seeds: Sequence[Config] = ()
) -> list[Config]:
    """Return the default plus deduplicated, grid-aligned starting points."""
    default = default_config(case)
    axes = search_axes(case)
    grid = _grid_configs(case, axes)
    starts = {_key(default): default}
    for config in [default, *seeds]:
        snapped = _snap(config, axes, grid)
        if snapped is not None:
            starts.setdefault(_key(snapped), snapped)
    return list(starts.values())


def lora_scratch_bytes(case: LoRATuningCase) -> int:
    """Return peak scratch memory in bytes."""
    itemsize = case.dtype.itemsize
    total_n = sum(case.slice_sizes)
    if case.op_type == "shrink":
        x = case.M * case.hidden_size * itemsize
        weights = case.num_slices * case.max_loras * case.rank * case.hidden_size
        out = case.num_slices * case.M * case.rank * 4
        return x + weights * itemsize + 4 * out
    x = case.num_slices * case.M * case.rank * 4
    weights = case.max_loras * total_n * case.rank * itemsize
    out = case.M * total_n * itemsize
    return x + weights + 4 * out


def make_lora_inputs(case: LoRATuningCase) -> LoRAInputs:
    """Create representative scratch inputs."""
    device, dtype = case.device, case.dtype
    meta = LoRAKernelMeta.make(case.max_loras, case.M, device)
    # Contiguous runs of tokens, one per active adapter.
    mapping = torch.arange(case.M, device=device) * case.num_active_loras // case.M
    meta.prepare_tensors(mapping.to(torch.int32))
    meta_args = meta.meta_args(case.M, specialize_active_lora=False)

    if case.op_type == "shrink":
        K = case.hidden_size
        x = torch.randn(case.M, K, dtype=dtype, device=device)
        weights = [
            torch.randn(case.max_loras, 1, case.rank, K, dtype=dtype, device=device)
            / K**0.5
            for _ in case.slice_sizes
        ]
        out = torch.empty(
            case.num_slices, case.M, case.rank, dtype=torch.float32, device=device
        )
    else:
        # The shrink output feeds expand as float32.
        x = torch.randn(
            case.num_slices, case.M, case.rank, dtype=torch.float32, device=device
        )
        weights = [
            torch.randn(case.max_loras, 1, n, case.rank, dtype=dtype, device=device)
            / case.rank**0.5
            for n in case.slice_sizes
        ]
        out = torch.randn(case.M, sum(case.slice_sizes), dtype=dtype, device=device)
    return LoRAInputs(x, weights, out, out.clone(), meta_args)


@contextmanager
def _isolated_ptr_cache(case: LoRATuningCase, inputs: LoRAInputs) -> Iterator[None]:
    """Keep scratch weights out of the global pointer cache."""
    cache = _LORA_A_PTR_DICT if case.op_type == "shrink" else _LORA_B_PTR_DICT
    key = tuple(w.data_ptr() for w in inputs.weights)
    cache.pop(key, None)
    try:
        yield
    finally:
        cache.pop(key, None)


def _run(case: LoRATuningCase, inputs: LoRAInputs) -> None:
    if case.op_type == "shrink":
        lora_shrink(inputs.x, inputs.weights, inputs.out, *inputs.meta_args, 1.0)
    else:
        lora_expand(
            inputs.x,
            inputs.weights,
            inputs.out,
            *inputs.meta_args,
            offset_start=0,
            add_inputs=bool(case.add_inputs),
        )


def _output(case: LoRATuningCase, inputs: LoRAInputs, config: Config) -> torch.Tensor:
    inputs.out.copy_(inputs.out_init)
    with override_lora_op_config(case.op_type, config):
        _run(case, inputs)
    return inputs.out.clone()


def benchmark_lora_config(
    case: LoRATuningCase,
    config: Config,
    *,
    inputs: LoRAInputs | None = None,
    num_iters: int = 10,
    num_warmup: int = 5,
    graph_batch_size: int = 10,
    cutoff_us: float | None = None,
) -> float | None:
    """Return mean latency, stopping early when the first replay exceeds cutoff."""
    try:
        if inputs is None:
            inputs = make_lora_inputs(case)
        with (
            _isolated_ptr_cache(case, inputs),
            override_lora_op_config(case.op_type, config),
        ):
            for _ in range(num_warmup):
                _run(case, inputs)
            torch.accelerator.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_batch_size):
                    _run(case, inputs)
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
            logger.debug("LoRA config %s failed for %s: %s", config, case, e)
        return None


def validate_lora_config(
    case: LoRATuningCase,
    config: Config,
    *,
    inputs: LoRAInputs | None = None,
    reference: torch.Tensor | None = None,
    rtol: float = 1e-2,
) -> bool:
    """Compare a candidate with the default output."""
    if inputs is None:
        inputs = make_lora_inputs(case)
    with _isolated_ptr_cache(case, inputs):
        if reference is None:
            reference = _output(case, inputs, default_config(case))
        out = _output(case, inputs, config)
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
    case: LoRATuningCase,
    seeds: Sequence[Config],
    max_configs: int,
    benchmark: ConfigBenchmark,
) -> dict[ConfigKey, float | None]:
    axes = search_axes(case)
    timings: dict[ConfigKey, float | None] = {}
    best_us = _benchmark_configs(
        starting_configs(case, seeds),
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
        best_us = _benchmark_configs(
            list(_neighbors(dict(center), axes, case)),
            timings,
            max_configs=max_configs,
            best_us=best_us,
            benchmark=benchmark,
        )
    return timings


def tune_lora_case(
    case: LoRATuningCase,
    *,
    seeds: Sequence[Config] = (),
    max_configs: int = MAX_CONFIGS_PER_BUCKET,
    num_iters: int = 10,
    num_warmup: int = 5,
) -> Config | None:
    """Return the fastest valid config found within the search budget."""
    default = default_config(case)
    inputs = make_lora_inputs(case)
    with _isolated_ptr_cache(case, inputs):
        reference = _output(case, inputs, default)

    def time_config(config: Config, cutoff_us: float | None = None) -> float | None:
        return benchmark_lora_config(
            case,
            config,
            inputs=inputs,
            num_iters=num_iters,
            num_warmup=num_warmup,
            cutoff_us=cutoff_us,
        )

    timings = _search_configs(case, seeds, max_configs, time_config)
    winner = next(
        (
            dict(key)
            for key in _ranked_configs(timings)
            if validate_lora_config(case, dict(key), inputs=inputs, reference=reference)
        ),
        None,
    )
    if winner is None:
        return None

    # Recheck a close winner against the default.
    winner_us = default_us = float("inf")
    for _ in range(_HEAD_TO_HEAD_ROUNDS):
        default_us = min(default_us, time_config(default) or float("inf"))
        if winner != default:
            winner_us = min(winner_us, time_config(winner) or float("inf"))
    if default_us < winner_us * _MIN_SPEEDUP:
        winner, winner_us = default, default_us

    logger.info(
        "LoRA %s tuned slices=%s,rank=%d,M=%d to %s (%.2f us, default %.2f us, "
        "%d configs timed).",
        case.op_type,
        case.slice_sizes,
        case.rank,
        case.M,
        winner,
        winner_us,
        default_us,
        len(timings),
    )
    return winner
