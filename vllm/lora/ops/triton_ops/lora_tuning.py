# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable warmup autotuning helpers for the LoRA shrink and expand kernels."""

from __future__ import annotations

from collections.abc import Iterator
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

# Triton compiles M == 1 and M % 16 == 0 separately, so every config on this
# grid compiles twice per kernel variant.
LORA_M_BUCKETS = (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# Pruned from a wide sweep on MI300X (Llama-3.2-3B shapes at ranks 16 and 64,
# 1 and 4 adapters, M from 1 to 4096): these lists stay within 3% of the wide
# space's best. Shrink also tries BLOCK_N equal to the rank.
SHRINK_BLOCK_M_CHOICES = [16, 32, 64, 128]
SHRINK_BLOCK_N_CHOICES = [16]
SHRINK_BLOCK_K_CHOICES = [64, 128]
SHRINK_SPLIT_K_CHOICES = [2, 4, 8, 16, 32]
EXPAND_BLOCK_M_CHOICES = [16, 32, 64]
EXPAND_BLOCK_N_CHOICES = [32, 64, 128, 256]
EXPAND_BLOCK_K = 16
EXPAND_NUM_WARPS_CHOICES = [2, 4, 8]

# A candidate whose first timed replay is this much slower than the best so
# far cannot win, so its full timing is skipped.
_EARLY_EXIT_RATIO = 1.5


@dataclass(frozen=True)
class LoRATuningCase:
    op_type: str  # "shrink" or "expand"
    M: int
    # Adapter slots (LoRAConfig.max_loras); the kernels see max_loras + 1 ids.
    max_loras: int
    rank: int
    # Hidden size of each slice: the input size for shrink (equal for every
    # slice), the output size for expand.
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
    """Grid entries up to the first one at or above max_tokens, so the
    runtime's nearest-M lookup never leaves the tuned range."""
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


def search_space(case: LoRATuningCase) -> list[Config]:
    """The default config first, then candidates that can differ for this case.

    BLOCK_M above next_pow2(M) only masks out rows, and a shrink K tile wider
    than K only masks out columns.
    """
    bm_ceiling = max(16, next_power_of_2(case.M))
    configs = [default_config(case)]
    if case.op_type == "shrink":
        rank_block = min(64, max(16, next_power_of_2(case.rank)))
        block_n_choices = sorted({*SHRINK_BLOCK_N_CHOICES, rank_block})
        for block_m, block_n, block_k, split_k in product(
            SHRINK_BLOCK_M_CHOICES,
            block_n_choices,
            SHRINK_BLOCK_K_CHOICES,
            SHRINK_SPLIT_K_CHOICES,
        ):
            if block_m > bm_ceiling or block_k * split_k > case.hidden_size:
                continue
            configs.append(
                {
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": block_k,
                    "split_k": split_k,
                    "num_warps": 4,
                    "num_ctas": 1,
                    "group_size_m": 8,
                    "num_stages": 2,
                }
            )
    else:
        for block_m, block_n, num_warps in product(
            EXPAND_BLOCK_M_CHOICES, EXPAND_BLOCK_N_CHOICES, EXPAND_NUM_WARPS_CHOICES
        ):
            if block_m > bm_ceiling:
                continue
            configs.append(
                {
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": EXPAND_BLOCK_K,
                    "num_warps": num_warps,
                    "num_ctas": 1,
                    "num_stages": 2,
                }
            )
    return [c for i, c in enumerate(configs) if c not in configs[:i]]


def lora_scratch_bytes(case: LoRATuningCase) -> int:
    """Peak scratch memory: input, weights, and four copies of the output
    (working, initial, reference, candidate)."""
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
    """Scratch tensors shaped like PunicaWrapperGPU's add_lora_linear call."""
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
    """Keep the scratch weights out of the LoRA pointer caches.

    The caches are keyed by data_ptr() and hold references to the weights, so a
    leftover entry would pin scratch memory and could be hit by a later tensor
    at the same address. An entry already present under these addresses is
    stale, because the scratch weights are fresh allocations.
    """
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
    """Benchmark one config on disposable buffers and return microseconds.

    If the first timed replay is slower than cutoff_us per call, that rough
    time is returned without the remaining replays.
    """
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
    """Check a candidate against the default launch config on identical inputs."""
    if inputs is None:
        inputs = make_lora_inputs(case)
    with _isolated_ptr_cache(case, inputs):
        if reference is None:
            reference = _output(case, inputs, default_config(case))
        out = _output(case, inputs, config)
    ref = reference.float()
    atol = rtol * ref.abs().max().item()
    return torch.allclose(out.float(), ref, rtol=rtol, atol=atol)


def tune_lora_case(
    case: LoRATuningCase,
    *,
    candidates: list[Config] | None = None,
    num_iters: int = 10,
    num_warmup: int = 5,
) -> Config | None:
    """Return the fastest candidate that reproduces the default config's output."""
    if candidates is None:
        candidates = search_space(case)
    default = default_config(case)
    inputs = make_lora_inputs(case)
    with _isolated_ptr_cache(case, inputs):
        reference = _output(case, inputs, default)

        best_us = float("inf")
        default_us: float | None = None
        timings: list[tuple[float, int]] = []
        for index, config in enumerate(candidates):
            elapsed_us = benchmark_lora_config(
                case,
                config,
                inputs=inputs,
                num_iters=num_iters,
                num_warmup=num_warmup,
                cutoff_us=best_us * _EARLY_EXIT_RATIO,
            )
            if elapsed_us is None:
                continue
            if config == default:
                default_us = elapsed_us
            timings.append((elapsed_us, index))
            best_us = min(best_us, elapsed_us)

        for elapsed_us, index in sorted(timings):
            config = candidates[index]
            if validate_lora_config(case, config, inputs=inputs, reference=reference):
                logger.info(
                    "LoRA %s tuned slices=%s,rank=%d,M=%d to %s "
                    "(%.2f us, default %s us).",
                    case.op_type,
                    case.slice_sizes,
                    case.rank,
                    case.M,
                    config,
                    elapsed_us,
                    "n/a" if default_us is None else f"{default_us:.2f}",
                )
                return config
    return None
