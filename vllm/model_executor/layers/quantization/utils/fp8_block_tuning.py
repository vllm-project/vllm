# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable warmup autotuning helpers for the W8A8 block FP8 Triton GEMM."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import triton

from .fp8_utils import _default_w8a8_block_fp8_config, w8a8_triton_block_scaled_mm

logger = init_logger(__name__)

Config = dict[str, int]
BlockFp8Inputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# Search space and M grid mirror benchmarks/kernels/benchmark_w8a8_block_fp8.py,
# which produced the bundled config tables.
NUM_STAGES_CHOICES = [2, 3, 4, 5]
BLOCK_SIZE_M_CHOICES = [16, 32, 64, 128, 256]
BLOCK_SIZE_K_CHOICES = [64, 128]
BLOCK_SIZE_N_CHOICES = [32, 64, 128, 256]
NUM_WARPS_CHOICES = [4, 8]
GROUP_SIZE_M_CHOICES = [1, 16, 32, 64]

DEFAULT_M_BUCKETS = (
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096,
)  # fmt: skip

# A candidate whose first timed replay is this much slower than the best so
# far cannot win, so its full timing is skipped.
_EARLY_EXIT_RATIO = 1.5


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


def search_space(case: BlockFp8TuningCase) -> list[Config]:
    """The offline search space, minus configs that cannot differ for this M.

    BLOCK_SIZE_M above next_pow2(M) only masks out rows, and with a single M
    tile every GROUP_SIZE_M launches the same tile order.
    """
    bsm_ceiling = max(16, triton.next_power_of_2(case.M))
    configs = []
    for num_stages, bsm, bsk, bsn, num_warps, group_m in product(
        NUM_STAGES_CHOICES,
        BLOCK_SIZE_M_CHOICES,
        BLOCK_SIZE_K_CHOICES,
        BLOCK_SIZE_N_CHOICES,
        NUM_WARPS_CHOICES,
        GROUP_SIZE_M_CHOICES,
    ):
        if bsm > bsm_ceiling or case.block_k % bsk != 0:
            continue
        if triton.cdiv(case.M, bsm) == 1 and group_m != 1:
            continue
        configs.append(
            {
                "BLOCK_SIZE_M": bsm,
                "BLOCK_SIZE_N": bsn,
                "BLOCK_SIZE_K": bsk,
                "GROUP_SIZE_M": group_m,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )
    return configs


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
    case: BlockFp8TuningCase, inputs: BlockFp8Inputs, config: dict[str, Any]
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
    """Benchmark one config on disposable buffers and return microseconds.

    If the first timed replay is slower than cutoff_us per call, that rough
    time is returned without the remaining replays.
    """
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
        reference = _run(case, inputs, _default_w8a8_block_fp8_config(case.block_size))
    out = _run(case, inputs, config)
    ref = reference.float()
    atol = rtol * ref.abs().max().item()
    return torch.allclose(out.float(), ref, rtol=rtol, atol=atol)


def tune_block_fp8_case(
    case: BlockFp8TuningCase,
    *,
    candidates: list[Config] | None = None,
    num_iters: int = 10,
    num_warmup: int = 5,
) -> Config | None:
    """Return the fastest candidate that reproduces the default config's output."""
    if candidates is None:
        candidates = search_space(case)
    inputs = make_block_fp8_inputs(case)
    reference = _run(case, inputs, _default_w8a8_block_fp8_config(case.block_size))

    best_us = float("inf")
    timings: list[tuple[float, int]] = []
    for index, config in enumerate(candidates):
        elapsed_us = benchmark_block_fp8_config(
            case,
            config,
            inputs=inputs,
            num_iters=num_iters,
            num_warmup=num_warmup,
            cutoff_us=best_us * _EARLY_EXIT_RATIO,
        )
        if elapsed_us is None:
            continue
        timings.append((elapsed_us, index))
        best_us = min(best_us, elapsed_us)

    for elapsed_us, index in sorted(timings):
        config = candidates[index]
        if validate_block_fp8_config(case, config, inputs=inputs, reference=reference):
            logger.info(
                "Block FP8 tuned N=%d,K=%d,M=%d to %s (%.2f us).",
                case.N,
                case.K,
                case.M,
                config,
                elapsed_us,
            )
            return config
    return None
