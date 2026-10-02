# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable warmup autotuning helpers for Mamba selective_state_update."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch

from vllm.logger import init_logger

from .mamba_ssm import override_ssm_config
from .mamba_ssm import selective_state_update as triton_selective_state_update

logger = init_logger(__name__)

_BSM_CHOICES_ALL = [4, 8, 16, 32, 64, 128, 256]
NUM_WARPS_CHOICES = [1, 2, 4, 8]

_CUDA_MAX_GRID_DIM = 65535
_MAX_EFFECTIVE_BATCH = 262144

# Mirrors the standalone SSU benchmark's default deployment batch grid. Warmup
# clips this to the actual server request limit and adds the exact max so the
# generated cache can replace the manual benchmark for common serving shapes.
DEFAULT_REQUEST_BATCH_BUCKETS = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048)


@dataclass(frozen=True)
class SSUTuningCase:
    batch: int
    nheads: int
    headdim: int
    dstate: int
    ngroups: int
    dtype: torch.dtype
    state_dtype: torch.dtype
    device: torch.device
    is_blackwell: bool = False

    @property
    def effective_batch(self) -> int:
        return self.batch * self.nheads

    @property
    def cache_dtype_name(self) -> str:
        return str(self.state_dtype).removeprefix("torch.")


def block_size_m_choices(headdim: int) -> list[int]:
    """Return candidate BLOCK_SIZE_M values worth sweeping for headdim."""
    ceiling = 1
    while ceiling < headdim:
        ceiling <<= 1
    return [b for b in _BSM_CHOICES_ALL if b <= ceiling]


def valid_request_batches(max_num_reqs: int) -> list[int]:
    """Benchmark-style request-batch buckets clipped to the server limit."""
    if max_num_reqs <= 0:
        return []

    buckets = [b for b in DEFAULT_REQUEST_BATCH_BUCKETS if b <= max_num_reqs]
    buckets.append(max_num_reqs)
    return sorted(set(buckets))


def make_active_cases(
    batch_sizes: list[int],
    nheads: int,
    ngroups: int,
) -> list[tuple[int, int, int]]:
    """Return sorted, deduped ``(effective_batch, batch, nheads)`` cases."""
    seen: dict[int, int] = {}
    for batch in batch_sizes:
        if batch <= 0 or nheads <= 0:
            continue
        if batch > _CUDA_MAX_GRID_DIM or nheads > _CUDA_MAX_GRID_DIM:
            continue
        if ngroups <= 0 or nheads % ngroups != 0:
            continue
        effective_batch = batch * nheads
        if effective_batch > _MAX_EFFECTIVE_BATCH:
            continue
        seen.setdefault(effective_batch, batch)
    return sorted((eb, batch, nheads) for eb, batch in seen.items())


def make_ssu_inputs(case: SSUTuningCase):
    state = torch.randn(
        case.batch,
        case.nheads,
        case.headdim,
        case.dstate,
        dtype=case.state_dtype,
        device=case.device,
    )
    x = torch.randn(
        case.batch,
        case.nheads,
        case.headdim,
        dtype=case.dtype,
        device=case.device,
    )
    dt = torch.randn_like(x)
    A = -torch.rand(
        case.nheads,
        case.headdim,
        case.dstate,
        dtype=torch.float32,
        device=case.device,
    )
    B = torch.randn(
        case.batch,
        case.ngroups,
        case.dstate,
        dtype=case.dtype,
        device=case.device,
    )
    C = torch.randn_like(B)
    D = torch.randn(
        case.nheads,
        case.headdim,
        dtype=case.dtype,
        device=case.device,
    )
    dt_bias = torch.randn_like(D)
    out = torch.empty_like(x)
    return state, x, dt, A, B, C, D, dt_bias, out


def benchmark_ssu_config(
    case: SSUTuningCase,
    block_size_m: int,
    num_warps: int,
    *,
    num_iters: int = 20,
    num_warmup: int = 5,
    graph_batch_size: int = 5,
) -> float | None:
    """Benchmark one config on disposable buffers and return microseconds."""
    try:
        state, x, dt, A, B, C, D, dt_bias, out = make_ssu_inputs(case)

        def call_kernel() -> None:
            triton_selective_state_update(
                state,
                x,
                dt,
                A,
                B,
                C,
                D=D,
                dt_bias=dt_bias,
                dt_softplus=True,
                out=out,
                is_blackwell=case.is_blackwell,
            )

        with override_ssm_config((block_size_m, num_warps)):
            for _ in range(num_warmup):
                call_kernel()
            torch.accelerator.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(graph_batch_size):
                    call_kernel()
            torch.accelerator.synchronize()

            for _ in range(3):
                graph.replay()
            torch.accelerator.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            latencies: list[float] = []
            for _ in range(num_iters):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                latencies.append(start.elapsed_time(end))
            graph.reset()
        return sum(latencies) / (num_iters * graph_batch_size) * 1000
    except Exception as e:
        if "OutOfResources" not in str(e):
            logger.debug(
                "SSU config M=%d,w=%d failed for %s: %s",
                block_size_m,
                num_warps,
                case,
                e,
            )
        return None


def tune_ssu_case(
    case: SSUTuningCase,
    *,
    num_iters: int = 20,
    num_warmup: int = 5,
) -> dict[str, int] | None:
    best_time = float("inf")
    best_config: dict[str, int] | None = None

    for block_size_m, num_warps in product(
        block_size_m_choices(case.headdim), NUM_WARPS_CHOICES
    ):
        elapsed_us = benchmark_ssu_config(
            case,
            block_size_m,
            num_warps,
            num_iters=num_iters,
            num_warmup=num_warmup,
        )
        if elapsed_us is None:
            continue
        if elapsed_us < best_time:
            best_time = elapsed_us
            best_config = {
                "BLOCK_SIZE_M": block_size_m,
                "num_warps": num_warps,
            }

    if best_config is not None:
        logger.info(
            "Mamba SSU tuned effective_batch=%d to %s (%.2f us).",
            case.effective_batch,
            best_config,
            best_time,
        )
    return best_config
