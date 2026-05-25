# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepSeek V4 sparse compressor Triton and CuTe DSL paths.

The benchmark uses DeepSeek V4's vLLM page sizing by default:
main MLA block size 256, so C4 stores 64 compressed tokens per KV block and
C128 stores 2 compressed tokens per KV block.

Measured variants:

* C4 Triton fused baseline
* C4 CuTe DSL fused kernel
* C4 CuTe DSL split kernels, using the same split path as C128
* C128 Triton fused baseline
* C128 CuTe DSL split kernels

Run from the vLLM repo root, for example:

    ../.venv/bin/python tools/benchmark_dsv4_sparse_compressor_cutedsl.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch
from compare_dsv4_sparse_compressor_cutedsl import (
    DEEPSEEK_V4_MAIN_BLOCK_SIZE,
    FP8_MAX,
    HEAD_SIZE,
    QUANT_BLOCK,
    ROPE_HEAD_DIM,
    SCALE_DIM,
    TOKEN_STRIDE,
    SparseCase,
    compare,
    empty_k_cache,
    make_case,
    storage_block_size,
)

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)
from vllm.models.deepseek_v4.common.ops.sparse_attn_compress_cutedsl import (
    _compress_kv_sparse_attn_cutedsl,
    _fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl,
    _norm_rope_insert_sparse_attn_cutedsl,
)
from vllm.triton_utils import triton

DEFAULT_BENCH_TOKENS = 512


@dataclass
class Runner:
    case: SparseCase
    label: str
    output: torch.Tensor
    launch: Callable[[], None]
    launches_per_iter: int


@dataclass
class BenchResult:
    case_name: str
    backend: str
    tokens: int
    kv_cache_block_size: int
    launches_per_iter: int
    latency_ms: float

    @property
    def us_per_token(self) -> float:
        return self.latency_ms * 1000.0 / self.tokens


def launch_triton(case: SparseCase, k_cache: torch.Tensor) -> None:
    _fused_kv_compress_norm_rope_insert_sparse_attn[(case.positions.numel(),)](
        case.state_cache,
        case.state_cache.stride(0),
        case.state_cache.stride(1),
        case.token_to_req_indices,
        case.positions,
        case.slot_mapping,
        case.block_table,
        case.block_table.stride(0),
        case.state_block_size,
        case.rms_norm_weight,
        1.0e-6,
        case.cos_sin_cache,
        case.cos_sin_cache.stride(0),
        k_cache,
        case.kv_slot_mapping,
        case.kv_cache_block_size,
        HEAD_SIZE=HEAD_SIZE,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(HEAD_SIZE),
        STATE_WIDTH=case.state_width,
        COMPRESS_RATIO=case.compress_ratio,
        OVERLAP=case.overlap,
        ROPE_HEAD_DIM=ROPE_HEAD_DIM,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=QUANT_BLOCK,
        TOKEN_STRIDE=TOKEN_STRIDE,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        num_warps=4,
    )


def launch_c4_fused_cutedsl(case: SparseCase, k_cache: torch.Tensor) -> None:
    _fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
        case.state_cache,
        case.token_to_req_indices,
        case.positions,
        case.slot_mapping,
        case.block_table,
        case.state_block_size,
        case.rms_norm_weight,
        1.0e-6,
        case.cos_sin_cache,
        k_cache,
        case.kv_slot_mapping,
        case.kv_cache_block_size,
        k_cache.stride(0),
        head_size=HEAD_SIZE,
        state_width=case.state_width,
        rope_head_dim=ROPE_HEAD_DIM,
        fp8_max=FP8_MAX,
        quant_block=QUANT_BLOCK,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
        compress_ratio=case.compress_ratio,
        overlap=case.overlap,
    )


def launch_split_cutedsl(
    case: SparseCase,
    k_cache: torch.Tensor,
    compressed_kv: torch.Tensor,
) -> None:
    _compress_kv_sparse_attn_cutedsl(
        case.state_cache,
        case.token_to_req_indices,
        case.positions,
        case.slot_mapping,
        case.block_table,
        case.state_block_size,
        compressed_kv,
        head_size=HEAD_SIZE,
        state_width=case.state_width,
        compress_ratio=case.compress_ratio,
        overlap=case.overlap,
    )
    _norm_rope_insert_sparse_attn_cutedsl(
        compressed_kv,
        case.positions,
        case.slot_mapping,
        case.rms_norm_weight,
        1.0e-6,
        case.cos_sin_cache,
        k_cache,
        case.kv_slot_mapping,
        case.kv_cache_block_size,
        k_cache.stride(0),
        head_size=HEAD_SIZE,
        rope_head_dim=ROPE_HEAD_DIM,
        fp8_max=FP8_MAX,
        quant_block=QUANT_BLOCK,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
        compress_ratio=case.compress_ratio,
    )


def make_runner(case: SparseCase, backend: str) -> Runner:
    k_cache = empty_k_cache(case)
    if backend == "triton":
        return Runner(
            case=case,
            label="triton_fused",
            output=k_cache,
            launch=lambda: launch_triton(case, k_cache),
            launches_per_iter=1,
        )
    if backend == "c4_fused":
        if case.compress_ratio != 4:
            raise ValueError("c4_fused backend requires compress_ratio=4.")
        return Runner(
            case=case,
            label="c4_fused_cutedsl",
            output=k_cache,
            launch=lambda: launch_c4_fused_cutedsl(case, k_cache),
            launches_per_iter=1,
        )
    if backend == "split":
        compressed_kv = torch.empty(
            (case.positions.numel(), HEAD_SIZE),
            dtype=torch.float32,
            device=case.positions.device,
        )
        label = f"c{case.compress_ratio}_split_cutedsl"
        return Runner(
            case=case,
            label=label,
            output=k_cache,
            launch=lambda: launch_split_cutedsl(case, k_cache, compressed_kv),
            launches_per_iter=2,
        )
    raise ValueError(f"Unknown backend: {backend}.")


def benchmark_runner(runner: Runner, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        runner.launch()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        runner.launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def make_sparse_case(
    *,
    compress_ratio: int,
    num_tokens: int,
    vllm_main_block_size: int,
    kv_cache_block_size: int | None,
    device: torch.device,
    seed: int,
) -> SparseCase:
    storage_bs = (
        kv_cache_block_size
        if kv_cache_block_size is not None
        else storage_block_size(vllm_main_block_size, compress_ratio)
    )
    return make_case(
        compress_ratio=compress_ratio,
        num_tokens=num_tokens,
        kv_cache_block_size=storage_bs,
        device=device,
        seed=seed + compress_ratio,
    )


def print_results(results: list[BenchResult]) -> None:
    print("\nResults")
    print("case,backend,tokens,kv_cache_block_size,launches,latency_ms,us_per_token")
    for result in results:
        print(
            f"{result.case_name},{result.backend},{result.tokens},"
            f"{result.kv_cache_block_size},{result.launches_per_iter},"
            f"{result.latency_ms:.6f},{result.us_per_token:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=DEFAULT_BENCH_TOKENS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--vllm-main-block-size",
        type=int,
        default=DEEPSEEK_V4_MAIN_BLOCK_SIZE,
        help="DeepSeek V4 sparse MLA block_size before compression.",
    )
    parser.add_argument(
        "--kv-cache-block-size",
        type=int,
        default=None,
        help=(
            "Override compressed KV cache storage block size for all cases. "
            "If omitted, uses vllm_main_block_size // compress_ratio."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case", choices=["all", "c4", "c128"], default="all")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton and CuTe DSL kernels.")
    if args.tokens <= 0 or args.warmup < 0 or args.iters <= 0:
        raise ValueError("--tokens and --iters must be positive; --warmup >= 0.")

    device = torch.device(args.device)
    results: list[BenchResult] = []

    ratios: list[int] = []
    if args.case in ("all", "c4"):
        ratios.append(4)
    if args.case in ("all", "c128"):
        ratios.append(128)

    for ratio in ratios:
        case = make_sparse_case(
            compress_ratio=ratio,
            num_tokens=args.tokens,
            vllm_main_block_size=args.vllm_main_block_size,
            kv_cache_block_size=args.kv_cache_block_size,
            device=device,
            seed=args.seed,
        )
        print(
            f"\nCase {case.name}: tokens={case.positions.numel()} "
            f"kv_cache_block_size={case.kv_cache_block_size}"
        )

        backend_names = ["triton"]
        if ratio == 4:
            backend_names.extend(["c4_fused", "split"])
        else:
            backend_names.append("split")

        runners = [make_runner(case, backend) for backend in backend_names]

        baseline = runners[0]
        baseline.launch()
        torch.cuda.synchronize()
        if not args.skip_correctness:
            for runner in runners[1:]:
                runner.launch()
                compare(
                    case,
                    runner.label,
                    baseline.output,
                    runner.output,
                    atol=args.atol,
                    rtol=args.rtol,
                    require_byte_equal=False,
                )

        for runner in runners:
            latency_ms = benchmark_runner(runner, args.warmup, args.iters)
            results.append(
                BenchResult(
                    case_name=case.name,
                    backend=runner.label,
                    tokens=case.positions.numel(),
                    kv_cache_block_size=case.kv_cache_block_size,
                    launches_per_iter=runner.launches_per_iter,
                    latency_ms=latency_ms,
                )
            )
            print(f"{runner.label}: {latency_ms:.6f} ms")

    print_results(results)


if __name__ == "__main__":
    main()
