# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MiniMax-M2 fused MoE top-k routing + FP8 input quantization.

Example:
    python3 benchmarks/kernels/benchmark_minimax_moe_topk_quant.py \
        --m-values 1 2 4 8 16 32 64 128 256 512 1024 \
        --modes eager cudagraph
"""

import argparse
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.topk_quant_kernels import (
    minimax_moe_topk_sigmoid_quant_dispatch,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed


AccuracyResult = tuple[int | None, float | None, float | None, float | None]
MAX_FUSED_TOKENS = 128


@dataclass
class BenchResult:
    mode: str
    m: int
    hidden_size: int
    num_experts: int
    top_k: int
    block_k: int
    topk_us: float
    quant_us: float
    baseline_us: float
    fused_us: float
    speedup: float
    delta_us: float
    topk_id_mismatch: int | None
    topk_weight_max_abs_err: float | None
    a1q_max_abs_err: float | None
    a1q_scale_max_abs_err: float | None


def _baseline_topk_sigmoid(
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = router_logits.shape[0]
    topk_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=router_logits.device
    )
    topk_ids = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=router_logits.device
    )
    token_expert_indices = torch.empty_like(topk_ids)
    ops.topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        router_logits,
        True,
        e_score_correction_bias,
    )
    return topk_weights, topk_ids


def _baseline_quant(
    hidden_states: torch.Tensor,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return per_token_group_quant_fp8(hidden_states, block_k, use_ue8m0=False)


def _baseline_topk_quant(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_weights, topk_ids = _baseline_topk_sigmoid(
        router_logits, e_score_correction_bias, top_k
    )
    a1q, a1q_scale = _baseline_quant(hidden_states, block_k)
    return topk_weights, topk_ids, a1q, a1q_scale


def _dispatch_topk_quant(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return minimax_moe_topk_sigmoid_quant_dispatch(
        hidden_states,
        router_logits,
        e_score_correction_bias,
        top_k,
        block_k,
        MAX_FUSED_TOKENS,
    )


def _bench_us(
    fn: Callable[[], object],
    mode: str,
    quantiles: list[float],
) -> float:
    torch.cuda.synchronize()
    if mode == "cudagraph":
        ms, _, _ = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    elif mode == "eager":
        ms, _, _ = triton.testing.do_bench(fn, quantiles=quantiles)
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")
    return ms * 1000.0


def _max_abs_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return torch.max(torch.abs(actual - expected)).item()


def _check_correctness(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    block_k: int,
) -> AccuracyResult:
    ref_topk_weights, ref_topk_ids, ref_a1q, ref_a1q_scale = _baseline_topk_quant(
        hidden_states,
        router_logits,
        e_score_correction_bias,
        top_k,
        block_k,
    )
    topk_weights, topk_ids, a1q, a1q_scale = _dispatch_topk_quant(
        hidden_states,
        router_logits,
        e_score_correction_bias,
        top_k,
        block_k,
    )
    topk_id_mismatch = torch.count_nonzero(topk_ids != ref_topk_ids).item()
    topk_weight_max_abs_err = _max_abs_err(topk_weights, ref_topk_weights)
    a1q_max_abs_err = _max_abs_err(a1q.float(), ref_a1q.float())
    a1q_scale_max_abs_err = _max_abs_err(a1q_scale, ref_a1q_scale)

    torch.testing.assert_close(topk_ids, ref_topk_ids, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights, ref_topk_weights, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a1q.float(), ref_a1q.float(), atol=0, rtol=0)
    torch.testing.assert_close(a1q_scale, ref_a1q_scale, atol=1e-6, rtol=1e-6)
    return (
        topk_id_mismatch,
        topk_weight_max_abs_err,
        a1q_max_abs_err,
        a1q_scale_max_abs_err,
    )


def _run_one(
    *,
    mode: str,
    m: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    block_k: int,
    check_correctness: bool,
) -> BenchResult:
    if hidden_size % block_k != 0:
        raise ValueError(
            "The current baseline per_token_group_quant_fp8 path requires "
            f"hidden_size ({hidden_size}) to be divisible by block_k ({block_k})."
        )

    device = torch.device("cuda")
    set_random_seed(0)
    hidden_states = torch.randn(
        (m, hidden_size), dtype=torch.bfloat16, device=device
    )
    router_logits = torch.randn((m, num_experts), dtype=torch.float32, device=device)
    e_score_correction_bias = torch.randn(
        (num_experts,), dtype=torch.float32, device=device
    )

    accuracy: AccuracyResult = (None, None, None, None)
    if check_correctness:
        accuracy = _check_correctness(
            hidden_states,
            router_logits,
            e_score_correction_bias,
            top_k,
            block_k,
        )

    quantiles = [0.5, 0.2, 0.8]
    topk_us = _bench_us(
        lambda: _baseline_topk_sigmoid(
            router_logits,
            e_score_correction_bias,
            top_k,
        ),
        mode,
        quantiles,
    )
    quant_us = _bench_us(
        lambda: _baseline_quant(hidden_states, block_k),
        mode,
        quantiles,
    )
    baseline_us = _bench_us(
        lambda: _baseline_topk_quant(
            hidden_states,
            router_logits,
            e_score_correction_bias,
            top_k,
            block_k,
        ),
        mode,
        quantiles,
    )
    fused_us = _bench_us(
        lambda: _dispatch_topk_quant(
            hidden_states,
            router_logits,
            e_score_correction_bias,
            top_k,
            block_k,
        ),
        mode,
        quantiles,
    )

    speedup = baseline_us / fused_us if fused_us else math.inf
    return BenchResult(
        mode=mode,
        m=m,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        block_k=block_k,
        topk_us=topk_us,
        quant_us=quant_us,
        baseline_us=baseline_us,
        fused_us=fused_us,
        speedup=speedup,
        delta_us=baseline_us - fused_us,
        topk_id_mismatch=accuracy[0],
        topk_weight_max_abs_err=accuracy[1],
        a1q_max_abs_err=accuracy[2],
        a1q_scale_max_abs_err=accuracy[3],
    )


def _format_optional_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2e}"


def _print_header() -> None:
    print(
        "mode".ljust(10),
        "M".rjust(6),
        "H".rjust(6),
        "E".rjust(5),
        "K".rjust(3),
        "blk".rjust(4),
        "topk_us".rjust(10),
        "quant_us".rjust(10),
        "base_us".rjust(10),
        "fused_us".rjust(10),
        "speedup".rjust(9),
        "delta_us".rjust(10),
        "id_mis".rjust(8),
        "w_err".rjust(10),
        "q_err".rjust(10),
        "s_err".rjust(10),
    )
    print("-" * 147)


def _print_result(result: BenchResult) -> None:
    print(
        result.mode.ljust(10),
        f"{result.m:6d}",
        f"{result.hidden_size:6d}",
        f"{result.num_experts:5d}",
        f"{result.top_k:3d}",
        f"{result.block_k:4d}",
        f"{result.topk_us:10.3f}",
        f"{result.quant_us:10.3f}",
        f"{result.baseline_us:10.3f}",
        f"{result.fused_us:10.3f}",
        f"{result.speedup:9.3f}",
        f"{result.delta_us:10.3f}",
        _format_optional_int(result.topk_id_mismatch).rjust(8),
        _format_optional_float(result.topk_weight_max_abs_err).rjust(10),
        _format_optional_float(result.a1q_max_abs_err).rjust(10),
        _format_optional_float(result.a1q_scale_max_abs_err).rjust(10),
    )


def parse_args():
    parser = FlexibleArgumentParser(
        description=(
            "Benchmark MiniMax-M2 fused MoE top-k sigmoid routing + FP8 "
            "per-token group quantization against the unfused baseline."
        )
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        help="Token counts to benchmark. Decode batch size maps directly to M.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[3072],
        help="Hidden sizes to benchmark.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[256],
        help="Number of experts to benchmark.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[8],
        help="Top-k values to benchmark.",
    )
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument(
        "--modes",
        choices=["eager", "cudagraph"],
        nargs="+",
        default=["eager", "cudagraph"],
        help="Benchmark eager launches, CUDA graph replay, or both.",
    )
    parser.add_argument(
        "--check-correctness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare fused outputs with the baseline before timing each shape.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not current_platform.is_cuda():
        raise RuntimeError("CUDA device is required to run this benchmark.")

    args = parse_args()
    _print_header()

    configs = itertools.product(
        args.modes,
        args.m_values,
        args.hidden_sizes,
        args.num_experts,
        args.top_k,
    )
    for mode, m, hidden_size, num_experts, top_k in configs:
        result = _run_one(
            mode=mode,
            m=m,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            block_k=args.block_k,
            check_correctness=args.check_correctness,
        )
        _print_result(result)
