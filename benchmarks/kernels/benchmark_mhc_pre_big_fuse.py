#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TileLang mHC-pre residual staging dtype variants.

This benchmark isolates the post-GEMM fused mHC-pre TileLang backend and
compares the current production bf16 shared-memory staging against the old
float32 shared-memory staging.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.import_utils import has_tilelang

if has_tilelang():
    import tilelang
    import tilelang.language as T
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]

HC_MULT = 4
HC_MULT3 = HC_MULT * (2 + HC_MULT)
DEEPSEEK_V4_HIDDEN_SIZE = 4096
DEEPSEEK_V4_SINKHORN_ITERS = 20


@dataclass(frozen=True)
class ShapeCase:
    num_tokens: int
    hidden_size: int
    n_splits: int


def _parse_shape(text: str) -> ShapeCase:
    parts = text.lower().replace("x", ",").split(",")
    if len(parts) not in (2, 3):
        raise ValueError("shape must be tokens,hidden[,splits]")
    return ShapeCase(
        num_tokens=int(parts[0]),
        hidden_size=int(parts[1]),
        n_splits=int(parts[2]) if len(parts) == 3 else 1,
    )


if tilelang is not None:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def mhc_pre_big_fuse_tilelang_float32_shared(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 16,
        hc_mult: int = 4,
    ):
        """TileLang baseline with the old float32 residual staging buffer."""
        num_tokens = T.dynamic("num_tokens")
        hc_mult3 = hc_mult * (2 + hc_mult)
        hidden_block = math.gcd(512, hidden_size)

        gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
        gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
        hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
        hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
        residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
        post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
        comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
        layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

        with T.Kernel(num_tokens, threads=96) as i:
            T.pdl_sync()
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            mixes_shared = T.alloc_shared(hc_mult3, T.float32)
            T.copy(mixes, mixes_shared)

            if T.get_thread_binding() < 32:
                cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                for j in T.Parallel(hc_mult):
                    post_mix[i, j] = (
                        T.sigmoid(
                            mixes_shared[j + hc_mult] * hc_scale[1]
                            + hc_base[j + hc_mult]
                        )
                        * hc_post_mult_value
                    )
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = (
                        mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                        + hc_base[j * hc_mult + k + hc_mult * 2]
                    )

                row_sum = T.alloc_fragment(hc_mult, T.float32)
                col_sum = T.alloc_fragment(hc_mult, T.float32)
                row_max = T.alloc_fragment(hc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_mix[i, j * hc_mult + k] = cm[j, k]
            else:
                pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
                for j in T.Parallel(hc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )

                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                    xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                    T.copy(residual[i, 0, i0_h * hidden_block], xs)
                    T.copy(xs, xl)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)
                    for i_hc in T.serial(hc_mult):
                        pre = pre_mix_shared[i_hc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_hc, i1_h]

                    T.copy(ol, layer_input[i, i0_h * hidden_block])
            T.pdl_trigger()

else:

    def mhc_pre_big_fuse_tilelang_float32_shared(*args, **kwargs):
        raise RuntimeError("TileLang is required for this benchmark.")


def _event_time_ms(fn: Callable[[], None], repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _summary(times: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _diff_summary(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, dict[str, float]]:
    result = {}
    for name, act, exp in zip(
        ("post_mix", "comb_mix", "layer_input"), actual, expected
    ):
        diff = (act.to(torch.float32) - exp.to(torch.float32)).abs()
        result[name] = {
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
        }
    return result


def _make_fused_inputs(
    case: ShapeCase,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + case.num_tokens + case.hidden_size + case.n_splits)
    gemm_out_mul = torch.randn(
        case.n_splits,
        case.num_tokens,
        HC_MULT3,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    gemm_out_sqrsum = (
        torch.rand(
            case.n_splits,
            case.num_tokens,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * (HC_MULT * case.hidden_size)
        + 1.0
    )
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda", generator=generator)
    hc_base = torch.randn(
        HC_MULT3, dtype=torch.float32, device="cuda", generator=generator
    )
    residual = torch.randn(
        case.num_tokens,
        HC_MULT,
        case.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual


def _alloc_fused_outputs(
    num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(num_tokens, HC_MULT, dtype=torch.float32, device="cuda"),
        torch.empty(num_tokens, HC_MULT * HC_MULT, dtype=torch.float32, device="cuda"),
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"),
    )


def _run_case(case: ShapeCase, args) -> dict[str, Any]:
    from vllm._tilelang_ops import mhc_pre_big_fuse_tilelang

    gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual = _make_fused_inputs(
        case, args.seed
    )
    params = (
        case.hidden_size,
        args.rms_eps,
        args.hc_pre_eps,
        args.hc_sinkhorn_eps,
        args.hc_post_mult_value,
        args.sinkhorn_repeat,
        case.n_splits,
        HC_MULT,
    )
    outputs = {
        variant: _alloc_fused_outputs(case.num_tokens, case.hidden_size)
        for variant in args.variants
    }

    def make_runner(variant: str) -> Callable[[], None]:
        out = outputs[variant]

        def run() -> None:
            if variant == "bf16_shared":
                mhc_pre_big_fuse_tilelang(
                    gemm_out_mul,
                    gemm_out_sqrsum,
                    hc_scale,
                    hc_base,
                    residual,
                    out[0],
                    out[1],
                    out[2],
                    *params,
                )
            else:
                mhc_pre_big_fuse_tilelang_float32_shared(
                    gemm_out_mul,
                    gemm_out_sqrsum,
                    hc_scale,
                    hc_base,
                    residual,
                    out[0],
                    out[1],
                    out[2],
                    *params,
                )

        return run

    timing = {}
    for variant in args.variants:
        runner = make_runner(variant)
        for _ in range(args.warmup):
            runner()
        torch.cuda.synchronize()
        timing[variant] = _summary(_event_time_ms(runner, args.repeats))

    baseline_variant = "bf16_shared" if "bf16_shared" in outputs else args.variants[0]
    diff = {
        variant: _diff_summary(outputs[variant], outputs[baseline_variant])
        for variant in args.variants
    }
    return {
        "shape": case.__dict__,
        "timing": timing,
        "diff_vs": baseline_variant,
        "diff": diff,
    }


def _print_result(result: dict[str, Any], variants: list[str]) -> None:
    shape = result["shape"]
    timing = " ".join(
        f"{variant}={result['timing'][variant]['median_ms']:.4f}ms"
        for variant in variants
    )
    print(
        f"tokens={shape['num_tokens']:<6} hidden={shape['hidden_size']:<5} "
        f"splits={shape['n_splits']:<2} | {timing}"
    )
    for variant in variants:
        diff = result["diff"][variant]["layer_input"]["max_abs"]
        print(
            f"      diff[{variant} vs {result['diff_vs']}].layer_input.max={diff:.3e}"
        )


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["bf16_shared", "float32_shared"],
        default=["bf16_shared", "float32_shared"],
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=_parse_shape,
        default=[
            ShapeCase(1, DEEPSEEK_V4_HIDDEN_SIZE, 64),
            ShapeCase(2, DEEPSEEK_V4_HIDDEN_SIZE, 64),
            ShapeCase(4, DEEPSEEK_V4_HIDDEN_SIZE, 64),
            ShapeCase(8, DEEPSEEK_V4_HIDDEN_SIZE, 64),
            ShapeCase(16, DEEPSEEK_V4_HIDDEN_SIZE, 64),
            ShapeCase(32, DEEPSEEK_V4_HIDDEN_SIZE, 64),
            ShapeCase(64, DEEPSEEK_V4_HIDDEN_SIZE, 64),
            ShapeCase(128, DEEPSEEK_V4_HIDDEN_SIZE, 64),
            ShapeCase(256, DEEPSEEK_V4_HIDDEN_SIZE, 37),
            ShapeCase(512, DEEPSEEK_V4_HIDDEN_SIZE, 18),
            ShapeCase(1024, DEEPSEEK_V4_HIDDEN_SIZE, 9),
            ShapeCase(2048, DEEPSEEK_V4_HIDDEN_SIZE, 4),
            ShapeCase(4096, DEEPSEEK_V4_HIDDEN_SIZE, 2),
            ShapeCase(8192, DEEPSEEK_V4_HIDDEN_SIZE, 1),
            ShapeCase(16384, DEEPSEEK_V4_HIDDEN_SIZE, 1),
            ShapeCase(32768, DEEPSEEK_V4_HIDDEN_SIZE, 1),
            ShapeCase(65536, DEEPSEEK_V4_HIDDEN_SIZE, 1),
            ShapeCase(131072, DEEPSEEK_V4_HIDDEN_SIZE, 1),
        ],
        help="Each shape is tokens,hidden[,splits]. DeepSeek-V4 uses hidden=4096.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--hc-pre-eps", type=float, default=1e-6)
    parser.add_argument("--hc-sinkhorn-eps", type=float, default=1e-6)
    parser.add_argument("--hc-post-mult-value", type=float, default=2.0)
    parser.add_argument(
        "--sinkhorn-repeat", type=int, default=DEEPSEEK_V4_SINKHORN_ITERS
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if not current_platform.is_cuda():
        raise RuntimeError("CUDA is required to run this benchmark.")
    if not has_tilelang():
        raise RuntimeError("TileLang is required to run this benchmark.")

    results = []
    for case in args.shapes:
        result = _run_case(case, args)
        results.append(result)
        _print_result(result, args.variants)

    if args.output_json is not None:
        payload = {
            "config": {
                "variants": args.variants,
                "warmup": args.warmup,
                "repeats": args.repeats,
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
