# SPDX-License-Identifier: Apache-2.0
"""Check explicit tl.fma accuracy for fused DeepSeek V4 indexer-Q RoPE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fma_variants import HEAD_DIM, MAX_POS, N_HEAD, ROPE_DIM, run_variant
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def make_inputs(
    num_tokens: int,
    cache_dtype: torch.dtype,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(
        num_tokens,
        N_HEAD,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.randint(
        0,
        MAX_POS,
        (num_tokens,),
        dtype=torch.int64,
        device=device,
    )
    cos_sin_cache = torch.randn(
        MAX_POS,
        ROPE_DIM,
        dtype=cache_dtype,
        device=device,
    )
    weights = torch.randn(
        num_tokens,
        N_HEAD,
        dtype=torch.bfloat16,
        device=device,
    )
    return positions, q, cos_sin_cache, weights


def reference(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = q.clone()
    ops.rotary_embedding(
        positions,
        q_rot,
        None,
        HEAD_DIM,
        cos_sin_cache,
        False,
        HEAD_DIM - ROPE_DIM,
        False,
    )
    q_fp8, q_scale = per_token_group_quant_fp8(
        q_rot.view(-1, HEAD_DIM).contiguous(),
        HEAD_DIM,
        use_ue8m0=True,
    )
    q_fp8 = q_fp8.view(-1, N_HEAD, HEAD_DIM)
    q_scale = q_scale.view(-1, N_HEAD)
    weights_out = weights.to(torch.float32) * q_scale * softmax_scale * head_scale
    return q_fp8, weights_out


def compare_case(
    num_tokens: int,
    cache_dtype_name: str,
    device: str,
    seed: int,
) -> list[dict[str, object]]:
    cache_dtype = _dtype(cache_dtype_name)
    positions, q, cos_sin_cache, weights = make_inputs(
        num_tokens,
        cache_dtype,
        device,
        seed,
    )
    softmax_scale = HEAD_DIM**-0.5
    head_scale = N_HEAD**-0.5
    q_ref, weights_ref = reference(
        positions,
        q,
        cos_sin_cache,
        weights,
        softmax_scale,
        head_scale,
    )

    rows: list[dict[str, object]] = []
    for label, use_fma in (("muladd", False), ("fma", True)):
        q_actual, weights_actual = run_variant(
            positions,
            q,
            cos_sin_cache,
            weights,
            softmax_scale,
            head_scale,
            use_fma=use_fma,
        )
        torch.cuda.synchronize()
        ref_bits = q_ref.view(torch.int8)
        actual_bits = q_actual.view(torch.int8)
        q_mismatches = int((ref_bits != actual_bits).sum().item())
        weight_diff = (weights_ref - weights_actual).abs()
        rows.append(
            {
                "num_tokens": num_tokens,
                "cache_dtype": cache_dtype_name,
                "variant": label,
                "q_mismatches": q_mismatches,
                "q_elements": ref_bits.numel(),
                "weights_equal": bool(torch.equal(weights_ref, weights_actual)),
                "max_weight_abs_diff": float(weight_diff.max().item()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 7, 32, 257, 1023],
    )
    parser.add_argument(
        "--cache-dtypes",
        nargs="+",
        default=["float32", "bfloat16"],
        choices=["float32", "bfloat16"],
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--no-strict-fma",
        action="store_true",
        help="Do not return non-zero if the fma variant mismatches reference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    for num_tokens in args.tokens:
        for cache_dtype_name in args.cache_dtypes:
            all_rows.extend(
                compare_case(num_tokens, cache_dtype_name, args.device, args.seed)
            )

    print(
        "tokens cache_dtype variant q_mismatches/q_elements "
        "weights_equal max_weight_abs_diff"
    )
    for row in all_rows:
        print(
            f"{row['num_tokens']:>6} {row['cache_dtype']:<8} "
            f"{row['variant']:<6} {row['q_mismatches']}/{row['q_elements']} "
            f"{row['weights_equal']} {row['max_weight_abs_diff']:.9g}"
        )

    if args.json is not None:
        args.json.write_text(json.dumps(all_rows, indent=2) + "\n")

    if not args.no_strict_fma:
        for row in all_rows:
            if (
                row["variant"] == "fma"
                and (row["q_mismatches"] != 0 or not row["weights_equal"])
            ):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
