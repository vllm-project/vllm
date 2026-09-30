# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import math
import statistics
from pathlib import Path

import torch

import vllm._aiter_ops  # noqa: F401
import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform


def _median_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return statistics.median(times_ms) * 1000.0


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def _slice_scale_tokens(
    x_scale: torch.Tensor, tokens: int, payload_rows: int
) -> torch.Tensor:
    if x_scale.dim() != 2:
        raise ValueError(f"expected rank-2 x_scale, got {x_scale.shape}")
    if int(x_scale.shape[0]) == payload_rows:
        return x_scale[:tokens].contiguous()
    if int(x_scale.shape[1]) == payload_rows:
        return x_scale[:, :tokens].contiguous()
    raise ValueError(
        f"unable to determine token dimension for x_scale shape={x_scale.shape}, "
        f"payload_rows={payload_rows}"
    )


def _repeat_scale_tokens(
    x_scale: torch.Tensor, tokens: int, payload_rows: int, repeats: int
) -> torch.Tensor:
    if x_scale.dim() != 2:
        raise ValueError(f"expected rank-2 x_scale, got {x_scale.shape}")
    if int(x_scale.shape[0]) == payload_rows:
        return x_scale.repeat((repeats, 1))[:tokens].contiguous()
    if int(x_scale.shape[1]) == payload_rows:
        return x_scale.repeat((1, repeats))[:, :tokens].contiguous()
    raise ValueError(
        f"unable to determine token dimension for x_scale shape={x_scale.shape}, "
        f"payload_rows={payload_rows}"
    )


def _slice_or_repeat_token_rows(
    x_fp8: torch.Tensor, x_scale: torch.Tensor, tokens: int, payload_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens <= payload_rows:
        return x_fp8[:tokens].contiguous(), _slice_scale_tokens(
            x_scale, tokens, payload_rows
        )

    repeats = math.ceil(tokens / payload_rows)
    x_fp8 = x_fp8.repeat((repeats, 1))[:tokens].contiguous()
    x_scale = _repeat_scale_tokens(x_scale, tokens, payload_rows, repeats)
    return x_fp8, x_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--candidate",
        choices=("blockscale_skinny",),
        default="blockscale_skinny",
    )
    args = parser.parse_args()

    if not current_platform.is_rocm():
        raise SystemExit("This benchmark is ROCm-specific.")

    payload = torch.load(Path(args.payload), map_location="cpu")
    weight = payload["weight"].cuda()
    weight_scale = payload["weight_scale"].cuda()
    payload_rows = int(
        payload["x_fp8"].shape[0] if "x_fp8" in payload else payload["x_bf16"].shape[0]
    )
    if "x_fp8" in payload and "x_scale" in payload:
        x_fp8 = payload["x_fp8"].cuda()
        x_scale = payload["x_scale"].cuda()
    else:
        x_bf16 = payload["x_bf16"].cuda()
        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x_bf16,
                                                        transpose_scale=True)

    if args.m is not None:
        if args.m <= 0 or args.m > 8:
            raise SystemExit(f"--m must be in [1, 8], got {args.m}")
        x_fp8, x_scale = _slice_or_repeat_token_rows(
            x_fp8, x_scale, args.m, payload_rows
        )

    out_ref = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
        x_fp8, weight, x_scale, weight_scale, output_dtype=torch.bfloat16
    )
    out_candidate = torch.empty_like(out_ref)

    cu_count = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count

    def run_baseline():
        return rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_fp8, weight, x_scale, weight_scale, output_dtype=torch.bfloat16
        )

    def run_candidate():
        ops.wvSplitKQBlockScale(
            weight,
            x_fp8,
            x_scale,
            weight_scale,
            out_candidate,
            cu_count,
            True,
        )
        return out_candidate

    baseline_us = _median_us(run_baseline, args.warmup, args.iters)
    candidate_us = _median_us(run_candidate, args.warmup, args.iters)
    out_final = run_candidate()
    correct = bool(torch.isfinite(out_final).all().item()) and torch.allclose(
        out_ref.float(), out_final.float(), atol=1e-2, rtol=1e-2
    )

    print(f"payload={args.payload}")
    print(f"M={x_fp8.shape[0]}")
    print(f"baseline_us={baseline_us:.2f}")
    print(f"candidate_us={candidate_us:.2f}")
    print(f"delta_us={baseline_us - candidate_us:.2f}")
    print(f"speedup={baseline_us / candidate_us:.4f}")
    print(f"correct={correct}")
    print(f"max_abs_diff={_max_abs_diff(out_ref, out_final):.6g}")
    print(f"mean_abs_diff={_mean_abs_diff(out_ref, out_final):.6g}")


if __name__ == "__main__":
    main()
