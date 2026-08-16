# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from collections.abc import Callable

import torch

from vllm.models.kimi_k3.nvidia.ops.kda_mixed import (
    pack_kda_mixed_inputs,
    scatter_rms_norm_kda_mixed_outputs,
)
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.triton_utils import triton


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Kimi-K3 mixed KDA boundary fusion."
    )
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--spec-fraction", type=float, default=0.5)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    return parser.parse_args()


def _bench(
    fn: Callable[[], object],
    warmup_ms: int,
    rep_ms: int,
) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
        )
    )


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if args.num_tokens < 1:
        raise ValueError("--num-tokens must be positive")
    if not 0.0 < args.spec_fraction < 1.0:
        raise ValueError("--spec-fraction must be strictly between 0 and 1")

    dtype = getattr(torch, args.dtype)
    T, H, D = args.num_tokens, args.num_heads, 128
    num_spec = round(T * args.spec_fraction)
    num_non_spec = T - num_spec
    if not num_spec or not num_non_spec:
        raise ValueError("The rounded benchmark partition must contain both groups")
    torch.manual_seed(17)

    is_spec = torch.zeros(T, dtype=torch.bool)
    if num_spec:
        is_spec[torch.randperm(T)[:num_spec]] = True
    indices = torch.argsort(is_spec, stable=True).to(device="cuda")
    non_spec_indices = indices[:num_non_spec]
    spec_indices = indices[num_non_spec:]

    qkv_width = 3 * H * D
    mixed_qkv = torch.randn(
        T,
        qkv_width + 7,
        dtype=dtype,
        device="cuda",
    )[:, :qkv_width]
    g1 = torch.randn(
        1,
        T,
        H,
        D + 3,
        dtype=dtype,
        device="cuda",
    )[..., :D]
    beta = torch.randn(
        1,
        T,
        H + 3,
        dtype=dtype,
        device="cuda",
    )[..., :H]

    non_spec_output = torch.randn(
        1,
        num_non_spec,
        H,
        D,
        dtype=dtype,
        device="cuda",
    )
    spec_output = torch.randn(
        1,
        num_spec,
        H,
        D,
        dtype=dtype,
        device="cuda",
    )
    gate = torch.randn(T, H, D, dtype=dtype, device="cuda")
    norm = FusedRMSNormGated(
        D,
        activation="sigmoid",
        device=torch.device("cuda"),
        dtype=dtype,
    )
    torch.nn.init.normal_(norm.weight)
    baseline_output = torch.empty(1, T, H, D, dtype=dtype, device="cuda")
    fused_output = torch.empty_like(baseline_output)

    def before_pack() -> tuple[torch.Tensor, ...]:
        return (
            mixed_qkv.index_select(0, spec_indices),
            g1.index_select(1, spec_indices),
            beta.index_select(1, spec_indices),
            mixed_qkv.index_select(0, non_spec_indices),
            g1.index_select(1, non_spec_indices),
            beta.index_select(1, non_spec_indices),
        )

    def after_pack() -> tuple[torch.Tensor, ...]:
        return pack_kda_mixed_inputs(
            mixed_qkv,
            g1,
            beta,
            non_spec_indices,
            spec_indices,
        )

    def before_scatter_norm() -> None:
        baseline_output.index_copy_(1, spec_indices, spec_output)
        baseline_output.index_copy_(1, non_spec_indices, non_spec_output)
        baseline_output.copy_(norm.forward_cuda(baseline_output, gate))

    def after_scatter_norm() -> None:
        scatter_rms_norm_kda_mixed_outputs(
            non_spec_output,
            spec_output,
            non_spec_indices,
            spec_indices,
            gate,
            norm.weight,
            fused_output,
            norm.eps,
        )

    def before_combined() -> None:
        before_pack()
        before_scatter_norm()

    def after_combined() -> None:
        after_pack()
        after_scatter_norm()

    before_combined()
    after_combined()
    torch.accelerator.synchronize()

    rows = [
        (
            "input pack",
            _bench(before_pack, args.warmup_ms, args.rep_ms),
            _bench(after_pack, args.warmup_ms, args.rep_ms),
            6,
            1,
        ),
        (
            "scatter + gated RMSNorm",
            _bench(before_scatter_norm, args.warmup_ms, args.rep_ms),
            _bench(after_scatter_norm, args.warmup_ms, args.rep_ms),
            3,
            1,
        ),
        (
            "combined boundary",
            _bench(before_combined, args.warmup_ms, args.rep_ms),
            _bench(after_combined, args.warmup_ms, args.rep_ms),
            9,
            2,
        ),
    ]

    print(
        f"Kimi-K3 mixed KDA: T={T}, H={H}, D={D}, "
        f"non-spec={num_non_spec}, spec={num_spec}, dtype={args.dtype}"
    )
    print("| stage | before (ms) | after (ms) | speedup | launches |")
    print("|---|---:|---:|---:|---:|")
    for stage, before_ms, after_ms, before_launches, after_launches in rows:
        print(
            f"| {stage} | {before_ms:.4f} | {after_ms:.4f} | "
            f"{before_ms / after_ms:.2f}x | "
            f"{before_launches} -> {after_launches} |"
        )


if __name__ == "__main__":
    main()
