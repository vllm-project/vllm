# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the ROCm fused MXFP8 MoE against BF16 emulation.

The ``tp`` and ``ep`` profiles match MiniMax-M3 on an eight-GPU node:

* TP8: 128 local experts, hidden size 6144, intermediate shard 384.
* TP8+EP8: 16 local experts, hidden size 6144, intermediate size 3072.

The BF16 provider consumes weights dequantized from the same MXFP8 bytes, so
the comparison isolates the runtime backend rather than checkpoint rounding.
"""

import argparse
import json
from dataclasses import dataclass
from functools import partial

import torch

from vllm import _custom_ops as ops
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
    _grouped_gemm_mxfp8,
    fused_moe_mxfp8_native,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
    normalize_mxfp8_e4m3fn_to_e4m3fnuz,
)
from vllm.models.minimax_m3.amd.ops import swiglu_oai_quantize_mxfp8
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.worker.workspace import init_workspace_manager


@dataclass(frozen=True)
class Profile:
    local_experts: int
    global_experts: int
    hidden_size: int
    intermediate_size: int


PROFILES = {
    "smoke": Profile(8, 8, 256, 512),
    "tp": Profile(128, 128, 6144, 384),
    "ep": Profile(16, 128, 6144, 3072),
}


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f = actual.float()
    expected_f = expected.float()
    return ((actual_f - expected_f).norm() / (expected_f.norm() + 1e-8)).item()


def _make_routing(
    tokens: int,
    profile: Profile,
    top_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    logits = torch.randn(tokens, profile.global_experts, device=device)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    expert_map = None
    if profile.local_experts != profile.global_experts:
        expert_map = torch.full(
            (profile.global_experts,), -1, dtype=torch.int32, device=device
        )
        expert_map[: profile.local_experts] = torch.arange(
            profile.local_experts, dtype=torch.int32, device=device
        )
        # Keep random global routing representative of EP. Pin one assignment
        # so even the one-token benchmark exercises a local expert.
        topk_ids[0, 0] = 0
    return topk_weights.float(), topk_ids.int(), expert_map


def _bench(fn, warmup: int, rep: int) -> tuple[float, float, float]:
    median, low, high = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        quantiles=[0.5, 0.2, 0.8],
    )
    return float(median), float(low), float(high)


def _reduce_moe_output(
    routed_output: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    ops.moe_sum(routed_output, output)
    return output


@torch.inference_mode()
@default_vllm_config()
def run(args: argparse.Namespace) -> None:
    assert current_platform.is_rocm(), "This benchmark requires ROCm."
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert "gfx94" in arch or "gfx95" in arch, (
        f"ROCm fused MXFP8 requires gfx94x/gfx95x, got {arch}."
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    init_workspace_manager(device)
    profile = PROFILES[args.profile]
    E = profile.local_experts
    H = profile.hidden_size
    inter = profile.intermediate_size

    w13_source = torch.randn(E, 2 * inter, H, device=device, dtype=torch.bfloat16)
    w2_source = torch.randn(E, H, inter, device=device, dtype=torch.bfloat16)
    w13_source.mul_(args.weight_scale)
    w2_source.mul_(args.weight_scale)
    w13, w13_scale = _mxfp8_e4m3_quantize_torch(w13_source, is_sf_swizzled_layout=False)
    w2, w2_scale = _mxfp8_e4m3_quantize_torch(w2_source, is_sf_swizzled_layout=False)
    del w13_source, w2_source
    w13_bf16 = dequant_mxfp8_to_bf16(w13, w13_scale)
    w2_bf16 = dequant_mxfp8_to_bf16(w2, w2_scale)
    if current_platform.is_fp8_fnuz():
        w13, w13_scale = normalize_mxfp8_e4m3fn_to_e4m3fnuz(w13, w13_scale)
        w2, w2_scale = normalize_mxfp8_e4m3fn_to_e4m3fnuz(w2, w2_scale)

    bf16_config = biased_moe_quant_config(
        None,
        None,
        gemm1_alpha=args.alpha,
        gemm1_beta=args.beta,
        gemm1_clamp_limit=args.limit,
    )
    moe_config = FusedMoEConfig(
        num_experts=profile.global_experts,
        experts_per_token=args.top_k,
        hidden_dim=H,
        intermediate_size=inter,
        num_local_experts=E,
        num_logical_experts=profile.global_experts,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        device=device,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        max_num_tokens=max(args.tokens),
        swiglu_limit=args.limit,
        swiglu_alpha=args.alpha,
        swiglu_beta=args.beta,
    )
    bf16_kernel = FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=bf16_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        TritonExperts(moe_config, bf16_config),
    )

    for tokens in args.tokens:
        hidden_states = torch.randn(
            tokens, H, device=device, dtype=torch.bfloat16
        ).mul_(args.activation_scale)
        topk_weights, topk_ids, expert_map = _make_routing(
            tokens, profile, args.top_k, device
        )
        mxfp8_output = torch.empty_like(hidden_states)

        run_mxfp8 = partial(
            fused_moe_mxfp8_native,
            hidden_states,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            alpha=args.alpha,
            beta=args.beta,
            limit=args.limit,
            global_num_experts=profile.global_experts,
            expert_map=expert_map,
            output=mxfp8_output,
        )
        run_bf16 = partial(
            bf16_kernel.apply,
            hidden_states=hidden_states,
            w1=w13_bf16,
            w2=w2_bf16,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            global_num_experts=profile.global_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=False,
        )

        actual = run_mxfp8()
        expected = run_bf16()
        torch.accelerator.synchronize()
        relative_error = _relative_error(actual, expected)
        if relative_error >= args.max_relative_error:
            raise AssertionError(
                f"tokens={tokens}: relative error {relative_error:.6f} exceeds "
                f"{args.max_relative_error:.6f}"
            )

        if args.breakdown:
            M = tokens * args.top_k
            tokens_per_expert = max(1, M // profile.global_experts)
            block_m = max(16, min(1 << (tokens_per_expert - 1).bit_length(), 64))
            run_align = partial(
                moe_align_block_size,
                topk_ids,
                block_m,
                profile.global_experts,
                expert_map,
                ignore_invalid_experts=expert_map is not None,
            )
            sorted_ids, expert_ids, num_post = run_align()
            a_q, a_s = mxfp8_e4m3_quantize(hidden_states)
            run_gemm1 = partial(
                _grouped_gemm_mxfp8,
                a_q,
                a_s,
                w13,
                w13_scale,
                sorted_ids,
                expert_ids,
                num_post,
                M,
                args.top_k,
                block_m,
                hidden_states.dtype,
                args.top_k,
                expert_map=expert_map,
            )
            g1 = run_gemm1()
            run_activation = partial(
                swiglu_oai_quantize_mxfp8,
                g1,
                alpha=args.alpha,
                beta=args.beta,
                limit=args.limit,
            )
            act_q, act_s = run_activation()
            run_gemm2 = partial(
                _grouped_gemm_mxfp8,
                act_q,
                act_s,
                w2,
                w2_scale,
                sorted_ids,
                expert_ids,
                num_post,
                M,
                args.top_k,
                block_m,
                hidden_states.dtype,
                1,
                mul_weight_by=topk_weights.reshape(-1),
                expert_map=expert_map,
            )
            g2 = run_gemm2()
            run_quant1 = partial(mxfp8_e4m3_quantize, hidden_states)
            run_reduce = partial(
                _reduce_moe_output,
                g2.view(tokens, args.top_k, H),
                mxfp8_output,
            )
            stages = {
                "align_ms": _bench(run_align, args.warmup, args.rep)[0],
                "quant1_ms": _bench(run_quant1, args.warmup, args.rep)[0],
                "gemm1_ms": _bench(run_gemm1, args.warmup, args.rep)[0],
                "activation_quant_ms": _bench(run_activation, args.warmup, args.rep)[0],
                "gemm2_ms": _bench(run_gemm2, args.warmup, args.rep)[0],
                "reduce_ms": _bench(run_reduce, args.warmup, args.rep)[0],
            }
            print(
                json.dumps(
                    {
                        "kind": "breakdown",
                        "profile": args.profile,
                        "tokens": tokens,
                        "block_m": block_m,
                        **stages,
                        "stage_sum_ms": sum(stages.values()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        mxfp8_ms, mxfp8_low, mxfp8_high = _bench(run_mxfp8, args.warmup, args.rep)
        bf16_ms, bf16_low, bf16_high = _bench(run_bf16, args.warmup, args.rep)
        record = {
            "profile": args.profile,
            "arch": arch,
            "tokens": tokens,
            "local_experts": E,
            "global_experts": profile.global_experts,
            "hidden_size": H,
            "intermediate_size": inter,
            "top_k": args.top_k,
            "relative_error": relative_error,
            "mxfp8_ms": mxfp8_ms,
            "mxfp8_p20_ms": mxfp8_low,
            "mxfp8_p80_ms": mxfp8_high,
            "bf16_ms": bf16_ms,
            "bf16_p20_ms": bf16_low,
            "bf16_p80_ms": bf16_high,
            "speedup": bf16_ms / mxfp8_ms,
        }
        print(json.dumps(record, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.702)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--limit", type=float, default=7.0)
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument("--activation-scale", type=float, default=0.5)
    parser.add_argument("--max-relative-error", type=float, default=0.05)
    parser.add_argument("--breakdown", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
