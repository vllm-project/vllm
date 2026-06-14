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
from vllm.models.minimax_m3.amd.ops import (
    swiglu_oai_quantize_mxfp8,
    swiglu_oai_split,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import init_workspace_manager


@dataclass(frozen=True)
class Profile:
    local_experts: int
    global_experts: int
    hidden_size: int
    intermediate_size: int
    shared_intermediate_size: int


PROFILES = {
    "smoke": Profile(8, 8, 256, 512, 512),
    "tp": Profile(128, 128, 6144, 384, 384),
    "ep": Profile(16, 128, 6144, 3072, 384),
}


def _accuracy_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    error = actual_f - expected_f
    return {
        "relative_error": (error.norm() / (expected_f.norm() + 1e-8)).item(),
        "mean_abs_error": error.abs().mean().item(),
        "max_abs_error": error.abs().max().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            actual_f.flatten(),
            expected_f.flatten(),
            dim=0,
        ).item(),
    }


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


def _overlap_with_shared_expert(
    routed_fn,
    shared_fn,
    shared_stream: torch.cuda.Stream,
) -> torch.Tensor:
    current = torch.cuda.current_stream()
    shared_stream.wait_stream(current)
    routed_output = routed_fn()
    with torch.cuda.stream(shared_stream):
        shared_output = shared_fn()
    current.wait_stream(shared_stream)
    return routed_output + shared_output


@triton.jit
def _mxfp8_grouped_gemm_w8a16_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    A_DIV: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
    SCALE_BITCAST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Experimental Marlin-style W8A16 grouped GEMM for gfx94x.

    MXFP8 weights remain compressed in HBM. Each weight tile is expanded and
    scaled to BF16 in registers immediately before the BF16 matrix multiply.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_post = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_post:
        return

    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_tid).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_row = offs_token // A_DIV
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + off_e * stride_be
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )
    bs_ptrs = (
        b_scale_ptr
        + off_e * stride_bse
        + offs_n[:, None] * stride_bsn
        + (offs_k[None, :] // 32) * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        bq = tl.load(b_ptrs, mask=n_mask[:, None], other=0.0)
        bsc = tl.load(bs_ptrs, mask=n_mask[:, None], other=0)
        if SCALE_BITCAST:
            # E8M0 is exactly a biased IEEE exponent. BF16 has the same
            # eight-bit exponent, so shifting into its exponent field avoids
            # a transcendental exp2 instruction.
            scale_bits = bsc.to(tl.uint16) << 7
            scale = scale_bits.to(tl.bfloat16, bitcast=True)
            b = (bq.to(tl.bfloat16) * scale).to(tl.bfloat16)
        else:
            scale = tl.exp2(bsc.to(tl.float32) - 127.0)
            b = (bq.to(tl.float32) * scale).to(tl.bfloat16)
        acc += tl.dot(a, b.T)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        bs_ptrs += (BLOCK_K // 32) * stride_bsk

    if MUL_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )


def _grouped_gemm_mxfp8_w8a16(
    a: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    block_m: int,
    a_div: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    scale_bitcast: bool,
    mul_weight_by: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    _, N, K = w.shape
    assert a.dtype == torch.bfloat16
    assert w.dtype == torch.float8_e4m3fnuz
    assert K % block_k == 0 and block_k % 32 == 0
    max_post_padded = min(sorted_token_ids.shape[0], num_valid_tokens * block_m)
    m_blocks = triton.cdiv(max_post_padded, block_m)
    n_blocks = triton.cdiv(N, block_n)
    alloc = torch.zeros if expert_map is not None else torch.empty
    out = alloc((num_valid_tokens, N), dtype=a.dtype, device=a.device)
    _mxfp8_grouped_gemm_w8a16_kernel[(m_blocks, n_blocks)](
        a,
        w,
        w_scale,
        out,
        mul_weight_by if mul_weight_by is not None else a,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        out.stride(0),
        out.stride(1),
        A_DIV=a_div,
        MUL_WEIGHT=mul_weight_by is not None,
        SCALE_BITCAST=scale_bitcast,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def _fused_moe_mxfp8_w8a16(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: float | None,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    output: torch.Tensor,
    block_m_override: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    scale_bitcast: bool,
) -> torch.Tensor:
    T, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    M = T * top_k
    tokens_per_expert = max(1, M // global_num_experts)
    block_m = (
        block_m_override
        if block_m_override > 0
        else max(16, min(1 << (tokens_per_expert - 1).bit_length(), 64))
    )
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids,
        block_m,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=expert_map is not None,
    )
    g1 = _grouped_gemm_mxfp8_w8a16(
        hidden_states,
        w13,
        w13_scale,
        sorted_ids,
        expert_ids,
        num_post,
        M,
        block_m,
        top_k,
        block_n,
        block_k,
        num_warps,
        scale_bitcast,
        expert_map=expert_map,
    )
    act = swiglu_oai_split(
        g1,
        alpha=alpha,
        beta=beta,
        limit=limit,
        out_dtype=hidden_states.dtype,
    )
    g2 = _grouped_gemm_mxfp8_w8a16(
        act,
        w2,
        w2_scale,
        sorted_ids,
        expert_ids,
        num_post,
        M,
        block_m,
        1,
        block_n,
        block_k,
        num_warps,
        scale_bitcast,
        mul_weight_by=topk_weights.reshape(-1),
        expert_map=expert_map,
    )
    ops.moe_sum(g2.view(T, top_k, H), output)
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
    shared_stream = torch.cuda.Stream() if args.shared_expert_overlap else None
    if args.shared_expert_overlap:
        shared_inter = profile.shared_intermediate_size
        shared_w13 = torch.randn(
            2 * shared_inter,
            H,
            device=device,
            dtype=torch.bfloat16,
        ).mul_(args.weight_scale)
        shared_w2 = torch.randn(
            H,
            shared_inter,
            device=device,
            dtype=torch.bfloat16,
        ).mul_(args.weight_scale)

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
        if args.shared_expert_overlap:
            assert shared_stream is not None

            def run_shared(
                hidden_states=hidden_states,
                shared_w13=shared_w13,
                shared_w2=shared_w2,
            ):
                gate_up = torch.nn.functional.linear(hidden_states, shared_w13)
                act = swiglu_oai_split(
                    gate_up,
                    alpha=args.alpha,
                    beta=args.beta,
                    limit=args.limit,
                    out_dtype=hidden_states.dtype,
                )
                return torch.nn.functional.linear(act, shared_w2)

            run_mxfp8_timed = partial(
                _overlap_with_shared_expert,
                run_mxfp8,
                run_shared,
                shared_stream,
            )
            run_bf16_timed = partial(
                _overlap_with_shared_expert,
                run_bf16,
                run_shared,
                shared_stream,
            )
        else:
            run_mxfp8_timed = run_mxfp8
            run_bf16_timed = run_bf16
        routed_tokens = tokens * args.top_k
        tokens_per_expert = max(1, routed_tokens // profile.global_experts)
        w8a16_block_m = (
            args.w8a16_block_m
            if args.w8a16_block_m > 0
            else max(16, min(1 << (tokens_per_expert - 1).bit_length(), 64))
        )
        run_w8a16 = partial(
            _fused_moe_mxfp8_w8a16,
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
            output=torch.empty_like(hidden_states),
            block_m_override=args.w8a16_block_m,
            block_n=args.w8a16_block_n,
            block_k=args.w8a16_block_k,
            num_warps=args.w8a16_num_warps,
            scale_bitcast=args.w8a16_scale_mode == "bitcast",
        )
        if args.shared_expert_overlap:
            assert shared_stream is not None
            run_w8a16_timed = partial(
                _overlap_with_shared_expert,
                run_w8a16,
                run_shared,
                shared_stream,
            )
        else:
            run_w8a16_timed = run_w8a16

        actual = run_mxfp8()
        expected = run_bf16()
        w8a16_actual = run_w8a16() if args.w8a16 else None
        torch.accelerator.synchronize()
        native_accuracy = _accuracy_metrics(actual, expected)
        relative_error = native_accuracy["relative_error"]
        if relative_error >= args.max_relative_error:
            raise AssertionError(
                f"tokens={tokens}: relative error {relative_error:.6f} exceeds "
                f"{args.max_relative_error:.6f}"
            )
        w8a16_accuracy = (
            _accuracy_metrics(w8a16_actual, expected)
            if w8a16_actual is not None
            else None
        )
        w8a16_relative_error = (
            w8a16_accuracy["relative_error"] if w8a16_accuracy is not None else None
        )
        if (
            w8a16_relative_error is not None
            and w8a16_relative_error >= args.w8a16_max_relative_error
        ):
            raise AssertionError(
                f"tokens={tokens}: W8A16 relative error "
                f"{w8a16_relative_error:.6f} exceeds "
                f"{args.w8a16_max_relative_error:.6f}"
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

        mxfp8_ms, mxfp8_low, mxfp8_high = _bench(run_mxfp8_timed, args.warmup, args.rep)
        bf16_ms, bf16_low, bf16_high = _bench(run_bf16_timed, args.warmup, args.rep)
        w8a16_result = (
            _bench(run_w8a16_timed, args.warmup, args.rep) if args.w8a16 else None
        )
        record = {
            "profile": args.profile,
            "arch": arch,
            "tokens": tokens,
            "local_experts": E,
            "global_experts": profile.global_experts,
            "hidden_size": H,
            "intermediate_size": inter,
            "top_k": args.top_k,
            "seed": args.seed,
            "shared_expert_overlap": args.shared_expert_overlap,
            **native_accuracy,
            "mxfp8_ms": mxfp8_ms,
            "mxfp8_p20_ms": mxfp8_low,
            "mxfp8_p80_ms": mxfp8_high,
            "bf16_ms": bf16_ms,
            "bf16_p20_ms": bf16_low,
            "bf16_p80_ms": bf16_high,
            "speedup": bf16_ms / mxfp8_ms,
        }
        if w8a16_result is not None:
            w8a16_ms, w8a16_low, w8a16_high = w8a16_result
            assert w8a16_accuracy is not None
            record.update(
                {
                    **{
                        f"w8a16_{name}": value for name, value in w8a16_accuracy.items()
                    },
                    "w8a16_ms": w8a16_ms,
                    "w8a16_p20_ms": w8a16_low,
                    "w8a16_p80_ms": w8a16_high,
                    "w8a16_vs_native": mxfp8_ms / w8a16_ms,
                    "w8a16_vs_bf16": bf16_ms / w8a16_ms,
                    "w8a16_scale_mode": args.w8a16_scale_mode,
                    "w8a16_block_m": w8a16_block_m,
                    "w8a16_block_n": args.w8a16_block_n,
                    "w8a16_block_k": args.w8a16_block_k,
                    "w8a16_num_warps": args.w8a16_num_warps,
                }
            )
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
    parser.add_argument("--w8a16-max-relative-error", type=float, default=0.01)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--shared-expert-overlap", action="store_true")
    parser.add_argument("--w8a16", action="store_true")
    parser.add_argument(
        "--w8a16-block-m",
        type=int,
        default=0,
        help="Override the routed-token M tile; 0 uses the native heuristic.",
    )
    parser.add_argument("--w8a16-block-n", type=int, default=32)
    parser.add_argument("--w8a16-block-k", type=int, default=64)
    parser.add_argument("--w8a16-num-warps", type=int, default=1)
    parser.add_argument(
        "--w8a16-scale-mode",
        choices=["bitcast", "exp2"],
        default="bitcast",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
