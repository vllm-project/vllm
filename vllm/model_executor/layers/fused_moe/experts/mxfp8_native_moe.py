# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MXFP8 (1x32 block, E8M0 scale) MoE for AMD CDNA3/CDNA4.

The expert GEMMs consume the FP8 E4M3 weights and their E8M0 block scales
directly (no dequant-to-BF16), and activations are MXFP8-quantized per token.
CDNA4 uses ``tl.dot_scaled`` and native MX matrix-core ops. CDNA3 stores the
weights as E4M3FNUZ, runs one native FP8 ``tl.dot`` per 32-value MX block, and
applies the E8M0 scale products in-register. Both paths keep weights compressed
in HBM instead of expanding them to persistent BF16.

Structure mirrors vLLM's ``fused_moe_kernel``: tokens are sorted by expert
(``moe_align_block_size``); each program computes a ``[BLOCK_M, BLOCK_N]`` tile
for one expert, accumulating over K with the architecture-specific fused path.
SwiGLU-OAI activation and the top-k weighted reduction run between/after the
two GEMMs.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (
    Mxfp8TritonExpertsBase,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_BF16_DECODE_TOKEN_THRESHOLD = 16


def _should_use_bf16_decode_fallback(moe_config: FusedMoEConfig) -> bool:
    """Limit duplicate BF16 weights to the short-context MiniMax-M3 TP case."""
    return (
        moe_config.ep_size == 1
        and moe_config.has_shared_experts
        and moe_config.num_experts == 128
        and moe_config.experts_per_token == 4
        and moe_config.hidden_dim == 6144
        and moe_config.intermediate_size == 3072
        and 0 < moe_config.max_model_len <= 4096
    )


@triton.jit
def _mxfp8_grouped_gemm_dot_scaled_kernel(
    a_ptr,
    a_scale_ptr,
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
    top_k,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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
    offs_sk = tl.arange(0, BLOCK_K // 32)
    a_row = offs_token // A_DIV

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    as_ptrs = a_scale_ptr + a_row[:, None] * stride_asm + offs_sk[None, :] * stride_ask
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
        + offs_sk[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=n_mask[:, None], other=0.0)
        asc = tl.load(as_ptrs, mask=token_mask[:, None], other=0)
        bsc = tl.load(bs_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(a, asc, "e4m3", b.T, bsc, "e4m3")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        as_ptrs += (BLOCK_K // 32) * stride_ask
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


@triton.jit
def _mxfp8_grouped_gemm_fnuz_kernel(
    a_ptr,
    a_scale_ptr,
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
    top_k,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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
    offs_k = tl.arange(0, 32)
    a_row = offs_token // A_DIV

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    as_ptrs = a_scale_ptr + a_row * stride_asm
    b_ptrs = (
        b_ptr
        + off_e * stride_be
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )
    bs_ptrs = b_scale_ptr + off_e * stride_bse + offs_n * stride_bsn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        for k_offset in tl.static_range(0, BLOCK_K, 32):
            a = tl.load(
                a_ptrs + k_offset * stride_ak,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(
                b_ptrs + k_offset * stride_bk,
                mask=n_mask[:, None],
                other=0.0,
            )
            asc = tl.load(
                as_ptrs + (k_offset // 32) * stride_ask,
                mask=token_mask,
                other=0,
            ).to(tl.uint16)
            bsc = tl.load(
                bs_ptrs + (k_offset // 32) * stride_bsk,
                mask=n_mask,
                other=0,
            ).to(tl.uint16)

            # E8M0 and BF16 use the same eight-bit biased exponent. Shift each
            # scale byte into a BF16 exponent field, as Marlin does, then form
            # the per-token/per-output scale product around the FP8 dot.
            asc_scale = (asc << 7).to(tl.bfloat16, bitcast=True)
            bsc_scale = (bsc << 7).to(tl.bfloat16, bitcast=True)
            block_scale = asc_scale[:, None].to(tl.float32) * bsc_scale[None, :].to(
                tl.float32
            )
            acc += tl.dot(a, b.T) * block_scale

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        as_ptrs += (BLOCK_K // 32) * stride_ask
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


def _grouped_gemm_mxfp8(
    a_q: torch.Tensor,  # [M, K] fp8 e4m3
    a_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    w: torch.Tensor,  # [E, N, K] fp8 e4m3
    w_scale: torch.Tensor,  # [E, N, K//32] uint8 (E8M0)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
    out_dtype: torch.dtype,
    a_div: int,
    mul_weight_by: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    M_routed = num_valid_tokens
    E, N, K = w.shape
    k_alignment = 32 if current_platform.is_fp8_fnuz() else 128
    assert K % k_alignment == 0, (
        f"MXFP8 native MoE requires K%{k_alignment}==0, got K={K}"
    )
    BLOCK_K = (
        128
        if current_platform.is_fp8_fnuz() and K % 128 == 0 and block_m <= 16
        else 64
        if current_platform.is_fp8_fnuz() and K % 64 == 0
        else 32
        if current_platform.is_fp8_fnuz()
        else 128
    )
    # moe_align_block_size allocates for the worst case where every expert is
    # active. At small batches that can be much larger than the number of
    # blocks that can contain valid assignments. Limit the launch to the
    # tighter static upper bound; the device-side num_post check handles the
    # remaining tail.
    max_post_padded = min(sorted_token_ids.shape[0], M_routed * block_m)
    if current_platform.is_fp8_fnuz() and block_m <= 16:
        # One wave per 32 output columns avoids the register pressure of the
        # original 128-column tile. At the very smallest routed batch, pairing
        # two waves in a 64-column program amortizes launch/indexing overhead.
        BLOCK_N = 64 if M_routed < 32 else 32
        num_warps = 2 if M_routed < 32 else 1
    elif current_platform.is_fp8_fnuz() and block_m >= 64 and N >= 2048 and K >= 2048:
        # EP prefill GEMMs remain register-bound at a 128-column tile even with
        # 64 rows. Two-wave 64-column programs expose more independent work.
        BLOCK_N = 64
        num_warps = 2
    else:
        BLOCK_N = 128
        num_warps = 4 if current_platform.is_fp8_fnuz() and block_m <= 32 else 8
    m_blocks = triton.cdiv(max_post_padded, block_m)
    n_blocks = triton.cdiv(N, BLOCK_N)

    # Under expert parallelism (expert_map set) tokens routed to non-local
    # experts are dropped from sorted_token_ids, so their output rows are never
    # written.
    alloc = torch.zeros if expert_map is not None else torch.empty
    out = alloc((M_routed, N), dtype=out_dtype, device=a_q.device)
    grid = (m_blocks, n_blocks)
    kernel = (
        _mxfp8_grouped_gemm_fnuz_kernel
        if current_platform.is_fp8_fnuz()
        else _mxfp8_grouped_gemm_dot_scaled_kernel
    )
    if current_platform.is_fp8_fnuz() and (
        a_q.dtype != torch.float8_e4m3fnuz or w.dtype != torch.float8_e4m3fnuz
    ):
        raise ValueError("gfx94x MXFP8 MoE requires E4M3FNUZ inputs.")
    kernel[grid](
        a_q,
        a_scale,
        w,
        w_scale,
        out,
        mul_weight_by if mul_weight_by is not None else a_q,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        num_valid_tokens,
        top_k,
        a_q.stride(0),
        a_q.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
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
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out


def fused_moe_mxfp8_native(
    hidden_states: torch.Tensor,  # [T, H] bf16
    w13: torch.Tensor,  # [E, 2I, H] fp8
    w13_scale: torch.Tensor,  # [E, 2I, H//32] uint8
    w2: torch.Tensor,  # [E, H, I] fp8
    w2_scale: torch.Tensor,  # [E, H, I//32] uint8
    topk_weights: torch.Tensor,  # [T, top_k]
    topk_ids: torch.Tensor,  # [T, top_k] (global expert ids)
    *,
    alpha: float,
    beta: float,
    limit: float | None,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    T, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    M = T * top_k

    if current_platform.is_fp8_fnuz():
        # Padding is per expert, so tile from the average expert occupancy
        # rather than the total routed-token count. MiniMax-M3 has 128 experts;
        # a 64-row tile wastes most of both GEMMs at low occupancy.
        tokens_per_expert = max(1, M // global_num_experts)
        block_m = max(16, min(1 << (tokens_per_expert - 1).bit_length(), 64))
    else:
        block_m = 64
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids,
        block_m,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=expert_map is not None,
    )

    # GEMM1: x (mxfp8) @ w13^T -> [M, 2I]
    a_q, a_s = mxfp8_e4m3_quantize(hidden_states)
    g1 = _grouped_gemm_mxfp8(
        a_q,
        a_s,
        w13,
        w13_scale,
        sorted_ids,
        expert_ids,
        num_post,
        M,
        top_k,
        block_m,
        hidden_states.dtype,
        a_div=top_k,
        expert_map=expert_map,
    )  # [M, 2I]

    # SwiGLU-OAI (split layout: gate=g1[:, :I], up=g1[:, I:]) FUSED with the
    # GEMM2 MXFP8 activation-quant in one fp32 Triton pass — no bf16 ``act``
    # round-trip to HBM. Bit-exact vs the unfused swiglu+quant chain on measured
    # MoE shapes, and ~1.2-1.9x faster on that step in isolation. (Not the #22
    # ``silu_and_mul_with_clamp`` op: it rounds intermediates to bf16, rel ~3e-3.)
    # Lazy import: the amd.ops package pulls in the minimax_m3 platform dispatch,
    # only resolvable after the model module finishes loading.
    from vllm.models.minimax_m3.amd.ops import swiglu_oai_quantize_mxfp8

    # GEMM2: act (mxfp8) @ w2^T -> [M, H], weighted by topk_weights, then reduce.
    act_q, act_s = swiglu_oai_quantize_mxfp8(g1, alpha=alpha, beta=beta, limit=limit)
    g2 = _grouped_gemm_mxfp8(
        act_q,
        act_s,
        w2,
        w2_scale,
        sorted_ids,
        expert_ids,
        num_post,
        M,
        top_k,
        block_m,
        hidden_states.dtype if current_platform.is_fp8_fnuz() else torch.float32,
        a_div=1,
        mul_weight_by=topk_weights.reshape(-1).to(torch.float32),
        expert_map=expert_map,
    )  # [M, H] == [T*top_k, H]

    if current_platform.is_fp8_fnuz():
        if output is None:
            output = torch.empty_like(hidden_states)
        ops.moe_sum(g2.view(T, top_k, H), output)
        return output

    result = g2.view(T, top_k, H).sum(dim=1).to(hidden_states.dtype)
    if output is not None:
        output.copy_(result)
        return output
    return result


class Mxfp8NativeTritonExperts(Mxfp8TritonExpertsBase):
    """Fused MXFP8 MoE on gfx94x/gfx95x."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.w1_bf16: torch.Tensor | None = None
        self.w2_bf16: torch.Tensor | None = None
        self.bf16_experts: TritonExperts | None = None
        if current_platform.is_fp8_fnuz() and _should_use_bf16_decode_fallback(
            moe_config
        ):
            bf16_config = biased_moe_quant_config(
                None,
                None,
                gemm1_alpha=quant_config.gemm1_alpha,
                gemm1_beta=quant_config.gemm1_beta,
                gemm1_clamp_limit=quant_config.gemm1_clamp_limit,
            )
            self.bf16_experts = TritonExperts(moe_config, bf16_config)

    @property
    def requires_bf16_decode_weights(self) -> bool:
        return self.bf16_experts is not None

    def bind_bf16_weights(
        self,
        w1_bf16: torch.Tensor,
        w2_bf16: torch.Tensor,
    ) -> None:
        if self.bf16_experts is None:
            raise RuntimeError("BF16 decode experts are not enabled for this config.")
        self.w1_bf16 = w1_bf16
        self.w2_bf16 = w2_bf16

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.quant_dtype

    @property
    def block_shape(self) -> list[int] | None:
        return self.quant_config.block_shape

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Activations are MXFP8-quantized inside ``fused_moe_mxfp8_native``.
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_rocm() and (
            current_platform.supports_mx() or current_platform.is_fp8_fnuz()
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if (
            self.bf16_experts is not None
            and hidden_states.shape[0] <= _BF16_DECODE_TOKEN_THRESHOLD
        ):
            if self.w1_bf16 is None or self.w2_bf16 is None:
                raise RuntimeError("BF16 decode weights were not bound after loading.")
            self.bf16_experts.apply(
                output=output,
                hidden_states=hidden_states,
                w1=self.w1_bf16,
                w2=self.w2_bf16,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=None,
                a2_scale=None,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            return

        alpha = self.quant_config.gemm1_alpha
        alpha = 1.702 if alpha is None else float(alpha)
        beta = self.quant_config.gemm1_beta
        beta = 1.0 if beta is None else float(beta)
        limit = self.quant_config.gemm1_clamp_limit
        limit = None if limit is None else float(limit)
        out = fused_moe_mxfp8_native(
            hidden_states,
            w1,
            self.w1_scale_val,
            w2,
            self.w2_scale_val,
            topk_weights,
            topk_ids,
            alpha=alpha,
            beta=beta,
            limit=limit,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            output=output,
        )
        assert out is output
