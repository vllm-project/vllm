# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MXFP8 (1x32 block, E8M0 scale) MoE for AMD CDNA4 (gfx950) via Triton
``tl.dot_scaled`` (hardware microscaling matmul).

The expert GEMMs consume the FP8 E4M3 weights and their E8M0 block scales
directly (no dequant-to-BF16), and activations are MXFP8-quantized per token.
On CDNA4 ``dot_scaled`` maps to the native MX matrix-core ops; on other archs
Triton upcasts to BF16 (so this stays correct, just not faster) — but the
oracle only selects this path on gfx950 and routes everything else to the
BF16 ``Mxfp8EmulationTritonExperts`` fallback.

Structure mirrors vLLM's ``fused_moe_kernel``: tokens are sorted by expert
(``moe_align_block_size``); each program computes a ``[BLOCK_M, BLOCK_N]`` tile
for one expert, accumulating over K with ``dot_scaled``. SwiGLU-OAI activation
and the top-k weighted reduction run in PyTorch between/after the two GEMMs.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.mxfp8_emulation_moe import (
    Mxfp8TritonExpertsBase,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _mxfp8_grouped_gemm_kernel(
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
    block_n: int = 128,
    num_warps: int = 8,
    num_stages: int = 2,
) -> torch.Tensor:
    M_routed = num_valid_tokens
    E, N, K = w.shape
    assert K % 128 == 0, f"MXFP8 native MoE requires K%128==0, got K={K}"
    # Under expert parallelism (expert_map set) tokens routed to non-local
    # experts are dropped from sorted_token_ids, so their output rows are never
    # written — zero them so the downstream reduction ignores their garbage.
    alloc = torch.zeros if expert_map is not None else torch.empty
    out = alloc((M_routed, N), dtype=out_dtype, device=a_q.device)
    BLOCK_K = 128
    grid = (triton.cdiv(sorted_token_ids.shape[0], block_m), triton.cdiv(N, block_n))
    _mxfp8_grouped_gemm_kernel[grid](
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
        BLOCK_N=block_n,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# Tuned native-MXFP8 launch tiles for gfx950 (CDNA4) at MiniMax-M3 MoE shapes.
# For example, 8k/1k, 1k/1k cases.

_MXFP8_PREFILL_TILES = dict(block_m=128, block_n=256, num_warps=8, num_stages=2)
_MXFP8_DECODE_TILES = dict(block_m=64, block_n=64, num_warps=4, num_stages=2)
_MXFP8_PREFILL_MIN_TOKENS = 1024


def _mxfp8_moe_tiles(num_tokens: int) -> dict:
    """Pick grouped-GEMM launch tiles by regime (token count)."""
    if num_tokens >= _MXFP8_PREFILL_MIN_TOKENS:
        return _MXFP8_PREFILL_TILES
    return _MXFP8_DECODE_TILES


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
) -> torch.Tensor:
    T, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    M = T * top_k

    tiles = _mxfp8_moe_tiles(T)
    block_m = tiles["block_m"]
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
        block_n=tiles["block_n"],
        num_warps=tiles["num_warps"],
        num_stages=tiles["num_stages"],
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
        torch.float32,
        a_div=1,
        mul_weight_by=topk_weights.reshape(-1).to(torch.float32),
        expert_map=expert_map,
        block_n=tiles["block_n"],
        num_warps=tiles["num_warps"],
        num_stages=tiles["num_stages"],
    )  # [M, H] == [T*top_k, H]

    return g2.view(T, top_k, H).sum(dim=1).to(hidden_states.dtype)


class Mxfp8NativeTritonExperts(Mxfp8TritonExpertsBase):
    """Native MXFP8 MoE (CDNA4 ``dot_scaled``) on gfx950."""

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
        return current_platform.is_rocm() and current_platform.supports_mx()

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
        )
        output.copy_(out)
