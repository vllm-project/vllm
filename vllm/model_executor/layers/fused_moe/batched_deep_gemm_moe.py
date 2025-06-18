# SPDX-License-Identifier: Apache-2.0
import importlib.util
from typing import Optional

import triton
import triton.language as tl
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.masked_kernels import (
    masked_per_token_group_quant_fp8)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None




@triton.jit
def _per_token_group_quant_fp8_3d(
    # Pointers ------------------------------------------------------------
    y_ptr,                 # FP16  activations  (E, T, H)
    y_q_ptr,               # FP8   quantized activations (E, T, H)

    y_s_ptr,               # FP32  scales (E, T, G)
    counts_ptr,            # INT32 number of tokens per expert (E)

    # Sizes ---------------------------------------------------------------
    E: tl.constexpr,       # num_experts
    T: tl.constexpr,       # max_num_tokens
    H: tl.constexpr,       # hidden dimension
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)

    # Strides for y (elements) -------------------------------------------
    stride_y_e,
    stride_y_t,
    stride_y_h,

    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,


    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,

    # Stride for counts (elements)
    stride_counts_e,

    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,

    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
):
    """Dynamic FP8 quantisation over a 3‑D tensor laid out **(E, T, H)**.

    *   Each program instance handles **one** `GROUP_SIZE`‑length slice along H
        for a single (expert *e*, token *t*).
    *   Scales are produced with shape **(E, T, G)** where
        `G = H // GROUP_SIZE` and with *element* strides
        `(T*G, 1, T)` so that the *token* dimension is the fastest‑varying in
        memory – matching the downstream reshape you showed.
    *   All strides are expressed **in elements**, not bytes.
    """

    G = H // GROUP_SIZE  # groups per hidden dim

    # ----------------------- map program id -> (e, g) --------------------
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int32)

    # block for H dimension
    cols = tl.arange(0, BLOCK)
    mask_h = cols < BLOCK

    # iterate over tokens for this (expert, group)
    t = tl.zeros([], tl.int32)
    while t < n_tokens:
        base_y_offset = e * stride_y_e + t * stride_y_t + g * GROUP_SIZE * stride_y_h
        base_yq_offset = e * stride_yq_e + t * stride_yq_t + g * GROUP_SIZE * stride_yq_h
        base_ys_offset = e * stride_ys_e + t * stride_ys_t + g * stride_ys_g

        mask = mask_h
        y = tl.load(y_ptr + base_y_offset + cols * stride_y_h,
                    mask=mask, other=0.0).to(tl.float32)

        _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
        y_s = _absmax / fp8_max

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + cols * stride_yq_h,
                 y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset, y_s)

        t += 1


def quant_fp8_3d(
    y: torch.Tensor,               # (E, T, H)
    tokens_per_expert: torch.Tensor, # (E,) number of valid tokens per expert
    group_size: int = 128,
    fp8_dtype = torch.float8_e4m3fn,
    eps: float = 1e-6,
):
    """Quantize y into FP8 with per‑(expert, token, group) scales.

    Only the first `tokens_per_expert[e]` tokens are quantized per expert;
    the remaining positions in each (E, T, H) slice are treated as padding.

    Returns `(y_q, y_s)` where
    * `y_q` is the FP8 tensor, same shape and **standard PyTorch order** as *y*.
    * `y_s` has shape `(E, T, H // group_size)` and element strides
      `(T * G, 1, T)` so that the *token* dimension is contiguous.
    """

    assert y.ndim == 3, "y must be (E, T, H)"
    E, T, H = y.shape
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E, \
        "tokens_per_expert must be shape (E,)"
    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    # ---------------- allocate outputs ----------------------------------
    y_q = torch.empty_like(y, dtype=fp8_dtype)

    # desired scale strides (elements): (T*G, 1, T)
    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T

    # allocate scale buffer with proper shape and stride
    y_s = torch.empty_strided((E, T, G), (stride_ys_e, stride_ys_t, stride_ys_g),
                              dtype=torch.float32, device=y.device)

    # ---------------- stride bookkeeping (elements, not bytes) ----------
    stride_y_e, stride_y_t, stride_y_h = y.stride()

    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()


    # stride for tokens_per_expert (elements)
    stride_cnt_e = tokens_per_expert.stride()[0]

    # static grid over experts and H-groups; tokens loop is internal to the kernel
    grid = (E * G,)

    f_info = torch.finfo(fp8_dtype)
    fp8_max = f_info.max
    fp8_min = -f_info.max

    _per_token_group_quant_fp8_3d[grid](
        y, y_q, y_s, tokens_per_expert,
        E, T, H, group_size,
        stride_y_e, stride_y_t, stride_y_h,
        stride_yq_e, stride_yq_t, stride_yq_h,
        stride_ys_e, stride_ys_t, stride_ys_g,
        stride_cnt_e,
        eps, fp8_min, fp8_max,
        BLOCK=group_size,
        num_warps=4,
    )

    return y_q, y_s

class BatchedDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    # The Deep Gemm kernels only support block size of 128
    DEEPGEMM_BLOCK_SHAPE = 128

    def __init__(self, max_num_tokens: int, world_size: int, dp_size: int,
                 block_shape: list[int]):
        """
        max_num_tokens: Maximum number of tokens from a DP Rank
        world_size: Number of EP ranks
        dp_size: Number of data-parallel ranks
        block_shape: Block quantization block shape
        """
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.dp_size = dp_size
        self.block_shape = block_shape

        assert (len(self.block_shape) == 2 and all(
            [v == self.DEEPGEMM_BLOCK_SHAPE for v in self.block_shape]))

    def supports_chunking(self) -> bool:
        return False

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        assert a.dim() == 2
        # FIXME (varun): We should be able to dispatch only from the leader
        # DP ranks in the case of TP > 1. At the moment, all the Ranks
        # end up sending their tokens. This needs to be fixed.
        num_dispatchers = self.world_size
        num_experts = local_num_experts
        max_num_tokens = a.size(
            0) if self.max_num_tokens is None else self.max_num_tokens
        workspace13 = (num_experts, max_num_tokens * num_dispatchers,
                       max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, (N // 2))
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_num_tokens: Optional[torch.Tensor],
    ):
        import deep_gemm as dg
        assert hidden_states.ndim == 3
        assert(w1_zp is None and w2_zp is None)
        assert(a2_scale is None)

        a1q = hidden_states
        _, N, K = w1.size()

        assert w2.size(1) == K

        E, max_num_tokens, N, K, top_k_num = mk._moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))
        workspace2 = _resize_cache(workspace2, (E, max_num_tokens, N // 2))

        # (from deepgemm docs) : A value hint (which is a value on CPU)
        # for the M expectation of each batch, correctly setting this value
        # may lead to better performance.
        expected_m = max_num_tokens

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked((a1q, a1q_scale),
                                                 (w1, w1_scale),
                                                 out=workspace1,
                                                 masked_m=expert_num_tokens,
                                                 expected_m=expected_m)

        self.masked_activation(activation, workspace2, workspace1,
                               expert_num_tokens)

        a2q, a2q_scale = quant_fp8_3d(workspace2,
                                      tokens_per_expert=expert_num_tokens,
                                      group_size=self.block_shape[1])

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked((a2q, a2q_scale),
                                                 (w2, w2_scale),
                                                 out=output,
                                                 masked_m=expert_num_tokens,
                                                 expected_m=expected_m)
