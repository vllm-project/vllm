# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import per_block_cast_to_int8
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize, BatchedTritonExperts, NaiveBatchedExperts)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.utils import round_up
from vllm.utils.deep_gemm import per_block_cast_to_fp8


def triton_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    quant_dtype: Optional[torch.dtype] = None,
    per_act_token_quant=False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    return fused_experts(a,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale,
                         per_channel_quant=per_act_token_quant,
                         use_fp8_w8a8=quant_dtype == torch.float8_e4m3fn,
                         block_shape=block_shape)


def batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    quant_dtype: Optional[torch.dtype] = None,
    per_act_token_quant: bool = False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    max_num_tokens = round_up(a.shape[0], 64)

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(max_num_tokens,
                                  num_dispatchers=1,
                                  num_local_experts=w1.shape[0],
                                  rank=0),
        BatchedTritonExperts(
            max_num_tokens=max_num_tokens,
            num_dispatchers=1,
            use_fp8_w8a8=quant_dtype == torch.float8_e4m3fn,
            per_act_token_quant=per_act_token_quant,
            block_shape=block_shape,
        ),
    )

    return fused_experts(a,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale)


def naive_batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    quant_dtype: Optional[torch.dtype] = None,
    per_act_token_quant: bool = False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    max_num_tokens = round_up(a.shape[0], 64)

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(max_num_tokens,
                                  num_dispatchers=1,
                                  num_local_experts=w1.shape[0],
                                  rank=0),
        NaiveBatchedExperts(
            max_num_tokens=max_num_tokens,
            num_dispatchers=1,
            use_fp8_w8a8=quant_dtype == torch.float8_e4m3fn,
            per_act_token_quant=per_act_token_quant,
            block_shape=block_shape,
        ),
    )

    return fused_experts(a,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         a1_scale=a1_scale,
                         a2_scale=a2_scale)


def chunk_scales(scales: Optional[torch.Tensor], start: int,
                 end: int) -> Optional[torch.Tensor]:
    if scales is not None:
        if scales.numel() == 1:
            return scales
        else:
            return scales[start:end]
    return None


def make_quantized_test_activations(
    E: int,
    m: int,
    k: int,
    in_dtype: torch.dtype,
    quant_dtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    per_act_token_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    a = torch.randn((E, m, k), device="cuda", dtype=in_dtype) / 10
    a_q = a
    a_scale = None

    if quant_dtype is not None:
        assert (quant_dtype == torch.float8_e4m3fn
                or quant_dtype == torch.int8), "only fp8/int8 supported"
        a_q = torch.zeros_like(a, dtype=quant_dtype)
        a_scale_l = [None] * E
        for e in range(E):
            a_q[e], a_scale_l[e] = moe_kernel_quantize_input(
                a[e], None, quant_dtype, per_act_token_quant, block_shape)
        a_scale = torch.stack(a_scale_l)

        if not per_act_token_quant and block_shape is None:
            a_scale = a_scale.view(E, 1, 1)

    return a, a_q, a_scale


def moe_quantize_weights(
    w: torch.Tensor,
    w_s: Optional[torch.Tensor],
    quant_dtype: Optional[torch.dtype],
    per_token_quant: bool,
    block_shape: Optional[list[int]],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    assert (quant_dtype == torch.float8_e4m3fn
            or quant_dtype == torch.int8), "only fp8/int8 supported"

    if block_shape is not None:
        assert not per_token_quant
        if quant_dtype == torch.int8:
            w, w_s = per_block_cast_to_int8(w, block_shape)
        else:
            w, w_s = per_block_cast_to_fp8(w, block_shape)
    else:
        if quant_dtype == torch.int8:
            w, w_s = ops.scaled_int8_quant(
                w, w_s, use_per_token_if_dynamic=per_token_quant)
        else:
            w, w_s = ops.scaled_fp8_quant(
                w, w_s, use_per_token_if_dynamic=per_token_quant)

    return w, w_s


def make_test_weight(
    e: int,
    rows: int,
    cols: int,
    in_dtype: torch.dtype = torch.bfloat16,
    quant_dtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    per_act_token_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    w_16 = torch.randn((e, rows, cols), device="cuda", dtype=in_dtype) / 15

    if quant_dtype is not None:
        w_l = [None] * e
        w_s_l = [None] * e
        for idx in range(e):
            w_l[idx], w_s_l[idx] = moe_quantize_weights(
                w_16[idx], None, quant_dtype, per_act_token_quant, block_shape)

        w = torch.stack(w_l)
        w_s = torch.stack(w_s_l)
        if w_s.ndim == 2:
            assert w_s.shape[-1] == 1
            w_s = w_s.view(-1, 1, 1)

        if block_shape is not None:
            block_n, block_k = block_shape
            n_tiles = (rows + block_n - 1) // block_n
            k_tiles = (cols + block_k - 1) // block_k
            assert w_s.shape == (e, n_tiles, k_tiles)
    else:
        w = w_16
        w_s = None

    return w_16, w, w_s


def make_test_weights(
    e: int,
    n: int,
    k: int,
    in_dtype: torch.dtype = torch.bfloat16,
    quant_dtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    per_act_token_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor,
           torch.Tensor, Optional[torch.Tensor]]:
    return (
        *make_test_weight(e, 2 * n, k, in_dtype, quant_dtype, block_shape,
                          per_act_token_quant),
        *make_test_weight(e, k, n, in_dtype, quant_dtype, block_shape,
                          per_act_token_quant),
    )
