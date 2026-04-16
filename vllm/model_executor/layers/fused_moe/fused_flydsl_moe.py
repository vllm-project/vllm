# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE Triton kernels."""

import functools
import json
import os
from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    _get_config_dtype_str,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    disable_inplace,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.mxfp6_utils import dequant_mxfp6
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

import flydsl.compiler as flyc
from aiter.fused_moe import moe_sorting as aiter_moe_sorting
from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
)

logger = init_logger(__name__)

_FLYDSL_MOE_GEMM1_CACHE = {}
_FLYDSL_MOE_GEMM2_CACHE = {}

FLYDSL_MOE_TUNED_CONFIGS = {
    1: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 512,
        "tile_n2": 256,
        "tile_k2": 128
    },
    2: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 512,
        "tile_n2": 512,
        "tile_k2": 256
    },
    4: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 512,
        "tile_n2": 256,
        "tile_k2": 256
    },
    8: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 512,
        "tile_n2": 256,
        "tile_k2": 256
    },
    16: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 256,
        "tile_n2": 128,
        "tile_k2": 256
    },
    24: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 256,
        "tile_n2": 128,
        "tile_k2": 256
    },
    32: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 256,
        "tile_n2": 256,
        "tile_k2": 256
    },
    48: {
        "tile_m": 16,
        "tile_n": 128,
        "tile_k": 256,
        "tile_n2": 256,
        "tile_k2": 256
    },
    64: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "tile_n2": 256,
        "tile_k2": 256
    },
    128: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 256,
        "tile_n2": 128,
        "tile_k2": 256
    },
    256: {
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 256,
        "tile_n2": 256,
        "tile_k2": 256
    },
    512: {
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 128,
        "tile_n2": 256,
        "tile_k2": 256
    },
    1024: {
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 128,
        "tile_n2": 256,
        "tile_k2": 256
    },
    2048: {
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "tile_n2": 512,
        "tile_k2": 256
    },
    4096: {
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 128,
        "tile_n2": 256,
        "tile_k2": 256
    },
    8192: {
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 64,
        "tile_n2": 256,
        "tile_k2": 128
    }
}

def _maybe_aiter_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    block_m: int,
):
    topk_ids_i32 = topk_ids.to(torch.int32)
    topk_w_f32 = topk_weights.to(torch.float32)
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, _moe_buf = aiter_moe_sorting(
        topk_ids_i32,
        topk_w_f32,
        num_experts,
        model_dim,
        torch.float16,
        block_m,
    )
    # `num_valid_ids` is documented as [1]; some builds allocate [2]. Keep the first element.
    if num_valid_ids.numel() > 1:
        num_valid_ids = num_valid_ids[:1].contiguous()
    return sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids

def build_routing_buffers(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
    model_dim: int,
    tile_m: int,
):
    res = _maybe_aiter_moe_sorting(
        topk_ids,
        topk_weights,
        num_experts=experts,
        model_dim=model_dim,
        block_m=tile_m,
    )
    if res is None:
        raise RuntimeError("aiter moe_sorting failed/unavailable; cannot build routing buffers.")
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = res

    # Keep moe_sorting outputs as-is (no host trim/pad). Launch full expert-block range.
    sorted_token_ids = sorted_token_ids.contiguous()
    sorted_weights = sorted_weights.contiguous()
    sorted_expert_ids = sorted_expert_ids.contiguous()
    sorted_size = int(sorted_token_ids.numel())
    blocks = int(sorted_expert_ids.numel())
    return (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    )

@functools.lru_cache
def get_flydsl_config(M):
    return FLYDSL_MOE_TUNED_CONFIGS[min(FLYDSL_MOE_TUNED_CONFIGS.keys(), key=lambda x: abs(x - M))]

def _fused_flydsl_moe(
    hidden_states,
    w1,
    w2,
    experts,
    inter_dim,
    topk_weights,
    topk_ids,
    w1_scale=None,
    w2_scale=None,
    topk = 8,
    group_size=32,
    doweight_stage1=False,
    in_dtype="int4_bf16",
    out_dtype="bf16",
    scale_is_bf16=True
):  
    device = hidden_states.device
    tokens = hidden_states.shape[0]
    tuned_config = get_flydsl_config(tokens)
    model_dim = hidden_states.shape[1]
    out_torch_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    tile_m = tuned_config["tile_m"]
    tile_n = tuned_config["tile_n"]
    tile_k = tuned_config["tile_k"]
    tile_n2 = tuned_config["tile_n2"]
    tile_k2 = tuned_config["tile_k2"]

    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        model_dim=model_dim,
        tile_m=tile_m,
    )
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    ) = routing

    scale_x_1d = torch.empty((0,), device=device, dtype=torch.float32)
    sorted_weights_1d = sorted_weights.view(-1).contiguous()
    out_stage1 = torch.empty((tokens, topk, inter_dim), device=device, dtype=out_torch_dtype)

    stream = torch.cuda.current_stream()

    key1 = (model_dim,
        inter_dim,
        experts,
        topk,
        in_dtype,
        out_dtype,
        group_size,
        tile_m,
        tile_n,
        tile_k,
        bool(doweight_stage1),
        False)

    compiled_exe1 = _FLYDSL_MOE_GEMM1_CACHE.get(key1)
    if compiled_exe1 is None:
        exe1 = compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            group_size=group_size,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=bool(doweight_stage1),
            use_cshuffle_epilog=False,
            scale_is_bf16=scale_is_bf16
        )
        compiled_exe1 = flyc.compile(
            exe1,
            out_stage1,
            hidden_states,
            w1,
            scale_x_1d,
            w1_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights_1d,
            num_valid_ids,
            tokens,
            inter_dim,
            model_dim,
            int(blocks),
            stream
        )
        _FLYDSL_MOE_GEMM1_CACHE[key1] = compiled_exe1

    compiled_exe1(
        out_stage1,
        hidden_states,
        w1,
        scale_x_1d,
        w1_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
        num_valid_ids,
        tokens,
        inter_dim,
        model_dim,
        int(blocks),
        stream
    )

    a2_1d = out_stage1.view(-1).contiguous()
    a2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    out_stage2 = torch.empty((tokens, model_dim), device=device, dtype=out_torch_dtype)
    doweight_stage2 = not bool(doweight_stage1)

    if model_dim % tile_n2 != 0:
        raise ValueError(
            f"Invalid stage2 tiling: model_dim ({model_dim}) must be divisible by tile_n2 ({tile_n})."
        )
    if inter_dim % tile_k2 != 0:
        raise ValueError(
            "Invalid stage2 tiling: inter_dim ({inter_dim}) must be divisible by tile_k2 ({tile_k}). "
            "Try setting `--tile_k2` to a divisor of inter_dim. "
            "Tip: stage2 splits A2 loads across 256 threads; if you want smaller tile_k2, you may need a larger tile_m so (tile_m*tile_k2) stays divisible by 1024."
            .format(inter_dim=inter_dim, tile_k=tile_k2)
        )
    if (tile_m * tile_k2) % 256 != 0:
        raise ValueError(
            f"Invalid stage2 tiling: tile_m*tile_k2 must be divisible by 256 (total_threads=256). "
            f"Got tile_m={tile_m}, tile_k2={tile_k2} -> tile_m*tile_k2={tile_m * tile_k2}."
        )
    bytes_per_thread_x = (tile_m * tile_k2) // 256
    if bytes_per_thread_x % 4 != 0:
        raise ValueError(
            f"Invalid stage2 tiling for gmem loads: bytes_per_thread_x ((tile_m*tile_k2)/256) must be divisible by 4. "
            f"Got tile_m={tile_m}, tile_k2={tile_k2} -> bytes_per_thread_x={bytes_per_thread_x}. "
        )

    key2 = (
        model_dim,
        inter_dim,
        experts,
        topk,
        in_dtype,
        out_dtype,
        group_size,
        tile_m,
        tile_n2,
        tile_k2,
        bool(doweight_stage2)
    )

    compiled_exe2 = _FLYDSL_MOE_GEMM2_CACHE.get(key2)
    if compiled_exe2 is None:
        exe2 = compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            group_size=group_size,
            tile_m=tile_m,
            tile_n=tile_n2,
            tile_k=tile_k2,
            doweight_stage2=bool(doweight_stage2),
            scale_is_bf16=scale_is_bf16
        )
        compiled_exe2 = flyc.compile(
            exe2,
            out_stage2,
            a2_1d,
            w2,
            a2_scale_1d,
            w2_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights_1d,
            num_valid_ids,
            tokens,
            model_dim,
            inter_dim,
            int(blocks),
            stream,
        )
        _FLYDSL_MOE_GEMM2_CACHE[key2] = compiled_exe2

    out_stage2.zero_()
    compiled_exe2(
        out_stage2,
        a2_1d,
        w2,
        a2_scale_1d,
        w2_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
        num_valid_ids,
        tokens,
        model_dim,
        inter_dim,
        int(blocks),
        stream,
    )
    return out_stage2

def fused_flydsl_moe(
    hidden_states,
    w1,
    w2,
    num_experts,
    inter_dim,
    topk_weights,
    topk_ids,
    w1_scale=None,
    w2_scale=None,
    topk = 8,
    group_size=32,
    doweight_stage1=False,
    in_dtype="int4_bf16",
    out_dtype="bf16",
    scale_is_bf16=True,
):  
    return _fused_flydsl_moe(
        hidden_states,
        w1,
        w2,
        num_experts,
        inter_dim,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk=topk,
        group_size=group_size,
        doweight_stage1=doweight_stage1,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scale_is_bf16=scale_is_bf16
    )