# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ParamSpec

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kStaticTensorScale,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

from ..fx_utils import is_func
from ..vllm_inductor_pass import PatternReplacement, make_fusion_pass
from .matcher_utils import MatcherQuantFP8
from .rms_quant_fusion import QUANT_OPS, empty_bf16, empty_fp32, empty_i32

logger = init_logger(__name__)
P = ParamSpec("P")
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
RESHAPE_OP = torch.ops.aten.reshape.default


_FP8_QUANT_KEY = QuantKey(dtype=FP8_DTYPE, scale=kStaticTensorScale, symmetric=True)


def _fx_view_to_reshape(gm: torch.fx.GraphModule) -> None:
    from torch._inductor.fx_passes.post_grad import view_to_reshape

    view_to_reshape(gm)


def _remove_noop_permutes(gm: torch.fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        if not is_func(node, torch.ops.aten.permute.default):
            continue
        dims = node.args[1]
        if any(dim != i for i, dim in enumerate(dims)):
            continue
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)


def attn_fp8_static_quant(layer: Attention, dtype: torch.dtype) -> PatternReplacement:
    layer_name = layer.layer_name
    num_heads = layer.num_heads
    head_size = layer.head_size
    quant_matcher = MatcherQuantFP8(_FP8_QUANT_KEY)

    def pattern(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output_attn: torch.Tensor,
        scale: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor,
    ) -> torch.Tensor:
        at1 = auto_functionalized(
            ATTN_OP,
            query=q,
            key=k,
            value=v,
            output=output_attn,
            layer_name=layer_name,
            output_scale=None,
            output_block_scale=None,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
        attn_out_view = RESHAPE_OP(at1[1], [q.shape[0], num_heads * head_size])
        return quant_matcher(attn_out_view, scale)[0]

    def replacement(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output_attn: torch.Tensor,
        scale: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor,
    ) -> torch.Tensor:
        output_attn = torch.empty(
            [q.shape[0], num_heads, head_size], dtype=FP8_DTYPE, device=q.device
        )
        at1 = auto_functionalized(
            ATTN_OP,
            query=q,
            key=k,
            value=v,
            output=output_attn,
            layer_name=layer_name,
            output_scale=scale,
            output_block_scale=None,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
        return RESHAPE_OP(at1[1], [-1, num_heads * head_size])

    inputs = [
        torch.empty(5, num_heads, head_size, dtype=dtype, device="cuda"),  # q
        torch.empty(5, num_heads, head_size, dtype=dtype, device="cuda"),  # k
        torch.empty(5, num_heads, head_size, dtype=dtype, device="cuda"),  # v
        torch.empty(5, num_heads, head_size, dtype=dtype, device="cuda"),  # attn_output
        empty_fp32(1, 1),  # scale
        torch.empty(0, dtype=dtype, device="cuda"),  # kv_cache_dummy_dep
    ]
    return PatternReplacement(pattern, replacement, inputs)


def attn_nvfp4_quant(layer: Attention, dtype: torch.dtype) -> PatternReplacement:
    layer_name = layer.layer_name
    num_heads = layer.num_heads
    head_size = layer.head_size
    QUANT_OP = QUANT_OPS[kNvfp4Dynamic]

    def pattern(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output_attn: torch.Tensor,
        output_quant: torch.Tensor,
        output_scale: torch.Tensor,
        input_scale: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        at1 = auto_functionalized(
            ATTN_OP,
            query=q,
            key=k,
            value=v,
            output=output_attn,
            layer_name=layer_name,
            output_scale=None,
            output_block_scale=None,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
        attn_out_view = RESHAPE_OP(at1[1], [q.shape[0], num_heads * head_size])
        at2 = auto_functionalized(
            QUANT_OP,
            output=output_quant,
            input=attn_out_view,
            output_scale=output_scale,
            input_scale=input_scale,
            is_sf_swizzled_layout=True,
        )
        output_scale_view = torch.ops.aten.view.dtype(at2[2], FP8_DTYPE)
        return at2[1], output_scale_view

    def replacement(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output_attn: torch.Tensor,
        _output_quant: torch.Tensor,
        output_scale: torch.Tensor,
        input_scale: torch.Tensor,
        kv_cache_dummy_dep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_attn = torch.empty(
            [q.shape[0], num_heads, head_size // 2], dtype=FP4_DTYPE, device=q.device
        )
        output_scale_view = torch.ops.aten.view.dtype(output_scale, FP8_DTYPE)
        at2 = auto_functionalized(
            ATTN_OP,
            query=q,
            key=k,
            value=v,
            output=output_attn,
            layer_name=layer_name,
            output_scale=input_scale,
            output_block_scale=output_scale_view,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
        output = RESHAPE_OP(at2[1], [-1, num_heads * head_size // 2])
        return output, at2[2]

    inputs = [
        empty_bf16(5, num_heads, head_size),  # q
        empty_bf16(5, num_heads, head_size),  # k
        empty_bf16(5, num_heads, head_size),  # v
        empty_bf16(5, num_heads, head_size),  # output_attn
        torch.empty(
            5, num_heads * head_size // 2, dtype=FP4_DTYPE, device="cuda"
        ),  # output_quant
        empty_i32(128, round_up(num_heads * head_size // 16, 4)),  # output_scale
        empty_fp32(1, 1),  # input_scale
        torch.empty(0, dtype=dtype, device="cuda"),  # kv_cache_dummy_dep
    ]
    return PatternReplacement(pattern, replacement, inputs)


def build_attn_quant_patterns(config: VllmConfig) -> list[PatternReplacement]:
    dtype = config.model_config.dtype
    layers = get_layers_from_vllm_config(config, Attention).values()

    patterns = [
        attn_fp8_static_quant(layer, dtype)
        for layer in layers
        if layer.impl.fused_output_quant_supported(_FP8_QUANT_KEY)
    ]

    if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
        patterns += [
            attn_nvfp4_quant(layer, dtype)
            for layer in layers
            if layer.impl.fused_output_quant_supported(kNvfp4Dynamic)
        ]

    return patterns


AttnQuantFusionPass = make_fusion_pass(
    pass_name="attn_quant_fusion_pass",
    builder=build_attn_quant_patterns,
    preprocessors=[_fx_view_to_reshape, _remove_noop_permutes],
    extra_sources=[attn_fp8_static_quant, attn_nvfp4_quant],
)
