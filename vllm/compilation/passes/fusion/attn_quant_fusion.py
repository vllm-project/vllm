# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Callable

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

from ..vllm_inductor_pass import VllmFusionPatternMatcherPass, VllmPatternReplacement
from .matcher_utils import MatcherQuantFP8
from .rms_quant_fusion import QUANT_OPS

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
RESHAPE_OP = torch.ops.aten.reshape.default


_FP8_QUANT_KEY = QuantKey(dtype=FP8_DTYPE, scale=kStaticTensorScale, symmetric=True)


class AttnFp8StaticQuantPattern(VllmPatternReplacement[..., torch.Tensor]):
    """
    Fusion for Attention+Fp8StaticQuant.

    Only triggers when the attention implementation returns True in
    `fused_output_quant_supported()`. If the pattern is found, the
    Fp8StaticQuant op will be removed from the graph, and its scale
    will be passed into Attention op as the `output_scale` argument.
    """

    def __init__(self, layer: Attention, dtype: torch.dtype):
        self._layer_name = layer.layer_name
        self._num_heads = layer.num_heads
        self._head_size = layer.head_size
        self._dtype = dtype
        self._quant_matcher = MatcherQuantFP8(_FP8_QUANT_KEY)

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        def _pattern(
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
                layer_name=self._layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            attn_out_view = RESHAPE_OP(
                at1[1], [q.shape[0], self._num_heads * self._head_size]
            )
            return self._quant_matcher(attn_out_view, scale)[0]

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        def _replacement(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            output_attn: torch.Tensor,
            scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> torch.Tensor:
            output_attn = torch.empty(
                [q.shape[0], self._num_heads, self._head_size],
                dtype=FP8_DTYPE,
                device=q.device,
            )
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=self._layer_name,
                output_scale=scale,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return RESHAPE_OP(at1[1], [-1, self._num_heads * self._head_size])

        return _replacement

    def get_inputs(self):
        dtype = self._dtype
        num_heads = self._num_heads
        head_size = self._head_size
        return [
            self.empty(5, num_heads, head_size, dtype=dtype),  # q
            self.empty(5, num_heads, head_size, dtype=dtype),  # k
            self.empty(5, num_heads, head_size, dtype=dtype),  # v
            self.empty(5, num_heads, head_size, dtype=dtype),  # attn_output
            self.empty_fp32(1, 1),  # scale
            self.empty(0, dtype=dtype),  # kv_cache_dummy_dep
        ]


class AttnNvfp4QuantPattern(
    VllmPatternReplacement[..., tuple[torch.Tensor, torch.Tensor]]
):
    """
    Fusion for Attention+Nvfp4Quant.

    Only triggers when the attention implementation returns True in
    `fused_output_quant_supported()`. If the pattern is found, the
    Nvfp4Quant op will be removed from the graph, and its scale
    will be passed into Attention op as the `output_scale` argument.
    """

    def __init__(self, layer: Attention, dtype: torch.dtype):
        self._layer_name = layer.layer_name
        self._num_heads = layer.num_heads
        self._head_size = layer.head_size
        self._dtype = dtype
        self._QUANT_OP = QUANT_OPS[kNvfp4Dynamic]

    @property
    def pattern(self) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        def _pattern(
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
                layer_name=self._layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            attn_out_view = RESHAPE_OP(
                at1[1], [q.shape[0], self._num_heads * self._head_size]
            )
            at2 = auto_functionalized(
                self._QUANT_OP,
                input=attn_out_view,
                input_scale=input_scale,
                is_sf_swizzled_layout=True,
                output=output_quant,
                output_scale=output_scale,
            )
            output_scale_view = torch.ops.aten.view.dtype(at2[2], FP8_DTYPE)
            return at2[1], output_scale_view

        return _pattern

    @property
    def replacement(self) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        def _replacement(
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
                [q.shape[0], self._num_heads, self._head_size // 2],
                dtype=FP4_DTYPE,
                device=q.device,
            )
            output_scale_view = torch.ops.aten.view.dtype(output_scale, FP8_DTYPE)
            at2 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=self._layer_name,
                output_scale=input_scale,
                output_block_scale=output_scale_view,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            output = RESHAPE_OP(at2[1], [-1, self._num_heads * self._head_size // 2])
            return output, at2[2]

        return _replacement

    def get_inputs(self):
        dtype = self._dtype
        num_heads = self._num_heads
        head_size = self._head_size
        return [
            self.empty_bf16(5, num_heads, head_size),  # q
            self.empty_bf16(5, num_heads, head_size),  # k
            self.empty_bf16(5, num_heads, head_size),  # v
            self.empty_bf16(5, num_heads, head_size),  # output_attn
            self.empty(5, num_heads * head_size // 2, dtype=FP4_DTYPE),  # output_quant
            self.empty_i32(
                128, round_up(num_heads * head_size // 16, 4)
            ),  # output_scale
            self.empty_fp32(1, 1),  # input_scale
            self.empty(0, dtype=dtype),  # kv_cache_dummy_dep
        ]


class AttnQuantFusionPass(VllmFusionPatternMatcherPass):
    """
    This pass fuses post-attention quantization onto attention if supported.

    It uses the pattern matcher and matches each layer manually, as strings
    cannot be wildcarded. This also lets us check support on attention layers
    upon registration instead of during pattern matching.

    Currently, only static fp8 quant is supported, but patterns could easily be
    added for other quant schemes and dtypes. The bigger hurdle for wider
    support are attention kernels, which need to support fusing output quant.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "attn_quant_fusion_pass")

        dtype = config.model_config.dtype
        layers = list(get_layers_from_vllm_config(config, Attention).values())

        if len(layers) == 0:
            logger.warning(
                "Attention + quant fusion is enabled, but no attention layers "
                "were found in CompilationConfig.static_forward_context "
                "so no fusion patterns were registered."
            )

        for layer in layers:
            if layer.impl.fused_output_quant_supported(_FP8_QUANT_KEY):
                self.register(AttnFp8StaticQuantPattern(layer, dtype))

        if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
            for layer in layers:
                if layer.impl.fused_output_quant_supported(kNvfp4Dynamic):
                    self.register(AttnNvfp4QuantPattern(layer, dtype))

        self.dump_patterns(config, self.pm_pass)
