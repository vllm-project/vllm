# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses.fake_tensor import (FakeTensorMode,
                                           unset_fake_temporarily)

from vllm.attention import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import round_up

from .fusion import (QUANT_OPS, GroupShape, QuantKey, empty_bf16, empty_fp32,
                     empty_i32)
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
RESHAPE_OP = torch.ops.aten.reshape.default


class AttentionStaticQuantPattern:
    """
    Fusion for Attention+StaticQuant.

    Only triggers when the attention implementation returns True in
    `fused_output_quant_supported()`. If the pattern is found, the StaticQuant
    op will be removed from the graph, and its scale will be passed into
    Attention op as the `output_scale` argument.
    """

    def __init__(
        self,
        layer: Attention,
        quant_dtype: torch.dtype,
        symmetric=True,
    ):
        self.layer = layer
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.head_size = layer.head_size
        self.quant_dtype = quant_dtype

        if quant_dtype == FP8_DTYPE:
            self.quant_key = QuantKey(dtype=quant_dtype,
                                      static=True,
                                      group_shape=GroupShape.PER_TENSOR,
                                      symmetric=symmetric)
            assert self.quant_key in QUANT_OPS, \
                f"unsupported quantization scheme {self.quant_key}"
            self.QUANT_OP = QUANT_OPS[self.quant_key]
        elif quant_dtype == FP4_DTYPE:
            self.quant_key = QuantKey(dtype=quant_dtype)
            assert self.quant_key in QUANT_OPS, \
                f"unsupported quantization scheme {self.quant_key}"
            self.QUANT_OP = QUANT_OPS[self.quant_key]
        else:
            raise ValueError(f"Unsupported quantization dtype: {quant_dtype}")

    def empty_quant(self, *args, **kwargs):
        kwargs = {'dtype': self.quant_dtype, 'device': "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    @staticmethod
    def wrap_trace_fn(process_fx, trace_fn):

        def wrapped(*args, **kwargs):
            return process_fx(trace_fn(*args, **kwargs))

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule):
        from torch._inductor.fx_passes.post_grad import view_to_reshape
        view_to_reshape(gm)
        return gm

    def register_if_supported(self, pm_pass: PatternMatcherPass):
        if self.layer.impl.fused_output_quant_supported(
                self.quant_dtype, self.quant_key.static,
                self.quant_key.group_shape):
            if self.quant_dtype == FP8_DTYPE:
                self._register_fp8_quant(pm_pass)
            elif self.quant_dtype == FP4_DTYPE:
                self._register_nvfp4_quant(pm_pass)

    def _register_fp8_quant(self, pm_pass: PatternMatcherPass):

        def pattern(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    output_attn: torch.Tensor, output_quant: torch.Tensor,
                    scale: torch.Tensor):
            at1 = auto_functionalized(ATTN_OP,
                                      query=q,
                                      key=k,
                                      value=v,
                                      output=output_attn,
                                      layer_name=self.layer_name,
                                      output_scale=None,
                                      output_block_scale=None)
            attn_out_view = RESHAPE_OP(at1[1],
                                       [-1, self.num_heads * self.head_size])
            at2 = auto_functionalized(self.QUANT_OP,
                                      result=output_quant,
                                      input=attn_out_view,
                                      scale=scale)
            return at2[1]

        def replacement(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        output_attn: torch.Tensor, output_quant: torch.Tensor,
                        scale: torch.Tensor):
            # attn output in quant_dtype
            output_attn = torch.ops.aten.full.default(
                [q.shape[0], self.num_heads, self.head_size],
                0.0,
                dtype=self.quant_dtype,
                device=q.device)
            at1 = auto_functionalized(ATTN_OP,
                                      query=q,
                                      key=k,
                                      value=v,
                                      output=output_attn,
                                      layer_name=self.layer_name,
                                      output_scale=scale,
                                      output_block_scale=None)
            return RESHAPE_OP(at1[1], [-1, self.num_heads * self.head_size])

        # Need custom fake mode, otherwise tracing happens with real tensors.
        # That would not work for the unified_attention custom op.
        with unset_fake_temporarily(), FakeTensorMode():
            inputs = [
                empty_bf16(5, self.num_heads, self.head_size),  # q
                empty_bf16(5, self.num_heads, self.head_size),  # k
                empty_bf16(5, self.num_heads, self.head_size),  # v
                empty_bf16(5, self.num_heads, self.head_size),  # attn_output
                self.empty_quant(5, self.num_heads *
                                 self.head_size),  # quant_output
                empty_fp32(1, 1)  # scale
            ]

            pm.register_replacement(
                pattern, replacement, inputs,
                AttentionStaticQuantPattern.wrap_trace_fn(
                    AttentionStaticQuantPattern.fx_view_to_reshape,
                    pm.fwd_only), pm_pass)

    def _register_nvfp4_quant(self, pm_pass: PatternMatcherPass):

        def pattern(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    attn_output: torch.Tensor, quant_input_scale: torch.Tensor,
                    quant_output: torch.Tensor,
                    quant_output_scale: torch.Tensor):
            at1 = auto_functionalized(ATTN_OP,
                                      query=q,
                                      key=k,
                                      value=v,
                                      output=attn_output,
                                      layer_name=self.layer_name,
                                      output_scale=None,
                                      output_block_scale=None)
            attn_output_view = RESHAPE_OP(
                at1[1], [-1, self.num_heads * self.head_size])
            at2 = auto_functionalized(self.QUANT_OP,
                                      output=quant_output,
                                      input=attn_output_view,
                                      output_scale=quant_output_scale,
                                      input_scale=quant_input_scale)
            quant_output_scale_view = torch.ops.aten.view.dtype(
                at2[2], FP8_DTYPE)
            return at2[1], quant_output_scale_view

        def replacement(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        attn_output: torch.Tensor,
                        quant_input_scale: torch.Tensor,
                        quant_output: torch.Tensor,
                        quant_output_scale: torch.Tensor):
            # attention output in quant_dtype
            attn_output = torch.ops.aten.full.default(
                [q.shape[0], self.num_heads, self.head_size // 2],
                0.0,
                dtype=self.quant_dtype,
                device=q.device)
            # attention output block scale
            attn_output_block_scale = torch.ops.aten.view.dtype(
                quant_output_scale, FP8_DTYPE)
            at2 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=attn_output,
                layer_name=self.layer_name,
                output_scale=quant_input_scale,
                output_block_scale=attn_output_block_scale)
            output = RESHAPE_OP(at2[1],
                                [-1, self.num_heads * self.head_size // 2])
            return output, at2[2]

        # Need custom fake mode, otherwise tracing happens with real tensors.
        # That would not work for the unified_attention custom op.
        with unset_fake_temporarily(), FakeTensorMode():
            inputs = [
                empty_bf16(5, self.num_heads, self.head_size),  # q
                empty_bf16(5, self.num_heads, self.head_size),  # k
                empty_bf16(5, self.num_heads, self.head_size),  # v
                empty_bf16(5, self.num_heads, self.head_size),  # attn_output
                empty_fp32(1, 1),  # quant_input_scale
                self.empty_quant(5, self.num_heads * self.head_size //
                                 2),  # quant_output
                empty_i32(128,
                          round_up(self.num_heads * self.head_size // 16,
                                   4)),  # quant_output_scale
            ]

            pm.register_replacement(
                pattern, replacement, inputs,
                AttentionStaticQuantPattern.wrap_trace_fn(
                    AttentionStaticQuantPattern.fx_view_to_reshape,
                    pm.fwd_only), pm_pass)


class AttnFusionPass(VllmInductorPass):
    """
    This pass fuses post-attention quantization onto attention if supported.

    It uses the pattern matcher and matches each layer manually, as strings
    cannot be wildcarded. This also lets us check support on attention layers
    upon registration instead of during pattern matching.

    Currently, only static fp8 quant is supported, but patterns could easily be
    added for other quant schemes and dtypes. The bigger hurdle for wider
    support are attention kernels, which need to support fusing output quant.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        attn_layers_count = 0

        self.patterns = PatternMatcherPass(pass_name="attn_fusion_pass")

        for layer_name, layer in get_layers_from_vllm_config(
                config, Attention).items():
            pattern_fp8 = AttentionStaticQuantPattern(layer, FP8_DTYPE)
            pattern_fp8.register_if_supported(self.patterns)

            pattern_fp4 = AttentionStaticQuantPattern(layer, FP4_DTYPE)
            pattern_fp4.register_if_supported(self.patterns)

            attn_layers_count += 1

        if attn_layers_count == 0:
            logger.warning(
                "Attention + quant fusion is enabled, but no attention layers "
                "were found in CompilationConfig.static_forward_context "
                "so no fusion patterns were registered.")

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        self.begin()
        self.dump_graph(graph, "before_attn_fusion")

        count = self.patterns.apply(graph)
        logger.debug("Fused quantization onto %s attention nodes", count)
        self.dump_graph(graph, "after_attn_fusion")
        self.end_and_log()

    def uuid(self):
        return VllmInductorPass.hash_source(self, AttentionStaticQuantPattern)
