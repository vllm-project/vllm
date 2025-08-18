# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses.fake_tensor import (FakeTensorMode,
                                           unset_fake_temporarily)

from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .fusion import QUANT_OPS, GroupShape, QuantKey, empty_bf16, empty_fp32
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
RESHAPE_OP = torch.ops.aten.reshape.default


class AttentionStaticQuantPattern:

    def __init__(
        self,
        layer_name: str,
        num_heads: int,
        head_size: int,
        quant_dtype: torch.dtype,
        symmetric=True,
    ):
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.quant_dtype = quant_dtype
        self.quant_key = QuantKey(dtype=quant_dtype,
                                  static=True,
                                  group_shape=GroupShape.PER_TENSOR,
                                  symmetric=symmetric)
        assert self.quant_key in QUANT_OPS, \
            f"unsupported quantization scheme {self.quant_key}"
        self.QUANT_OP = QUANT_OPS[self.quant_key]

    def empty_quant(self, *args, **kwargs):
        kwargs = {'dtype': self.quant_dtype, 'device': "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    def register_if_supported(self, pm_pass: PatternMatcherPass,
                              layer: Attention):
        if layer.impl.fused_output_quant_supported(self.quant_dtype,
                                                   self.quant_key.static,
                                                   self.quant_key.group_shape):
            self._register(pm_pass)

    def _register(self, pm_pass: PatternMatcherPass):

        def pattern(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    output_attn: torch.Tensor, output_quant: torch.Tensor,
                    scale: torch.Tensor):
            view_7 = RESHAPE_OP(output_attn,
                                [-1, self.num_heads, self.head_size])

            at1 = auto_functionalized(ATTN_OP,
                                      query=q,
                                      key=k,
                                      value=v,
                                      output=view_7,
                                      layer_name=self.layer_name,
                                      output_scale=None)
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
            view_7 = RESHAPE_OP(output_quant,
                                [-1, self.num_heads, self.head_size])

            at1 = auto_functionalized(ATTN_OP,
                                      query=q,
                                      key=k,
                                      value=v,
                                      output=view_7,
                                      layer_name=self.layer_name,
                                      output_scale=scale)

            return RESHAPE_OP(at1[1], [-1, self.num_heads * self.head_size])

        # Need custom fake mode, otherwise tracing happens with real tensors.
        # That would not work for the unified_attention custom op.
        with unset_fake_temporarily(), FakeTensorMode():
            inputs = [
                empty_bf16(5, self.num_heads, self.head_size),  # q
                empty_bf16(5, self.num_heads, self.head_size),  # k
                empty_bf16(5, self.num_heads, self.head_size),  # v
                empty_bf16(5, self.num_heads * self.head_size),  # attn_output
                self.empty_quant(5, self.num_heads *
                                 self.head_size),  # quant_output
                empty_fp32(1, 1)  # scale
            ]

            def wrap_trace_fn(process_fx, trace_fn):

                def wrapped(*args, **kwargs):
                    return process_fx(trace_fn(*args, **kwargs))

                return wrapped

            def fx_view_to_reshape(gm: torch.fx.GraphModule):
                from torch._inductor.fx_passes.post_grad import view_to_reshape
                view_to_reshape(gm)
                return gm

            pm.register_replacement(
                pattern, replacement, inputs,
                wrap_trace_fn(fx_view_to_reshape, pm.fwd_only), pm_pass)


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
        self.static_fwd_ctx = config.compilation_config.static_forward_context

        self.patterns = PatternMatcherPass(pass_name="attn_fusion_pass")

        for key, layer in self.static_fwd_ctx.items():
            pattern = AttentionStaticQuantPattern(key, layer.num_heads,
                                                  layer.head_size,
                                                  current_platform.fp8_dtype())
            pattern.register_if_supported(self.patterns, layer)
        if len(self.static_fwd_ctx) == 0:
            logger.warning(
                "Attention + quant fusion is enabled, but "
                "CompilationConfig.static_forward_context is empty. "
                "Cannot access attention layers so no fusion "
                "patterns were registered.")

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        self.begin()
        self.dump_graph(graph, "before_attn_fusion")

        count = self.patterns.apply(graph)
        logger.debug("Fused quantization onto %s attention nodes", count)
        self.dump_graph(graph, "after_attn_fusion")
        self.end_and_log()

    def uuid(self):
        return VllmInductorPass.hash_source(self, AttentionStaticQuantPattern)
