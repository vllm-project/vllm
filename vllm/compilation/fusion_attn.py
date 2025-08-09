# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional

import regex as re
import torch
import torch._inductor.pattern_matcher as pm
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._subclasses.fake_tensor import (FakeTensorMode,
                                           unset_fake_temporarily)

import vllm.envs as envs
from vllm.attention import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.platforms import current_platform

from .fusion import QUANT_OPS, GroupShape, QuantKey, empty_bf16, empty_fp32
from .vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

ATTN_OP = torch.ops.vllm.unified_attention_with_output.default
RESHAPE_OP = torch.ops.aten.reshape.default


class AttentionQuantPattern:
    """
    The base class for Attn+Quant fusions.
    Should not be used directly.
    """

    def __init__(self, layer: Attention, quant_key: QuantKey):
        self.layer = layer
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.head_size = layer.head_size
        self.quant_dtype = quant_key.dtype
        self.quant_key = quant_key
        assert self.quant_key in QUANT_OPS, \
            f"unsupported quantization scheme {self.quant_key}"
        self.QUANT_OP = QUANT_OPS[self.quant_key]

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


class AttentionStaticQuantPattern(AttentionQuantPattern):
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
    ):
        quant_key = QuantKey(dtype=quant_dtype,
                             static=True,
                             group_shape=GroupShape.PER_TENSOR,
                             symmetric=True)
        super().__init__(layer, quant_key)

    def register_if_supported(self, pm_pass: PatternMatcherPass):
        if (self.layer.impl.fused_output_quant_supported(
                self.quant_key.dtype, self.quant_key.static,
                self.quant_key.group_shape)):
            self._register(pm_pass)

    def _register(self, pm_pass: PatternMatcherPass):

        def pattern(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    output_attn: torch.Tensor, output_quant: torch.Tensor,
                    out_scale: torch.Tensor, q_scale: torch.Tensor):
            view_7 = RESHAPE_OP(output_attn,
                                [-1, self.num_heads, self.head_size])

            at1 = auto_functionalized(ATTN_OP,
                                      query=q,
                                      key=k,
                                      value=v,
                                      output=view_7,
                                      layer_name=self.layer_name,
                                      query_scale=q_scale,
                                      output_scale=None)
            attn_out_view = RESHAPE_OP(at1[1],
                                       [-1, self.num_heads * self.head_size])

            at2 = auto_functionalized(self.QUANT_OP,
                                      result=output_quant,
                                      input=attn_out_view,
                                      scale=out_scale)
            return at2[1]

        def replacement(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        output_attn: torch.Tensor, output_quant: torch.Tensor,
                        out_scale: torch.Tensor, q_scale: torch.Tensor):
            view_7 = RESHAPE_OP(output_quant,
                                [-1, self.num_heads, self.head_size])

            at1 = auto_functionalized(ATTN_OP,
                                      query=q,
                                      key=k,
                                      value=v,
                                      output=view_7,
                                      layer_name=self.layer_name,
                                      query_scale=q_scale,
                                      output_scale=out_scale)

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
                empty_fp32(1, 1),  # out_scale
                empty_fp32(1, 1),  # q_scale
            ]

            pm.register_replacement(
                pattern, replacement, inputs,
                AttentionQuantPattern.wrap_trace_fn(
                    AttentionQuantPattern.fx_view_to_reshape, pm.fwd_only),
                pm_pass)


class QuantAttentionQuantPattern(AttentionQuantPattern):
    """
    Fusion for Attention+StaticQuant and replace it with StaticQuant+Attention.

    Only triggers when the attention implementation returns True for both
    fused_output_quant_supported() and insert_query_quant_supported().
    If the pattern is found, the StaticQuant op will be removed from the graph,
    its scale will be passed into Attention op as the output_scale argument.
    In addition, an additional StaticQuant will be inserted before the
    Attention to quantize the query input using the query_scale argument of the
    Attention op.
    """

    def __init__(
        self,
        layer: Attention,
        quant_dtype: torch.dtype,
        pre_quant_dtype: torch.dtype,
        cache_file: Optional[str] = None,
    ):
        # for matching post quant
        assert quant_dtype == current_platform.fp8_dtype()
        quant_key = QuantKey(dtype=quant_dtype,
                             static=True,
                             group_shape=GroupShape.PER_TENSOR,
                             symmetric=True)
        super().__init__(layer, quant_key)

        # for inserting pre quant
        assert pre_quant_dtype == current_platform.fp8_dtype()
        self.pre_quant_dtype = pre_quant_dtype
        self.pre_quant_key = QuantKey(dtype=pre_quant_dtype,
                                      static=True,
                                      group_shape=GroupShape.PER_TENSOR,
                                      symmetric=True)
        assert self.pre_quant_key in QUANT_OPS, \
            f"unsupported quantization scheme {self.pre_quant_key}"
        self.PRE_QUANT_OP = QUANT_OPS[self.pre_quant_key]

        # used for caching layer fusion info when fx graph is cached
        self.cache_file = cache_file

    def register_if_supported(self, pm_pass: PatternMatcherPass):
        attn_impl = self.layer.impl
        if (attn_impl.fused_output_quant_supported(self.quant_key.dtype,
                                                   self.quant_key.static,
                                                   self.quant_key.group_shape)
                and attn_impl.insert_query_quant_supported(
                    self.pre_quant_key.dtype, self.pre_quant_key.static,
                    self.pre_quant_key.group_shape)):
            self._register(pm_pass)

    def extra_check(self, match):
        """
        If the pattern is matched, set the Attention layer's fused_quant
        to True. Also, cache the result so that when the next run has the
        same input FX graph, we still remember to set the Attention layer's
        fused_quant to True before loading the cached compiled FX graph.
        """
        self.layer.fused_quant = True
        if self.cache_file is not None:
            with open(self.cache_file, "w") as file:
                file.write(str(int(self.layer.fused_quant)))
        logger.debug("%s fused the pattern QuantAttentionQuantPattern",
                     self.layer_name)
        return True

    def _register(self, pm_pass: PatternMatcherPass):

        def pattern(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    q_scale: torch.Tensor, attn_out: torch.Tensor,
                    output_scale: torch.Tensor, output: torch.Tensor):
            # attention
            at1 = auto_functionalized(ATTN_OP,
                                      query=q,
                                      key=k,
                                      value=v,
                                      output=attn_out,
                                      layer_name=self.layer_name,
                                      query_scale=q_scale,
                                      output_scale=None)
            # reshape output
            attn_out_view = RESHAPE_OP(at1[1],
                                       [-1, self.num_heads * self.head_size])
            # quant output
            at2 = auto_functionalized(self.QUANT_OP,
                                      result=output,
                                      input=attn_out_view,
                                      scale=output_scale)
            return at2[1]

        def replacement(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        q_scale: torch.Tensor, attn_out: torch.Tensor,
                        output_scale: torch.Tensor, output: torch.Tensor):
            # attention out in quant_dtype
            attn_out = torch.ops.aten.full.default(
                [q.shape[0], self.num_heads, self.head_size],
                0.0,
                dtype=self.quant_dtype,
                device=q.device)
            # q in pre_quant_dtype
            q_quant = torch.ops.aten.empty.memory_format(
                [q.shape[0], self.num_heads * self.head_size],
                dtype=self.pre_quant_dtype,
                device=q.device)

            # reshape q
            q_view1 = RESHAPE_OP(q, [-1, self.num_heads * self.head_size])
            # quant q
            at1 = auto_functionalized(self.PRE_QUANT_OP,
                                      result=q_quant,
                                      input=q_view1.contiguous(),
                                      scale=q_scale)
            # reshape q back
            q_view2 = RESHAPE_OP(at1[1], [-1, self.num_heads, self.head_size])
            # attention
            at2 = auto_functionalized(ATTN_OP,
                                      query=q_view2,
                                      key=k,
                                      value=v,
                                      output=attn_out,
                                      layer_name=self.layer_name,
                                      query_scale=q_scale,
                                      output_scale=output_scale)
            # reshape output
            output = RESHAPE_OP(at2[1], [-1, self.num_heads * self.head_size])
            return output

        # Need custom fake mode, otherwise tracing happens with real tensors.
        # That would not work for the unified_attention custom op.
        with unset_fake_temporarily(), FakeTensorMode():
            inputs = [
                empty_bf16(5, self.num_heads, self.head_size),  # q
                empty_bf16(5, self.num_heads, self.head_size),  # k
                empty_bf16(5, self.num_heads, self.head_size),  # v
                empty_fp32(1, 1),  # q_scale
                empty_bf16(5, self.num_heads, self.head_size),  # attn_out
                empty_fp32(1, 1),  # output_scale
                self.empty_quant(5, self.num_heads * self.head_size),  # output
            ]

            pm.register_replacement(
                pattern,
                replacement,
                inputs,
                AttentionQuantPattern.wrap_trace_fn(
                    AttentionQuantPattern.fx_view_to_reshape, pm.fwd_only),
                pm_pass,
                extra_check=lambda m: self.extra_check(m))


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

        # register AttentionStaticQuantPattern
        for layer_name, layer in get_layers_from_vllm_config(
                config, Attention).items():
            pattern1 = AttentionStaticQuantPattern(
                layer, current_platform.fp8_dtype())
            pattern1.register_if_supported(self.patterns)
            attn_layers_count += 1

        if attn_layers_count == 0:
            logger.warning('''
                Attention + quant fusion is enabled, but no attention layers
                were found in CompilationConfig.static_forward_context so
                no fusion patterns were registered.
                ''')

        # register QuantAttentionQuantPattern
        self.register_quant_attn_quant_pattern(config)

    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        self.begin()
        self.dump_graph(graph, "before_attn_fusion")

        count = self.patterns.apply(graph)
        graph.eliminate_dead_code()

        logger.debug("Fused quantization onto %s attention nodes", count)
        self.dump_graph(graph, "after_attn_fusion")
        self.end_and_log()

    def uuid(self):
        return VllmInductorPass.hash_source(self, AttentionStaticQuantPattern,
                                            QuantAttentionQuantPattern)

    def register_quant_attn_quant_pattern(self, config: VllmConfig):
        """
        Register QuantAttentionQuantPattern

        Instead of directly register the pattern, two WARs are involved to
        make the QuantAttentionQuantPattern fusion pass works.
        1. Caching is used to store the fusion status when fx graph is cached
        2. Finding the host output_scale offline for Flashinfer TRTLLM attn API
        """

        # Init cache. This cache will save the layer fusion info,
        # e.g. fused -> 1, not fused -> 0.
        # This info is required at runtime to determined the attn kernel
        # selection. Without this attn_fusion cache, the fusion status is
        # unknown when the fx graph is cached, since the cached compiled
        # graph will be loaded and the fusion pass will be skipped.
        disabled_cache = envs.VLLM_DISABLE_COMPILE_CACHE
        if not disabled_cache:
            # The TORCHINDUCTOR_CACHE_DIR env var is set in
            # compiler_interface.py and is not set by users.
            cache_dir = os.getenv("TORCHINDUCTOR_CACHE_DIR")
            # The cache_dir should have been created by compiler_interface.py.
            assert cache_dir is not None
            assert os.path.exists(cache_dir)
            cache_dir = os.path.join(cache_dir, "attn_fusion")
            os.makedirs(cache_dir, exist_ok=True)

        # Trying to find the input_scale from o_proj layer, which will be
        # used as attn layer output_scale for the fusion. Currently using
        # the layer name prefix matching to find the corresponding layers
        # within the same decoder.
        #
        # This is mainly because Flashinfer TRTLLM attn currently only
        # supports host scalar scale instead of device tensor scale, and
        # a device tensor is not allowed to transfer to host at runtime
        # with full cuda graph enabled.
        input_scale = {}  # map from layer prefix name to input_scale
        o_proj_pattern = re.compile(r"(.*layers\.\d+\.self_attn)\.o_proj$")
        attn_pattern = re.compile(r"(.*layers\.\d+\.self_attn)\.attn$")

        # Collect the input_scale for each self_attn.o_proj layer
        for layer_name, layer in get_layers_from_vllm_config(
                config, LinearBase).items():
            match = o_proj_pattern.search(layer_name)
            if match and hasattr(layer, "input_scale"):
                input_scale[match.group(1)] = layer.input_scale.item()

        # Only register the layer that the input_scale could be found
        for layer_name, layer in get_layers_from_vllm_config(
                config, Attention).items():
            match = attn_pattern.search(layer_name)
            # only register the pass when the input_scale is found
            if not match or not input_scale.get(match.group(1)):
                logger.debug(
                    "Cannot find o_proj layer or input_scale for fusing %s",
                    layer_name)
                continue
            layer._o_scale_float = input_scale[match.group(1)]

            # load the fusion status into the Attention layer's
            # fused_quant attribute
            cache_file: Optional[str] = None
            if not disabled_cache:
                assert cache_dir is not None
                cache_file = os.path.join(
                    cache_dir, VllmInductorPass.hash_source(layer_name))
                if os.path.exists(cache_file):
                    logger.debug("Found attn_fusion cache file: %s",
                                 cache_file)
                    with open(cache_file) as file:
                        fused = file.read().strip()
                        assert fused.isdigit()
                        layer.fused_quant = bool(fused)

            pattern2 = QuantAttentionQuantPattern(
                layer,
                quant_dtype=current_platform.fp8_dtype(),
                pre_quant_dtype=current_platform.fp8_dtype(),
                cache_file=cache_file)
            pattern2.register_if_supported(self.patterns)
