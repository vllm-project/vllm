# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kStaticTensorScale,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

from ..fx_utils import is_func
from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherQuantFP8
from .rms_quant_fusion import QUANT_OPS, empty_fp32, empty_i32

logger = init_logger(__name__)

P = ParamSpec("P")

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

MLA_ATTN_OP = torch.ops.vllm.unified_mla_attention_with_output.default


class MLAAttentionQuantPattern(ABC):
    """
    Base class for MLA Attn+Quant fusions.
    MLA attention output is 2D (T, N*V) unlike standard attention (T, N, H).
    """

    def __init__(
        self,
        layer: MLAAttention,
        quant_key: QuantKey,
        dtype: torch.dtype,
    ) -> None:
        self.layer = layer
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.v_head_dim = layer.v_head_dim
        self.kv_lora_rank = layer.kv_lora_rank
        self.qk_nope_head_dim = layer.qk_nope_head_dim
        self.qk_rope_head_dim = layer.qk_rope_head_dim
        self.qk_head_dim = layer.qk_nope_head_dim + layer.qk_rope_head_dim
        self.output_dim = self.num_heads * self.v_head_dim
        self.quant_key = quant_key
        self.quant_dtype = quant_key.dtype
        self.dtype = dtype

        assert self.quant_key in QUANT_OPS, (
            f"unsupported quantization scheme {self.quant_key}"
        )
        self.QUANT_OP = QUANT_OPS[self.quant_key]

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": self.dtype, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    def empty_quant(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs = {"dtype": self.quant_dtype, "device": "cuda", **kwargs}
        return torch.empty(*args, **kwargs)

    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> fx.GraphModule:
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)
            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None:
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    @staticmethod
    def remove_noop_permutes(gm: torch.fx.GraphModule) -> None:
        for node in gm.graph.nodes:
            if not is_func(node, torch.ops.aten.permute.default):
                continue

            dims = node.args[1]
            if any(dim != i for i, dim in enumerate(dims)):
                continue

            # this is now an identity op, remove
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)

    def register_if_supported(self, pm_pass: PatternMatcherPass) -> None:
        if self.layer.impl.fused_output_quant_supported(self.quant_key):
            self._register(pm_pass)

    @abstractmethod
    def _register(self, pm_pass: PatternMatcherPass) -> None:
        raise NotImplementedError


class MLAAttentionFp8StaticQuantPattern(MLAAttentionQuantPattern):
    """
    Fusion for MLA Attention+Fp8StaticQuant.

    Matches the pattern: MLA attention -> static FP8 quant, and replaces
    it with MLA attention(output_scale=scale, output=fp8_buffer).
    """

    def __init__(
        self,
        layer: MLAAttention,
        dtype: torch.dtype,
        symmetric: bool = True,
    ) -> None:
        quant_key = QuantKey(
            dtype=FP8_DTYPE, scale=kStaticTensorScale, symmetric=symmetric
        )
        super().__init__(layer, quant_key, dtype)
        self.quant_matcher = MatcherQuantFP8(quant_key)

    def _register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            q: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            output_attn: torch.Tensor,
            scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> torch.Tensor:
            at1 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            # MLA output is already 2D (T, N*V), no reshape needed
            return self.quant_matcher(at1[1], scale)[0]

        def replacement(
            q: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            output_attn: torch.Tensor,
            scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> torch.Tensor:
            # MLA output in quant_dtype
            output_attn = torch.empty(
                [q.shape[0], self.output_dim],
                dtype=self.quant_dtype,
                device=q.device,
            )
            at1 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=scale,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return at1[1]

        inputs = [
            self.empty(5, self.num_heads, self.qk_head_dim),  # q
            self.empty(5, self.kv_lora_rank),  # kv_c_normed
            self.empty(5, 1, self.qk_rope_head_dim),  # k_pe
            self.empty(5, self.output_dim),  # output_attn
            empty_fp32(1, 1),  # scale
            self.empty(0),  # kv_cache_dummy_dep
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            MLAAttentionQuantPattern.wrap_trace_fn(
                pm.fwd_only,
                MLAAttentionQuantPattern.fx_view_to_reshape,
                MLAAttentionQuantPattern.remove_noop_permutes,
            ),
            pm_pass,
        )


class MLAAttentionNvfp4QuantPattern(MLAAttentionQuantPattern):
    """
    Fusion for MLA Attention+Nvfp4Quant.

    Matches the pattern: MLA attention -> NVFP4 quant, and replaces
    it with MLA attention(output_scale=scale, output_block_scale=block_scale,
    output=fp4_buffer).
    """

    def __init__(self, layer: MLAAttention, dtype: torch.dtype) -> None:
        super().__init__(layer, kNvfp4Dynamic, dtype)

    def _register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            q: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            output_attn: torch.Tensor,
            output_quant: torch.Tensor,
            output_scale: torch.Tensor,
            input_scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at1 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            at2 = auto_functionalized(
                self.QUANT_OP,
                output=output_quant,
                input=at1[1],
                output_scale=output_scale,
                input_scale=input_scale,
                is_sf_swizzled_layout=True,
            )
            output_scale_view = torch.ops.aten.view.dtype(at2[2], FP8_DTYPE)
            return at2[1], output_scale_view

        def replacement(
            q: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            output_attn: torch.Tensor,
            output_quant: torch.Tensor,
            output_scale: torch.Tensor,
            input_scale: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # MLA output in quant_dtype (FP4 packed as uint8)
            output_attn = torch.empty(
                [q.shape[0], self.output_dim // 2],
                dtype=self.quant_dtype,
                device=q.device,
            )
            # attention output block scale
            output_scale_view = torch.ops.aten.view.dtype(output_scale, FP8_DTYPE)
            at2 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=self.layer_name,
                output_scale=input_scale,
                output_block_scale=output_scale_view,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return at2[1], at2[2]

        inputs = [
            self.empty(5, self.num_heads, self.qk_head_dim),  # q
            self.empty(5, self.kv_lora_rank),  # kv_c_normed
            self.empty(5, 1, self.qk_rope_head_dim),  # k_pe
            self.empty(5, self.output_dim),  # output_attn
            self.empty_quant(5, self.output_dim // 2),  # output_quant
            empty_i32(128, round_up(self.output_dim // 16, 4)),  # output_scale
            empty_fp32(1, 1),  # input_scale
            self.empty(0),  # kv_cache_dummy_dep
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            MLAAttentionQuantPattern.wrap_trace_fn(
                pm.fwd_only,
                MLAAttentionQuantPattern.fx_view_to_reshape,
                MLAAttentionQuantPattern.remove_noop_permutes,
            ),
            pm_pass,
        )


class MLAAttnFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses post-attention quantization onto MLA attention if supported.

    It uses the pattern matcher and matches each MLA layer manually, as strings
    cannot be wildcarded. This also lets us check support on attention layers
    upon registration instead of during pattern matching.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns = PatternMatcherPass(pass_name="mla_attn_fusion_pass")

        mla_layers = get_layers_from_vllm_config(config, MLAAttention)
        for layer_name, layer in mla_layers.items():
            pattern_fp8 = MLAAttentionFp8StaticQuantPattern(
                layer, config.model_config.dtype
            )
            pattern_fp8.register_if_supported(self.patterns)

            if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
                pattern_nvfp4 = MLAAttentionNvfp4QuantPattern(
                    layer, config.model_config.dtype
                )
                pattern_nvfp4.register_if_supported(self.patterns)

        if len(mla_layers) == 0:
            logger.warning(
                "MLA attention + quant fusion is enabled, but no MLA "
                "attention layers were found in "
                "CompilationConfig.static_forward_context "
                "so no fusion patterns were registered."
            )

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Fused quant onto %s MLA attention nodes", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            MLAAttentionQuantPattern,
            MLAAttentionFp8StaticQuantPattern,
            MLAAttentionNvfp4QuantPattern,
        )
