# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import operator
from collections.abc import Callable

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm._aiter_ops import rocm_aiter_ops
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
from vllm.utils.torch_utils import _USE_LAYERNAME, _encode_layer_name

from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)
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
        # When _USE_LAYERNAME is enabled (torch >= 2.11), layer_name is
        # passed as an explicit pattern input so the pattern matcher
        # treats it as a wildcard matching hoisted LayerName placeholders.
        # Otherwise it stays as a closure constant (original behavior).
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _pattern_with_ln(  # type: ignore[misc]
                q, k, v, output_attn, scale, kv_cache_dummy_dep, layer_name
            ):
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
                attn_out_view = RESHAPE_OP(
                    at1[1], [q.shape[0], self._num_heads * self._head_size]
                )
                return self._quant_matcher(attn_out_view, scale)[0]

            return _pattern_with_ln

        def _pattern(q, k, v, output_attn, scale, kv_cache_dummy_dep):
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=_ln,
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
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _replacement_with_ln(  # type: ignore[misc]
                q, k, v, output_attn, scale, kv_cache_dummy_dep, layer_name
            ):
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
                    layer_name=layer_name,
                    output_scale=scale,
                    output_block_scale=None,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
                return RESHAPE_OP(at1[1], [-1, self._num_heads * self._head_size])

            return _replacement_with_ln

        def _replacement(q, k, v, output_attn, scale, kv_cache_dummy_dep):
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
                layer_name=_ln,
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
        inputs: list = [
            self.empty(5, num_heads, head_size, dtype=dtype),  # q
            self.empty(5, num_heads, head_size, dtype=dtype),  # k
            self.empty(5, num_heads, head_size, dtype=dtype),  # v
            self.empty(5, num_heads, head_size, dtype=dtype),  # attn_output
            self.empty_fp32(1, 1),  # scale
            self.empty(0, dtype=dtype),  # kv_cache_dummy_dep
        ]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self._layer_name))
        return inputs


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
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _pattern_with_ln(  # type: ignore[misc]
                q,
                k,
                v,
                output_attn,
                output_quant,
                output_scale,
                input_scale,
                kv_cache_dummy_dep,
                layer_name,
            ):
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
                return at2[1], torch.ops.aten.view.dtype(at2[2], FP8_DTYPE)

            return _pattern_with_ln

        def _pattern(
            q,
            k,
            v,
            output_attn,
            output_quant,
            output_scale,
            input_scale,
            kv_cache_dummy_dep,
        ):
            at1 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=_ln,
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
            return at2[1], torch.ops.aten.view.dtype(at2[2], FP8_DTYPE)

        return _pattern

    @property
    def replacement(self) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _replacement_with_ln(  # type: ignore[misc]
                q,
                k,
                v,
                output_attn,
                _output_quant,
                output_scale,
                input_scale,
                kv_cache_dummy_dep,
                layer_name,
            ):
                output_attn = torch.empty(
                    [q.shape[0], self._num_heads, self._head_size // 2],
                    dtype=FP4_DTYPE,
                    device=q.device,
                )
                osv = torch.ops.aten.view.dtype(output_scale, FP8_DTYPE)
                at2 = auto_functionalized(
                    ATTN_OP,
                    query=q,
                    key=k,
                    value=v,
                    output=output_attn,
                    layer_name=layer_name,
                    output_scale=input_scale,
                    output_block_scale=osv,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
                return RESHAPE_OP(
                    at2[1], [-1, self._num_heads * self._head_size // 2]
                ), at2[2]

            return _replacement_with_ln

        def _replacement(
            q,
            k,
            v,
            output_attn,
            _output_quant,
            output_scale,
            input_scale,
            kv_cache_dummy_dep,
        ):
            output_attn = torch.empty(
                [q.shape[0], self._num_heads, self._head_size // 2],
                dtype=FP4_DTYPE,
                device=q.device,
            )
            osv = torch.ops.aten.view.dtype(output_scale, FP8_DTYPE)
            at2 = auto_functionalized(
                ATTN_OP,
                query=q,
                key=k,
                value=v,
                output=output_attn,
                layer_name=_ln,
                output_scale=input_scale,
                output_block_scale=osv,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return RESHAPE_OP(
                at2[1], [-1, self._num_heads * self._head_size // 2]
            ), at2[2]

        return _replacement

    def get_inputs(self):
        dtype = self._dtype
        num_heads = self._num_heads
        head_size = self._head_size
        inputs: list = [
            self.empty_bf16(5, num_heads, head_size),  # q
            self.empty_bf16(5, num_heads, head_size),  # k
            self.empty_bf16(5, num_heads, head_size),  # v
            self.empty_bf16(5, num_heads, head_size),  # output_attn
            self.empty(5, num_heads * head_size // 2, dtype=FP4_DTYPE),
            self.empty_i32(128, round_up(num_heads * head_size // 16, 4)),
            self.empty_fp32(1, 1),  # input_scale
            self.empty(0, dtype=dtype),  # kv_cache_dummy_dep
        ]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self._layer_name))
        return inputs


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
        super().__init__(config, "attn_quant_fusion")

        dtype = config.model_config.dtype
        layers = list(get_layers_from_vllm_config(config, Attention).values())

        if len(layers) == 0:
            logger.warning(
                "Attention + quant fusion is enabled, but no attention layers "
                "were found in CompilationConfig.static_forward_context "
                "so no fusion patterns were registered."
            )

        # When _USE_LAYERNAME is enabled, layer_name is a wildcard so all
        # layers produce the same pattern — register once then break.
        for layer in layers:
            if layer.impl.fused_output_quant_supported(_FP8_QUANT_KEY):
                self.register(AttnFp8StaticQuantPattern(layer, dtype))
                if _USE_LAYERNAME:
                    break

        if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
            for layer in layers:
                if layer.impl.fused_output_quant_supported(kNvfp4Dynamic):
                    self.register(AttnNvfp4QuantPattern(layer, dtype))
                    if _USE_LAYERNAME:
                        break

        self.dump_patterns(config, self.pm_pass)

    def _rewrite_q_pre_quant_to_aiter(self, graph: torch.fx.Graph) -> int:
        """Rewrite static FP8 Q-pre-quant nodes to AITER's per-tensor path.

        Matches: rope → reshape(q) → static_scaled_fp8_quant → reshape →
                 unified_attention(query=...)
        Replaces the quant node with rocm_aiter_per_tensor_quant.
        """
        if not (rocm_aiter_ops.is_enabled()
                and hasattr(torch.ops.vllm, "rocm_aiter_per_tensor_quant")):
            return 0

        replacements = 0
        created_q: dict[torch.fx.Node, torch.fx.Node] = {}

        for node in list(graph.nodes):
            if not (node.op == "call_function"
                    and node.target == auto_functionalized
                    and len(node.args) > 0
                    and node.args[0]
                    == torch.ops._C.static_scaled_fp8_quant.default):
                continue

            q_flat = node.kwargs.get("input", None)
            scale = node.kwargs.get("scale", None)
            if q_flat is None or scale is None:
                continue
            if tuple(node.kwargs.get("group_shape", ())) != (-1, -1):
                continue

            if not (isinstance(q_flat, torch.fx.Node)
                    and q_flat.op == "call_function"
                    and q_flat.target == RESHAPE_OP):
                continue

            q_after = q_flat.args[0]
            if not (isinstance(q_after, torch.fx.Node)
                    and q_after.op == "call_function"
                    and q_after.target == operator.getitem
                    and len(q_after.args) == 2
                    and q_after.args[1] == 1):
                continue

            rope_node = q_after.args[0]
            if not (isinstance(rope_node, torch.fx.Node)
                    and rope_node.op == "call_function"
                    and rope_node.target == auto_functionalized
                    and len(rope_node.args) > 0
                    and rope_node.args[0] == torch.ops.vllm
                    .fused_rope_and_unified_kv_cache_update.default):
                continue

            matched_getitems: list[torch.fx.Node] = []
            for q_getitem in list(node.users):
                if not (q_getitem.op == "call_function"
                        and q_getitem.target == operator.getitem
                        and len(q_getitem.args) == 2
                        and q_getitem.args[1] == 1):
                    continue
                for q_view in list(q_getitem.users):
                    if not (q_view.op == "call_function"
                            and q_view.target == RESHAPE_OP):
                        continue
                    for attn in list(q_view.users):
                        if not (attn.op == "call_function"
                                and attn.target == auto_functionalized
                                and len(attn.args) > 0
                                and attn.args[0] == ATTN_OP):
                            continue
                        if attn.kwargs.get("query") is not q_view:
                            continue
                        if attn.kwargs.get("output_scale") is not None:
                            continue
                        if attn.kwargs.get("output_block_scale") is not None:
                            continue
                        key_node = attn.kwargs.get("key")
                        dep_node = attn.kwargs.get("kv_cache_dummy_dep")
                        if not (isinstance(key_node, torch.fx.Node)
                                and key_node.op == "call_function"
                                and key_node.target == operator.getitem
                                and key_node.args[0] is rope_node
                                and key_node.args[1] == 2):
                            continue
                        if not (isinstance(dep_node, torch.fx.Node)
                                and dep_node.op == "call_function"
                                and dep_node.target == operator.getitem
                                and dep_node.args[0] is rope_node
                                and dep_node.args[1] == 0):
                            continue
                        matched_getitems.append(q_getitem)
                        break

            if not matched_getitems:
                continue

            if node not in created_q:
                with graph.inserting_before(matched_getitems[0]):
                    aiter_pair = graph.call_function(
                        torch.ops.vllm.rocm_aiter_per_tensor_quant.default,
                        args=(q_flat, FP8_DTYPE, scale),
                    )
                    aiter_q = graph.call_function(
                        operator.getitem, args=(aiter_pair, 0)
                    )
                created_q[node] = aiter_q
            else:
                aiter_q = created_q[node]

            for q_getitem in matched_getitems:
                q_getitem.replace_all_uses_with(aiter_q)
                replacements += 1

        if replacements > 0:
            graph.eliminate_dead_code()
            graph.lint()
            if graph.owning_module is not None:
                graph.owning_module.recompile()
            logger.debug("Q pre-quant → AITER: %d replacements", replacements)

        return replacements

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        pm_matches = self.pm_pass.apply(graph)
        fallback_matches = self._rewrite_q_pre_quant_to_aiter(graph)
        self.matched_count = pm_matches + fallback_matches
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
