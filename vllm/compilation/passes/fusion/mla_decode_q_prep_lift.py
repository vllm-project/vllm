# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lift the decode-side MLA q-prep block out of ``unified_mla_attention_with_output``.

Inserts a graph-visible ``mla_decode_q_prep(q, layer_name)`` op before every
``unified_mla_attention_with_output`` call and threads its output back as the
``q_prepped`` kwarg. The downstream attention impl then skips its internal
BMM+cat+(FP8)quant block and consumes ``q_prepped`` directly.

This pass is the prerequisite for ``MLAAiterQkRopeKVCacheFusionPass``, which
matches the pair ``(fused_rope_unified_mla_kv_cache_update, mla_decode_q_prep)``
and replaces them with AITER's ``fused_qk_rope_concat_and_cache_mla``.

Why a pass and not a model-code call site? See the design doc / risks section
of the plan: gating the lift via ``is_applicable_for_range`` is what
guarantees memory boundedness (the op never enters prefill / large-batch
graphs) AND CUDA-graph-capture safety (``q.size(0)`` is a backed SymInt
exactly equal to the compile bucket size, so the fake_impl shape
``[q.size(0), num_heads, kv_lora + pe]`` is honest and statically planable
by Inductor). Inserting the op from model code would force it into prefill
graphs as well, where ``q.size(0)`` could be 8192+, blowing memory; or, if
the fake declared an unbacked SymInt, would break ``cudagraph_mode=FULL``
capture entirely.
"""
from __future__ import annotations

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.fx_utils import is_func
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class MLADecodeQPrepLiftPass(VllmInductorPass):
    """Lift ``mla_decode_q_prep`` above ``unified_mla_attention_with_output``.

    Decode-bucket-only via :py:meth:`is_applicable_for_range`. When applicable,
    rewrites every ``unified_mla_attention_with_output(q, ..., layer_name, ...)``
    call to::

        q_prepped = mla_decode_q_prep(q, layer_name)
        unified_mla_attention_with_output(q, ..., layer_name, ...,
                                          q_prepped=q_prepped)

    Idempotent: skips calls that already have a non-``None`` ``q_prepped`` kwarg.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        cc = config.compilation_config
        max_token_num = cc.pass_config.aiter_qk_rope_kvcache_fusion_max_token_num
        # ``None`` means the user enabled the fusion but VllmConfig._set_compile_ranges
        # didn't auto-derive the value — should never happen in practice because the
        # auto-derivation runs before passes are constructed. Fail loud rather than
        # silently disabling the gate (which would let the lift pass fire on prefill
        # ranges and re-introduce the very memory bound this knob exists to enforce).
        assert max_token_num is not None, (
            "aiter_qk_rope_kvcache_fusion_max_token_num is None at pass-build time; "
            "expected VllmConfig._set_compile_ranges to have auto-derived it."
        )
        self.max_token_num: int = max_token_num
        self.matched_count = 0

    @property
    def UMA_OP(self):  # noqa: N802 - kept as class-style constant name
        # Lazy lookup: the op is registered when mla_attention.py is imported,
        # which may happen after this module is imported (during pass_manager
        # construction).
        return torch.ops.vllm.unified_mla_attention_with_output.default

    @property
    def QPREP_OP(self):  # noqa: N802
        return torch.ops.vllm.mla_decode_q_prep.default

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        # Decode bucket only. See module docstring for why this is load-bearing.
        return compile_range.end <= self.max_token_num

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if not (current_platform.is_rocm() and rocm_aiter_ops.is_enabled()):
            return

        self.matched_count = 0
        for node in list(graph.nodes):
            if is_func(node, self.UMA_OP):
                self._lift_direct(graph, node)
            elif is_func(node, auto_functionalized) and node.args[0] == self.UMA_OP:
                self._lift_autofn(graph, node)

        logger.debug(
            "MLADecodeQPrepLiftPass: lifted q-prep above %d unified_mla_attention_with_output call(s)",
            self.matched_count,
        )

    def _existing_q_prepped(self, node: fx.Node) -> object:
        return node.kwargs.get("q_prepped", None)

    def _lift_direct(self, graph: fx.Graph, node: fx.Node) -> None:
        if self._existing_q_prepped(node) is not None:
            return
        q = node.kwargs.get("q")
        layer_name = node.kwargs.get("layer_name")
        if q is None or layer_name is None:
            # Op was called positionally, fall back to args.
            # Signature: (q, kv_c_normed, k_pe, output, layer_name, ...)
            if len(node.args) < 5:
                return
            q = node.args[0]
            layer_name = node.args[4]
        with graph.inserting_before(node):
            q_prepped = graph.call_function(
                self.QPREP_OP, args=(q, layer_name)
            )
        new_kwargs = dict(node.kwargs)
        new_kwargs["q_prepped"] = q_prepped
        node.kwargs = new_kwargs
        self.matched_count += 1

    def _lift_autofn(self, graph: fx.Graph, node: fx.Node) -> None:
        # auto_functionalized(uma_op, q=..., layer_name=..., ...)
        if self._existing_q_prepped(node) is not None:
            return
        kwargs = node.kwargs
        q = kwargs.get("q")
        layer_name = kwargs.get("layer_name")
        if q is None or layer_name is None:
            return
        with graph.inserting_before(node):
            q_prepped = graph.call_function(
                self.QPREP_OP, args=(q, layer_name)
            )
        new_kwargs = dict(kwargs)
        new_kwargs["q_prepped"] = q_prepped
        node.kwargs = new_kwargs
        self.matched_count += 1

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self)
