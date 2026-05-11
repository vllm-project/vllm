# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse decode-side MLA RoPE + KV-cache write + q-absorb BMM + q concat + FP8
quant into AITER's ``fused_qk_rope_concat_and_cache_mla`` kernel.

This pass runs after :py:class:`MLARoPEKVCacheCatFusionPass` (PR #40392) and
:py:class:`MLADecodeQPrepLiftPass`. It matches the pair

* ``auto_functionalized(fused_rope_unified_mla_kv_cache_update, ...)``
* ``mla_decode_q_prep(q, layer_name)``

(with matching ``layer_name``) and rewrites them into a single
``fused_aiter_qk_rope_kvcache_q_concat_quant_mla`` call wrapping AITER's
fused kernel.

Decode-only via :py:meth:`is_applicable_for_range`, gated to ROCm + AITER.
"""
from __future__ import annotations

import operator

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.fx_utils import is_func
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)

_FP8_DTYPE = current_platform.fp8_dtype() if hasattr(current_platform, "fp8_dtype") else torch.float8_e4m3fn


def _q_out_dtype(layer) -> torch.dtype:
    """Match mla_decode_q_prep_fake's dtype contract."""
    fp8_attention = is_quantized_kv_cache(layer.kv_cache_dtype)
    if fp8_attention and layer.impl.supports_quant_query_input:
        return _FP8_DTYPE
    return torch.bfloat16  # fallback; in practice the AITER fusion is gated on FP8


def fused_aiter_qk_rope_kvcache_q_concat_quant_mla_impl(
    q: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Wraps AITER's ``fused_qk_rope_concat_and_cache_mla``.

    Mutates ``q_pe`` and ``k_pe`` (in-place RoPE), writes ``kv_cache`` (cat of
    ``kv_c`` and rotated ``k_pe`` quantized to FP8), and returns
    ``q_prepped`` of shape ``[q.size(0), num_heads, kv_lora + qk_rope]`` (FP8
    when the layer is FP8-attention).
    """
    import aiter

    layer_name = _resolve_layer_name(layer_name)
    _, layer, kv_cache, slot_mapping = get_attention_context(layer_name)

    # vLLM stores FP8 KV cache as ``torch.uint8`` (see
    # ``STR_DTYPE_TO_TORCH_DTYPE``: ``"fp8" -> torch.uint8``). AITER's
    # ``fused_qk_rope_concat_and_cache_mla`` infers dtype from the tensor
    # itself and rejects uint8 with "kv cache data type is not supported"
    # (cache_kernels.cu line ~3494). vLLM's PR #40392 fusion side-steps this
    # because its underlying op (``_C_cache_ops.concat_and_cache_mla_rope_fused``)
    # takes an explicit ``kv_cache_dtype`` string, but the AITER kernel does
    # not. Reinterpret as the platform's FP8 dtype before dispatch.
    if is_quantized_kv_cache(kv_cache_dtype) and kv_cache.dtype == torch.uint8:
        kv_cache = kv_cache.view(_FP8_DTYPE)

    # AITER kernel expects split cos / sin caches of width rot_dim // 2.
    rot_dim = cos_sin_cache.shape[-1]
    cos_cache = cos_sin_cache[..., : rot_dim // 2].contiguous()
    sin_cache = cos_sin_cache[..., rot_dim // 2 :].contiguous()

    # Run the q-absorb BMM only (cat + quant happen inside the AITER kernel).
    q_nope_post_bmm, q_pe_raw = layer.do_decode_q_prep_inputs(q)
    # q_pe_raw is a slice of q from .split(); the AITER kernel receives the
    # already-passed-in q_pe (a separate tensor produced by RoPE input prep
    # in the model code) and rotates THAT in place. We pass q_pe (the kwarg
    # input) so that downstream code which observes the rotated q_pe sees it.
    # Note: q_pe_raw and q_pe should be the same data (both are q[..., nope:])
    # but as separate FX nodes — we use q_pe (the explicit RoPE input) so that
    # auto_functionalize's mutation tracking is correct.
    del q_pe_raw

    # k_pe arrives as [T, 1, qk_rope_head_dim]; AITER expects [T, qk_rope_head_dim]
    # or [T, num_kv_heads, qk_rope_head_dim]. Squeeze if needed.
    k_pe_for_aiter = k_pe.squeeze(1) if k_pe.dim() == 3 and k_pe.shape[1] == 1 else k_pe

    q_out = torch.empty(
        q.shape[0],
        layer.num_heads,
        layer.kv_lora_rank + layer.qk_rope_head_dim,
        dtype=_q_out_dtype(layer),
        device=q.device,
    )

    if slot_mapping is not None:
        aiter.fused_qk_rope_concat_and_cache_mla(
            q_nope_post_bmm,
            q_pe,
            kv_c,
            k_pe_for_aiter,
            kv_cache,
            q_out,
            slot_mapping.flatten(),
            k_scale,
            layer._q_scale,
            positions,
            cos_cache,
            sin_cache,
            is_neox,
            True,  # is_nope_first: vLLM q layout is [q_nope || q_pe]
        )

    return q_out


def fused_aiter_qk_rope_kvcache_q_concat_quant_mla_fake(
    q: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer_name = _resolve_layer_name(layer_name)
    from vllm.forward_context import get_forward_context

    layer = get_forward_context().no_compile_layers[layer_name]
    return torch.empty(
        q.size(0),
        layer.num_heads,
        layer.kv_lora_rank + layer.qk_rope_head_dim,
        dtype=_q_out_dtype(layer),
        device=q.device,
    )


direct_register_custom_op(
    op_name="fused_aiter_qk_rope_kvcache_q_concat_quant_mla",
    op_func=fused_aiter_qk_rope_kvcache_q_concat_quant_mla_impl,
    fake_impl=fused_aiter_qk_rope_kvcache_q_concat_quant_mla_fake,
    mutates_args=["q_pe", "k_pe"],
)


class MLAAiterQkRopeKVCacheFusionPass(VllmInductorPass):
    """Procedurally rewrites
    ``(auto_functionalized(fused_rope_unified_mla_kv_cache_update),
       mla_decode_q_prep)`` pairs (matched by ``layer_name``) into a single
    ``fused_aiter_qk_rope_kvcache_q_concat_quant_mla`` call.

    Why procedural and not a PatternMatcher pattern?
    - The pre-AITER graph mixes a HOP (``auto_functionalized``) with a regular
      call_function (``mla_decode_q_prep``); a procedural pass keeps the
      rewrite explicit and easy to debug, at the cost of being less
      declarative than the rest of the fusion-pass family.
    - Decode-bucket-only via :py:meth:`is_applicable_for_range`.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        cc = config.compilation_config
        max_token_num = cc.pass_config.aiter_qk_rope_kvcache_fusion_max_token_num
        # See ``MLADecodeQPrepLiftPass.__init__`` for the rationale.
        assert max_token_num is not None, (
            "aiter_qk_rope_kvcache_fusion_max_token_num is None at pass-build time; "
            "expected VllmConfig._set_compile_ranges to have auto-derived it."
        )
        self.max_token_num: int = max_token_num
        self.matched_count = 0

    @property
    def FRMKV_OP(self):  # noqa: N802
        return torch.ops.vllm.fused_rope_unified_mla_kv_cache_update.default

    @property
    def QPREP_OP(self):  # noqa: N802
        return torch.ops.vllm.mla_decode_q_prep.default

    @property
    def FUSED_AITER_OP(self):  # noqa: N802
        return torch.ops.vllm.fused_aiter_qk_rope_kvcache_q_concat_quant_mla.default

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if not (current_platform.is_rocm() and rocm_aiter_ops.is_enabled()):
            return

        self.matched_count = 0

        # Collect mla_decode_q_prep nodes indexed by layer_name (positional arg 1).
        qprep_by_layer: dict[object, fx.Node] = {}
        for node in graph.nodes:
            if is_func(node, self.QPREP_OP):
                ln = node.args[1] if len(node.args) >= 2 else node.kwargs.get("layer_name")
                qprep_by_layer[ln] = node

        # Iterate auto_functionalized fused_rope_unified_mla_kv_cache_update nodes,
        # match with mla_decode_q_prep by layer_name, and rewrite.
        for node in list(graph.nodes):
            if not (
                is_func(node, auto_functionalized) and node.args[0] == self.FRMKV_OP
            ):
                continue
            ln = node.kwargs.get("layer_name")
            qprep = qprep_by_layer.get(ln)
            if qprep is None:
                continue
            self._rewrite(graph, node, qprep)
            # Don't reuse the same qprep for multiple frmkv nodes.
            qprep_by_layer.pop(ln, None)

        logger.debug(
            "MLAAiterQkRopeKVCacheFusionPass: fused %d (RoPE+KVCache, q-prep) pair(s)",
            self.matched_count,
        )

    def _node_depends_on(self, node: object, target: fx.Node, depth: int = 64) -> bool:
        """Return True if ``node`` transitively depends on ``target`` (within
        ``depth`` hops). Used to detect ``slice_scatter`` ops whose scattered
        value comes from the RoPE+KVCache fused node we're replacing."""
        if not isinstance(node, fx.Node):
            return False
        if node is target:
            return True
        if depth <= 0:
            return False
        for inp in node.all_input_nodes:
            if self._node_depends_on(inp, target, depth - 1):
                return True
        return False

    def _unwrap_q_orig(self, q_node: fx.Node, frmkv_node: fx.Node) -> fx.Node:
        """Walk back through ``aten.slice_scatter`` ops whose scattered value
        depends on ``frmkv_node``.

        Why: PR #40392's fusion leaves the model's ``q[..., qk_nope:] = q_pe_rot``
        in-place write functionalized as a chain
        ``slice_scatter(q_orig, copy(slice_dst, getitem(frmkv, 1)), dim, start, end)``
        feeding ``unified_mla_attention_with_output(q=...)``. If we naively
        consumed that ``slice_scatter`` as the ``q`` input to our fused AITER
        node — which itself produces the post-RoPE ``q_pe`` consumed by the
        ``slice_scatter`` (transitively) — we'd introduce a cycle. AITER does
        the RoPE inside the kernel and only needs ``q_nope`` from ``q``, which
        is unaffected by the post-RoPE write, so it is always safe (and
        required for acyclicity) to use the pre-RoPE ``q_orig`` here.
        """
        cur = q_node
        aten_slice_scatter = torch.ops.aten.slice_scatter.default
        for _ in range(8):  # bounded; in practice depth == 1
            if not isinstance(cur, fx.Node):
                return cur
            if cur.op != "call_function" or cur.target is not aten_slice_scatter:
                return cur
            scattered = cur.args[1] if len(cur.args) >= 2 else cur.kwargs.get("src")
            if not isinstance(scattered, fx.Node):
                return cur
            if not self._node_depends_on(scattered, frmkv_node):
                return cur
            cur = cur.args[0]
        return cur

    def _rewrite(
        self, graph: fx.Graph, frmkv_node: fx.Node, qprep_node: fx.Node
    ) -> None:
        """Rewrite a single matched pair.

        ``frmkv_node`` is ``auto_functionalized(fused_rope_unified_mla_kv_cache_update,
        positions=, q_pe=, k_pe=, kv_c=, cos_sin_cache=, is_neox=,
        kv_cache_dtype=, kv_cache_scale=, layer_name=)`` whose outputs are
        accessed via ``getitem`` nodes:
        - ``[0]`` = dummy return value (used as ``kv_cache_dummy_dep``)
        - ``[1]`` = post-RoPE q_pe
        - ``[2]`` = post-RoPE k_pe (squeezed)

        ``qprep_node`` is ``mla_decode_q_prep(q, layer_name)`` returning
        ``q_prepped`` directly.

        Replacement: a single ``auto_functionalized(fused_aiter_qk_rope_kvcache_q_concat_quant_mla,
        q=, q_pe=, k_pe=, kv_c=, positions=, cos_sin_cache=, is_neox=,
        kv_cache_dtype=, k_scale=, layer_name=)`` whose outputs are
        - ``[0]`` = q_prepped (return value of the impl)
        - ``[1]`` = post-RoPE q_pe
        - ``[2]`` = post-RoPE k_pe
        """
        kw = frmkv_node.kwargs
        positions = kw["positions"]
        q_pe = kw["q_pe"]
        k_pe = kw["k_pe"]
        kv_c = kw["kv_c"]
        cos_sin_cache = kw["cos_sin_cache"]
        is_neox = kw["is_neox"]
        kv_cache_dtype = kw["kv_cache_dtype"]
        k_scale = kw["kv_cache_scale"]
        layer_name = kw["layer_name"]

        q_post = qprep_node.args[0] if qprep_node.args else qprep_node.kwargs["q"]
        # Walk back through any slice_scatter that depends on frmkv_node so the
        # new fused node consumes the *pre*-RoPE q_orig — see _unwrap_q_orig.
        q = self._unwrap_q_orig(q_post, frmkv_node)

        with graph.inserting_before(frmkv_node):
            new_node = graph.call_function(
                auto_functionalized,
                args=(self.FUSED_AITER_OP,),
                kwargs={
                    "q": q,
                    "q_pe": q_pe,
                    "k_pe": k_pe,
                    "kv_c": kv_c,
                    "positions": positions,
                    "cos_sin_cache": cos_sin_cache,
                    "is_neox": is_neox,
                    "kv_cache_dtype": kv_cache_dtype,
                    "k_scale": k_scale,
                    "layer_name": layer_name,
                },
            )
            new_q_prepped = graph.call_function(
                operator.getitem, args=(new_node, 0)
            )
            new_q_pe = graph.call_function(operator.getitem, args=(new_node, 1))
            new_k_pe = graph.call_function(operator.getitem, args=(new_node, 2))

        # Rewire users of frmkv_node's getitem(0/1/2) to new outputs.
        for user in list(frmkv_node.users):
            if not is_func(user, operator.getitem):
                continue
            idx = user.args[1]
            if idx == 0:
                # Dummy was used as kv_cache_dummy_dep. Route it through the
                # new q_prepped (also a Tensor) — preserves the data dep
                # for ordering without needing a separate dummy node.
                user.replace_all_uses_with(new_q_prepped)
            elif idx == 1:
                user.replace_all_uses_with(new_q_pe)
            elif idx == 2:
                user.replace_all_uses_with(new_k_pe)
            graph.erase_node(user)

        # Rewire users of qprep_node (the q_prepped Tensor) to new_q_prepped.
        qprep_node.replace_all_uses_with(new_q_prepped)

        # Now safe to erase the old nodes.
        graph.erase_node(qprep_node)
        graph.erase_node(frmkv_node)
        self.matched_count += 1

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self)
