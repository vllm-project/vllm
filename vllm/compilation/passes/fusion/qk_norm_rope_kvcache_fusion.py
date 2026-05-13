# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRMSNorm, MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Custom op: Triton-based fused QKV split + QK-norm + RoPE + KV cache update
# ---------------------------------------------------------------------------


def fused_triton_qk_norm_rope_kvcache_update_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    attn_output_gate: bool,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)

    T = qkv.shape[0]
    dummy = torch.empty(0, device=qkv.device, dtype=qkv.dtype)

    if layer_slot_mapping is None:
        q = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        k = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        v = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        gate = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        return dummy, q, k, v, gate

    rdh = cos_sin_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.unbind(0)

    k_scale_f = getattr(attn_layer, "_k_scale_float", 1.0)
    v_scale_f = getattr(attn_layer, "_v_scale_float", 1.0)
    k_scale_t = (
        None
        if k_scale_f == 1.0
        else torch.tensor(k_scale_f, dtype=torch.float32, device=qkv.device)
    )
    v_scale_t = (
        None
        if v_scale_f == 1.0
        else torch.tensor(v_scale_f, dtype=torch.float32, device=qkv.device)
    )

    kv_layout = "HND" if key_cache.shape[1] == num_kv_heads else "NHD"

    q, gate, k, v = rocm_aiter_ops.triton_qk_norm_rope_kvcache(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos=cos_sin_cache[:, :rdh],
        sin=cos_sin_cache[:, rdh:],
        positions=positions,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=layer_slot_mapping,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_neox=is_neox,
        attn_output_gate=attn_output_gate,
        rms_norm_eps=rms_norm_eps,
        k_scale=k_scale_t,
        v_scale=v_scale_t,
        kv_cache_layout=kv_layout,
    )
    return dummy, q, k, v, gate


def fused_triton_qk_norm_rope_kvcache_update_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    attn_output_gate: bool,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_name: LayerNameType,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = qkv.shape[0]
    dummy = torch.empty(0, device=qkv.device, dtype=qkv.dtype)
    q = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    v = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    gate = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
    return dummy, q, k, v, gate


direct_register_custom_op(
    op_name="fused_triton_qk_norm_rope_kvcache_update",
    op_func=fused_triton_qk_norm_rope_kvcache_update_impl,
    mutates_args=[],
    fake_impl=fused_triton_qk_norm_rope_kvcache_update_fake,
)


# ---------------------------------------------------------------------------
# Pattern: Qwen3Next QK-norm + RoPE + unified_kv_cache_update (with gate)
# ---------------------------------------------------------------------------


class Qwen3NextQkNormRopeKvCachePattern:
    """Extend ``QkNormRopeKvCachePattern`` for the Qwen3Next attention layout.

    When *attn_output_gate* is True the QKV projection emits
    ``[q||gate, k, v]`` where the q portion is doubled.  The pattern
    matches the gate extraction (view -> chunk -> clone) before the
    standard QK-norm + RoPE + KV-cache sequence.  The replacement
    delegates to the Triton-based ``fused_qkv_split_qk_norm_rope_cache``
    kernel which handles the split, norm, RoPE, gate extraction, and KV
    cache update in a single launch.

    Qwen3Next uses GemmaRMSNorm which applies ``(1 + weight)`` before the
    norm.  The pattern includes the ``.float() + 1.0`` nodes so they are
    consumed; the fused op receives the raw weight and applies the
    adjustment internally.
    """

    FUSED_OP = torch.ops.vllm.fused_triton_qk_norm_rope_kvcache_update.default

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
        attn_output_gate: bool,
        rope_flashinfer: bool = False,
        match_rocm_aiter_rope: bool = False,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer
        self.attn_output_gate = attn_output_gate

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rmsnorm_matcher = MatcherRMSNorm(eps)
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=rope_flashinfer,
            match_rocm_aiter=match_rocm_aiter_rope,
        )

    def get_inputs(self) -> list:
        T = 5
        L = 4096
        q_portion = self.q_size * 2 if self.attn_output_gate else self.q_size
        qkv = empty_bf16(T, q_portion + self.k_size + self.v_size)
        positions = empty_i64(T)
        q_weight = empty_bf16(self.head_size)
        k_weight = empty_bf16(self.head_size)
        if self.rope_flashinfer:
            cos_sin_cache = empty_fp32(L, self.head_size)
        else:
            cos_sin_cache = empty_bf16(L, self.head_size)
        inputs: list = [qkv, positions, q_weight, k_weight, cos_sin_cache]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self.layer_name))
        return inputs

    def _mk_pattern_with_layer_name_input(self):
        """Pattern/replacement with layer_name as an explicit graph input.

        Used when ``_USE_LAYERNAME`` is True (torch >= 2.11): layer names are
        hoisted as opaque graph inputs so a single pattern matches every
        attention layer without per-layer registration.
        """
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_size
        head_dim_v = self.head_size_v
        q_size = self.q_size
        k_size = self.k_size
        v_size = self.v_size
        eps = self.eps
        is_neox = self.is_neox
        attn_output_gate = self.attn_output_gate

        rmsnorm_matcher = self.rmsnorm_matcher
        rope_matcher = self.rope_matcher
        FUSED_OP = self.FUSED_OP

        if attn_output_gate:

            def pattern(qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name):
                q_gate, k, v = qkv.split([q_size * 2, k_size, v_size], dim=-1)

                q_gate_3d = q_gate.view(-1, num_heads, 2 * head_dim)
                q_3d, gate_3d = q_gate_3d.chunk(2, dim=-1)

                q_3d = q_3d.contiguous()
                q_w = q_weight.float() + 1.0
                q_normed = rmsnorm_matcher(q_3d, q_w)
                q_normed_flat = q_normed.view(-1, q_size)

                k_3d = k.view(-1, num_kv_heads, head_dim)
                k_w = k_weight.float() + 1.0
                k_normed = rmsnorm_matcher(k_3d, k_w)
                k_flat = k_normed.view(-1, k_size)

                q_rope, k_rope = rope_matcher(
                    positions, q_normed_flat, k_flat, cos_sin_cache
                )

                q_rope = q_rope.view(-1, num_heads, head_dim)
                k_rope = k_rope.view(-1, num_kv_heads, head_dim)
                v = v.view(-1, num_kv_heads, head_dim_v)
                dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, layer_name)
                return dummy, q_rope, k_rope, v, gate_3d

            def replacement(
                qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
            ):
                results = FUSED_OP(
                    qkv=qkv,
                    positions=positions,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=is_neox,
                    attn_output_gate=True,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    layer_name=layer_name,
                )

                _, _, v = qkv.split([q_size * 2, k_size, v_size], dim=-1)
                v = v.view(qkv.shape[0], num_kv_heads, head_dim_v)

                return results[0], results[1], results[2], v, results[4]
        else:

            def pattern(  # type: ignore[misc]
                qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
            ):
                q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

                q_by_head = q.view(-1, q_size // head_dim, head_dim)
                q_w = q_weight.float() + 1.0
                q_normed = rmsnorm_matcher(q_by_head, q_w)
                q_flat = q_normed.view(-1, q_size)

                k_by_head = k.view(-1, k_size // head_dim, head_dim)
                k_w = k_weight.float() + 1.0
                k_normed = rmsnorm_matcher(k_by_head, k_w)
                k_flat = k_normed.view(-1, k_size)

                q_rope, k_rope = rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

                q_rope = q_rope.view(-1, num_heads, head_dim)
                k_rope = k_rope.view(-1, num_kv_heads, head_dim)
                v = v.view(-1, num_kv_heads, head_dim_v)
                dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, layer_name)
                return dummy, q_rope, k_rope, v

            def replacement(  # type: ignore[misc]
                qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
            ):
                results = FUSED_OP(
                    qkv=qkv,
                    positions=positions,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=is_neox,
                    attn_output_gate=False,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    layer_name=layer_name,
                )

                _, _, v = qkv.split([q_size, k_size, v_size], dim=-1)
                v = v.view(qkv.shape[0], num_kv_heads, head_dim_v)

                return results[0], results[1], results[2], v

        return pattern, replacement

    def _mk_pattern_with_layer_name_closure(self, _ln):
        """Pattern/replacement with layer_name as a closure constant.

        Used when ``_USE_LAYERNAME`` is False (torch < 2.11 or
        ``VLLM_USE_LAYERNAME=0``): each layer's pattern bakes its own
        ``_ln`` so per-layer registration is required.
        """
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_size
        head_dim_v = self.head_size_v
        q_size = self.q_size
        k_size = self.k_size
        v_size = self.v_size
        eps = self.eps
        is_neox = self.is_neox
        attn_output_gate = self.attn_output_gate

        rmsnorm_matcher = self.rmsnorm_matcher
        rope_matcher = self.rope_matcher
        FUSED_OP = self.FUSED_OP

        if attn_output_gate:

            def pattern(qkv, positions, q_weight, k_weight, cos_sin_cache):
                q_gate, k, v = qkv.split([q_size * 2, k_size, v_size], dim=-1)

                q_gate_3d = q_gate.view(-1, num_heads, 2 * head_dim)
                q_3d, gate_3d = q_gate_3d.chunk(2, dim=-1)

                q_3d = q_3d.contiguous()
                q_w = q_weight.float() + 1.0
                q_normed = rmsnorm_matcher(q_3d, q_w)
                q_normed_flat = q_normed.view(-1, q_size)

                k_3d = k.view(-1, num_kv_heads, head_dim)
                k_w = k_weight.float() + 1.0
                k_normed = rmsnorm_matcher(k_3d, k_w)
                k_flat = k_normed.view(-1, k_size)

                q_rope, k_rope = rope_matcher(
                    positions, q_normed_flat, k_flat, cos_sin_cache
                )

                q_rope = q_rope.view(-1, num_heads, head_dim)
                k_rope = k_rope.view(-1, num_kv_heads, head_dim)
                v = v.view(-1, num_kv_heads, head_dim_v)
                dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, _ln)
                return dummy, q_rope, k_rope, v, gate_3d

            def replacement(qkv, positions, q_weight, k_weight, cos_sin_cache):
                results = FUSED_OP(
                    qkv=qkv,
                    positions=positions,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=is_neox,
                    attn_output_gate=True,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    layer_name=_ln,
                )

                _, _, v = qkv.split([q_size * 2, k_size, v_size], dim=-1)
                v = v.view(qkv.shape[0], num_kv_heads, head_dim_v)

                return results[0], results[1], results[2], v, results[4]
        else:

            def pattern(qkv, positions, q_weight, k_weight, cos_sin_cache):  # type: ignore[misc]
                q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

                q_by_head = q.view(-1, q_size // head_dim, head_dim)
                q_w = q_weight.float() + 1.0
                q_normed = rmsnorm_matcher(q_by_head, q_w)
                q_flat = q_normed.view(-1, q_size)

                k_by_head = k.view(-1, k_size // head_dim, head_dim)
                k_w = k_weight.float() + 1.0
                k_normed = rmsnorm_matcher(k_by_head, k_w)
                k_flat = k_normed.view(-1, k_size)

                q_rope, k_rope = rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

                q_rope = q_rope.view(-1, num_heads, head_dim)
                k_rope = k_rope.view(-1, num_kv_heads, head_dim)
                v = v.view(-1, num_kv_heads, head_dim_v)
                dummy = torch.ops.vllm.unified_kv_cache_update(k_rope, v, _ln)
                return dummy, q_rope, k_rope, v

            def replacement(qkv, positions, q_weight, k_weight, cos_sin_cache):  # type: ignore[misc]
                results = FUSED_OP(
                    qkv=qkv,
                    positions=positions,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin_cache,
                    is_neox=is_neox,
                    attn_output_gate=False,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    layer_name=_ln,
                )

                _, _, v = qkv.split([q_size, k_size, v_size], dim=-1)
                v = v.view(qkv.shape[0], num_kv_heads, head_dim_v)

                return results[0], results[1], results[2], v

        return pattern, replacement

    def register(self, pm_pass: PatternMatcherPass) -> None:
        _ln = _encode_layer_name(self.layer_name)

        if _USE_LAYERNAME:
            pattern, replacement = self._mk_pattern_with_layer_name_input()
        else:
            pattern, replacement = self._mk_pattern_with_layer_name_closure(_ln)

        def fwd_and_view_to_reshape(*args, **kwargs) -> fx.GraphModule:
            gm = pm.fwd_only(*args, **kwargs)
            view_to_reshape(gm)
            return gm

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            fwd_and_view_to_reshape,
            pm_pass,
        )


# ---------------------------------------------------------------------------
# Pass class
# ---------------------------------------------------------------------------


class QkNormRopeKvCacheFusionPass(VllmPatternMatcherPass):
    """
    Fuse QK-norm + RoPE + KV cache update into a single AITER Triton kernel.

    Registers Qwen3NextQkNormRopeKvCachePattern for attention layers on
    ROCm with AITER enabled. The Triton kernel handles QKV split, QK-norm,
    RoPE, gate extraction, and KV cache update in a single launch.
    """

    def _register_variants(
        self,
        pattern_cls: type,
        layer: Attention,
        match_rocm_aiter_rope: bool,
        **extra_kwargs,
    ) -> None:
        # The pattern's traced ``eps`` literal is collapsed to ``Ignored()`` by
        # ``torch._inductor.pattern_matcher.fx_to_pattern`` (it elides Python
        # float constants on match), so a single registration matches every
        # eps value at apply time. Iterating eps here only produces duplicate
        # patterns that the matcher rejects. Only ``is_neox`` is iterated --
        # it parameterizes the actual rotary op used in the traced graph
        # (different ops for is_neox True/False), so patterns are distinct.
        for neox in [True, False]:
            pattern_cls(
                layer=layer,
                eps=1e-6,
                is_neox=neox,
                match_rocm_aiter_rope=match_rocm_aiter_rope,
                **extra_kwargs,
            ).register(self.patterns)

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="qk_norm_rope_kvcache_fusion_pass"
        )

        cc = config.compilation_config
        self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once(
                "QK Norm+RoPE+KVCache fusion not enabled: unsupported dtype %s", dtype
            )
            return

        if not rocm_aiter_ops.is_enabled():
            logger.warning_once(
                "QK Norm+RoPE+KVCache fusion not enabled: AITER not available"
            )
            return

        attn_layers = get_layers_from_vllm_config(config, Attention)

        rope_custom_enabled = cc.is_custom_op_enabled("rotary_embedding")
        rms_custom_enabled = cc.is_custom_op_enabled("rms_norm")
        logger.debug(
            "QkNormRopeKvCacheFusionPass init: "
            "RotaryEmbedding.enabled()=%s, rope_custom_enabled=%s, "
            "RMSNorm custom_op_enabled=%s",
            RotaryEmbedding.enabled(),
            rope_custom_enabled,
            rms_custom_enabled,
        )

        # Register a pattern only for the rope op that will actually appear
        # in the traced model graph at runtime. ``MatcherRotaryEmbedding``
        # resolves the same way as the live model layer, so iterating
        # match_rocm_aiter_rope ∈ {True, False} would generate two
        # registrations that resolve to the same rope op under
        # ``VLLM_ROCM_USE_AITER_TRITON_ROPE=1`` and trip
        # ``check_and_add_duplicate_pattern``.
        match_rocm_aiter_rope = rocm_aiter_ops.is_triton_rotary_embed_enabled()

        # When _USE_LAYERNAME is enabled, layer_name is hoisted as an
        # opaque graph input so every layer produces the same pattern --
        # register once per shape and break out of the per-layer loop.
        for _, layer in attn_layers.items():
            for gate in [True, False]:
                self._register_variants(
                    Qwen3NextQkNormRopeKvCachePattern,
                    layer,
                    match_rocm_aiter_rope,
                    attn_output_gate=gate,
                )
            if _USE_LAYERNAME:
                break

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        # ``fx_to_pattern`` is called twice in torch's pattern-matcher lifecycle:
        # once at registration (with a hard-coded
        # ``ignore_types=(int, float, list, torch.device, torch.dtype)``) so the
        # registered patterns have ``Ignored()`` in int-typed slots (reshape
        # dims, split sizes, head_size, etc.), and once at apply time with the
        # default empty ``ignore_types`` (``pattern_matcher.py:1548``). Without
        # the wrapper below, the apply-time fingerprint preserves the model
        # graph's concrete ``int``/``SymInt`` values and never matches the
        # ``Ignored()`` slots in the registered patterns -- yielding 0 matches.
        _orig_fx_to_pat = pm.fx_to_pattern

        def _relaxed_fx_to_pattern(*a, **kw):
            kw["ignore_types"] = (int, torch.SymInt)
            return _orig_fx_to_pat(*a, **kw)

        pm.fx_to_pattern = _relaxed_fx_to_pattern
        try:
            self.matched_count = self.patterns.apply(graph)
        finally:
            pm.fx_to_pattern = _orig_fx_to_pat

        logger.info(
            "QK-Norm+RoPE+KVCache fusion: replaced %s pattern(s) "
            "with AITER fused_qk_norm_rope_cache_pts_quant_shuffle",
            self.matched_count,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, Qwen3NextQkNormRopeKvCachePattern)
