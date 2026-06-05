# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm import ir
from vllm._aiter_ops import check_aiter_fused_qk_norm_rope_cache, rocm_aiter_ops
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

logger = init_logger(__name__)

# Pattern body / replacement body return tuples of varying arity (4 or 5
# tensors depending on ``attn_output_gate``); typed as ``tuple[Tensor, ...]``.
_PatternFn = Callable[..., tuple[torch.Tensor, ...]]


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
        # Profiling / dummy-run path: there is no real KV cache to write into,
        # so we return uninitialized output tensors that just satisfy the
        # downstream graph's shape/dtype expectations. The values are never
        # read (callers gate on profile mode upstream); ``empty`` is used
        # over ``zeros`` to avoid a needless device kernel during profiling.
        # ``gate`` is always materialized -- even when ``attn_output_gate``
        # is False the replacement returns it as a 5-tuple element which
        # downstream consumers ignore -- to keep the kernel signature
        # invariant across ``Qwen3NextQkNormRopeKvCachePattern`` variants.
        q = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        k = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        v = torch.empty(T, num_kv_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        gate = torch.empty(T, num_heads, head_dim, device=qkv.device, dtype=qkv.dtype)
        return dummy, q, k, v, gate

    rdh = cos_sin_cache.shape[-1] // 2
    # Delegate to the attention backend's KV split: ROCM_AITER_UNIFIED_ATTN
    # places K/V on dim 1 ((num_blocks, 2, ...)), older backends on dim 0.
    # `_split_kv_cache` is the canonical accessor and is also used by the
    # sibling rope_kvcache_fusion pass via attn_layer.impl.
    key_cache, value_cache = attn_layer.impl._split_kv_cache(kv_cache)  # type: ignore[attr-defined]

    # Mirror the unfused FP8-KV path (rocm_aiter_unified_attn.do_kv_cache_update
    # via reshape_and_cache_flash): when --kv-cache-dtype fp8 is in effect, the
    # KV cache storage dtype is torch.uint8 (raw bytes), and the AITER triton
    # kernel's `tl.store(... k.to(key_cache_ptr.dtype.element_ty))` would
    # truncate bf16 K/V values to uint8 instead of fp8e4m3 -- destroying KV.
    # View-cast to the platform fp8 dtype so the kernel's element_ty is fp8.
    kv_cache_dtype = getattr(attn_layer, "kv_cache_dtype", "auto")
    if kv_cache_dtype != "auto" and key_cache.dtype == torch.uint8:
        fp8_dtype = current_platform.fp8_dtype()
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)

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

    # This fusion pass is gated to ROCm + AITER + Qwen3-Next, where
    # ``RocmAiterUnifiedAttentionBackend.get_kv_cache_shape`` returns
    # ``(num_blocks, 2, block_size, num_kv_heads, head_size)``. After
    # ``_split_kv_cache`` (split on dim 1), each of ``key_cache`` and
    # ``value_cache`` has shape
    # ``(num_blocks, block_size, num_kv_heads, head_size)`` -- i.e. NHD.
    # Assert the invariant rather than inferring layout from a shape
    # coincidence (the previous heuristic ``shape[1] == num_kv_heads``
    # silently picks the wrong layout if a backend's ``block_size``
    # happens to equal ``num_kv_heads``). If a future backend with HND
    # storage is added to the gate, this will fail loud at first
    # invocation and the layout-determination needs to be revisited
    # (e.g. via a ``kv_cache_layout`` accessor on the attention impl).
    assert key_cache.shape[-2] == num_kv_heads, (
        f"QK-Norm+RoPE+KVCache fusion expects NHD KV layout "
        f"(.., num_kv_heads, head_size); got shape "
        f"{tuple(key_cache.shape)} with num_kv_heads={num_kv_heads} "
        f"for layer {layer_name}."
    )
    kv_layout = "NHD"

    # Qwen3-Next uses GemmaRMSNorm which applies ``(1 + weight)`` to the
    # weight before the norm. The traced pattern matches the ``.float() +
    # 1.0`` op so it is consumed by the match; the AITER kernel applies
    # the same ``(1.0 + weight)`` adjustment internally, so we pass the
    # raw weight here.

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
    """Match QK-norm + RoPE + KV-cache-update and replace with a fused kernel.

    When ``attn_output_gate`` is True the QKV projection emits
    ``[q || gate, k, v]`` (q portion doubled); the pattern matches the gate
    extraction (view -> chunk -> contiguous) before the QK-norm + RoPE + KV
    cache sequence. The replacement delegates to the Triton-based
    ``fused_qkv_split_qk_norm_rope_cache`` kernel, which handles split,
    norm, RoPE, gate extraction, and KV-cache update in a single launch.

    Qwen3-Next uses ``GemmaRMSNorm`` which applies ``(1 + weight)`` to the
    weight before the norm. The pattern includes the ``.float() + 1.0`` op
    so it is consumed by the match; the fused op receives the raw weight
    and applies the adjustment internally.
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
        # The fused AITER kernel takes a single ``head_dim`` argument and
        # uses it for both K and V (see ``triton_qk_norm_rope_kvcache`` in
        # ``vllm/_aiter_ops.py`` and the kernel's ``head_dim`` parameter).
        # If a future model added to ``_QK_NORM_MODEL_TYPES`` has an
        # asymmetric V head dim, the kernel would split V using the K
        # head_dim and silently corrupt the V cache. Fail loud at
        # registration so the gate has to be revisited rather than
        # producing numerically-silent wrong KV writes.
        assert self.head_size_v == self.head_size, (
            f"fused_triton_qk_norm_rope_kvcache_update assumes "
            f"head_size_v == head_size; got {self.head_size_v} != "
            f"{self.head_size} for layer {self.layer_name}. The AITER "
            f"kernel has no separate head_dim_v argument."
        )
        # The traced pattern (``MatcherRotaryEmbedding`` + the AITER fused
        # kernel below) assumes ``rotary_dim == head_size`` -- i.e. fully
        # rotary. A partial-rotary layer (``rotary_dim < head_size``, e.g.
        # GLM-style models with rotary applied only to the first half of
        # the head) would produce a different rope op that does not match
        # the registered pattern and would also be miscomputed by the
        # fused kernel (it has no ``rotary_dim`` parameter). Fail loud at
        # registration so a future model added to ``_QK_NORM_MODEL_TYPES``
        # is forced to either provide a partial-rotary variant or stay off
        # this fusion.
        rotary_emb = getattr(layer, "rotary_emb", None)
        rotary_dim = getattr(rotary_emb, "rotary_dim", self.head_size)
        assert rotary_dim == self.head_size, (
            f"fused_triton_qk_norm_rope_kvcache_update assumes fully-rotary "
            f"layers (rotary_dim == head_size); got rotary_dim={rotary_dim} "
            f"!= head_size={self.head_size} for layer {self.layer_name}. The "
            f"AITER kernel has no rotary_dim argument."
        )
        self.eps = eps
        self.is_neox = is_neox
        self.rope_flashinfer = rope_flashinfer
        self.attn_output_gate = attn_output_gate

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_flashinfer=rope_flashinfer,
            match_rocm_aiter=match_rocm_aiter_rope,
        )

    def get_inputs(self) -> list[torch.Tensor]:
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
        inputs: list[torch.Tensor] = [qkv, positions, q_weight, k_weight, cos_sin_cache]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self.layer_name))
        return inputs

    # ------------------------------------------------------------------
    # Shared pattern / replacement bodies
    # ------------------------------------------------------------------
    #
    # ``layer_name`` is the only thing that differs between the two
    # registration paths below (``_USE_LAYERNAME`` True vs False); keeping
    # the q/k/v ops here in a single helper ensures the two paths cannot
    # silently diverge.

    def _make_pattern_body(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, ...]:
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_size
        head_dim_v = self.head_size_v
        q_size = self.q_size
        k_size = self.k_size
        v_size = self.v_size

        if self.attn_output_gate:
            q_gate, k, v = qkv.split([q_size * 2, k_size, v_size], dim=-1)

            q_gate_3d = q_gate.view(-1, num_heads, 2 * head_dim)
            q_3d, gate_3d = q_gate_3d.chunk(2, dim=-1)

            # ``chunk`` produces a non-contiguous view; the model emits a
            # ``clone(memory_format=contiguous_format)`` here. Use
            # ``contiguous`` so the trace records the same clone op.
            #
            # The model uses ``GemmaRMSNorm`` whose forward path on the live
            # graph is ``chunk -> reshape(2D) -> norm -> view(3D)``, but
            # ``view_to_reshape`` (applied at registration via
            # ``fwd_and_view_to_reshape``) and the inductor's reshape
            # canonicalization rewrite both legs to the equivalent
            # ``view``-of-3D form recorded here. Calling ``ir.ops.rms_norm``
            # directly on the 3D tensor (no explicit ``reshape`` round-trip,
            # no ``.float()/.bf16()`` cast pair) matches that canonical form
            # for both NeoX and AITER rope variants.
            q_3d = q_3d.contiguous()
            q_w = q_weight.float() + 1.0
            q_normed = ir.ops.rms_norm(q_3d, q_w, self.eps)
            q_normed_flat = q_normed.view(-1, q_size)

            k_3d = k.view(-1, num_kv_heads, head_dim)
            k_w = k_weight.float() + 1.0
            k_normed = ir.ops.rms_norm(k_3d, k_w, self.eps)
            k_flat = k_normed.view(-1, k_size)

            q_rope, k_rope = self.rope_matcher(
                positions, q_normed_flat, k_flat, cos_sin_cache
            )

            k_rope_3d = k_rope.view(-1, num_kv_heads, head_dim)
            v_3d = v.view(-1, num_kv_heads, head_dim_v)
            dummy = torch.ops.vllm.unified_kv_cache_update(k_rope_3d, v_3d, layer_name)
            return dummy, q_rope, k_rope_3d, v_3d, gate_3d

        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)

        q_by_head = q.view(-1, q_size // head_dim, head_dim)
        q_w = q_weight.float() + 1.0
        q_normed = ir.ops.rms_norm(q_by_head, q_w, self.eps)
        q_flat = q_normed.view(-1, q_size)

        k_by_head = k.view(-1, k_size // head_dim, head_dim)
        k_w = k_weight.float() + 1.0
        k_normed = ir.ops.rms_norm(k_by_head, k_w, self.eps)
        k_flat = k_normed.view(-1, k_size)

        q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

        k_rope_3d = k_rope.view(-1, num_kv_heads, head_dim)
        v_3d = v.view(-1, num_kv_heads, head_dim_v)
        dummy = torch.ops.vllm.unified_kv_cache_update(k_rope_3d, v_3d, layer_name)
        return dummy, q_rope, k_rope_3d, v_3d

    def _make_replacement_body(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, ...]:
        results = self.FUSED_OP(
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            rms_norm_eps=self.eps,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
            attn_output_gate=self.attn_output_gate,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            layer_name=layer_name,
        )

        if self.attn_output_gate:
            split_sizes = [self.q_size * 2, self.k_size, self.v_size]
        else:
            split_sizes = [self.q_size, self.k_size, self.v_size]
        _, _, v = qkv.split(split_sizes, dim=-1)
        v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)

        # The pattern's q_rope is 2D (post-rope, pre-reshape) so the
        # replacement must return q in the same 2D shape; the fused op
        # produces (T, num_heads, head_dim) so view it back to (T, q_size).
        q_rope_2d = results[1].view(qkv.shape[0], self.q_size)

        if self.attn_output_gate:
            return results[0], q_rope_2d, results[2], v, results[4]
        return results[0], q_rope_2d, results[2], v

    # ------------------------------------------------------------------
    # Pattern / replacement variants for the two ``_USE_LAYERNAME`` modes
    # ------------------------------------------------------------------

    def _mk_pattern_with_layer_name_input(
        self,
    ) -> tuple[_PatternFn, _PatternFn]:
        """Pattern/replacement with layer_name as an explicit graph input.

        Used when ``_USE_LAYERNAME`` is True (torch >= 2.11): layer names
        are hoisted as opaque graph inputs so a single pattern matches
        every attention layer without per-layer registration.
        """

        def pattern(qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name):
            return self._make_pattern_body(
                qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
            )

        def replacement(qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name):
            return self._make_replacement_body(
                qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
            )

        return pattern, replacement

    def _mk_pattern_with_layer_name_closure(
        self, _ln: LayerNameType
    ) -> tuple[_PatternFn, _PatternFn]:
        """Pattern/replacement with layer_name as a closure constant.

        Used when ``_USE_LAYERNAME`` is False (torch < 2.11 or
        ``VLLM_USE_LAYERNAME=0``): each layer's pattern bakes its own
        ``_ln`` so per-layer registration is required.
        """

        def pattern(qkv, positions, q_weight, k_weight, cos_sin_cache):
            return self._make_pattern_body(
                qkv, positions, q_weight, k_weight, cos_sin_cache, _ln
            )

        def replacement(qkv, positions, q_weight, k_weight, cos_sin_cache):
            return self._make_replacement_body(
                qkv, positions, q_weight, k_weight, cos_sin_cache, _ln
            )

        return pattern, replacement

    def register(self, pm_pass: PatternMatcherPass) -> None:
        if _USE_LAYERNAME:
            pattern, replacement = self._mk_pattern_with_layer_name_input()
        else:
            _ln = _encode_layer_name(self.layer_name)
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

    Registers ``Qwen3NextQkNormRopeKvCachePattern`` for attention layers on
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
        # The pattern's traced ``eps`` literal is collapsed to ``Ignored()``
        # by ``torch._inductor.pattern_matcher.fx_to_pattern`` (it elides
        # Python float constants on match), so a single registration matches
        # every eps value at apply time. Iterating eps here only produces
        # duplicate patterns that the matcher rejects. ``is_neox`` is the
        # only axis that changes the rotary op recorded in the traced graph
        # (different ops for is_neox True/False), so its registrations are
        # structurally distinct.
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
            # Most fp32 / int dtype configurations are intentional (e.g.
            # debug / accuracy investigations on small CPU runs), so this
            # is debug rather than warning -- the auto-enable callable in
            # ``enable_qk_norm_rope_kvcache`` already considers this and
            # will not flip the gate on for unsupported dtypes.
            logger.debug(
                "QK Norm+RoPE+KVCache fusion not enabled: unsupported dtype %s", dtype
            )
            return

        if not rocm_aiter_ops.is_enabled():
            logger.warning_once(
                "QK Norm+RoPE+KVCache fusion not enabled: AITER not available"
            )
            return

        # The config-level auto-enable in ``enable_qk_norm_rope_kvcache``
        # already calls ``check_aiter_fused_qk_norm_rope_cache``, but a user
        # who explicitly sets ``pass_config.fuse_qk_norm_rope_kvcache=True``
        # bypasses that callable -- without this guard we would register
        # patterns and only crash at first invocation when the fused kernel
        # is dispatched.
        if not check_aiter_fused_qk_norm_rope_cache():
            logger.warning_once(
                "QK Norm+RoPE+KVCache fusion not enabled: bundled AITER "
                "does not provide aiter.ops.triton.rope."
                "fused_qkv_split_qk_norm_rope_cache"
            )
            return

        attn_layers = get_layers_from_vllm_config(config, Attention)

        # Resolve the rope op once: ``MatcherRotaryEmbedding`` resolves
        # ``rotary_op`` the same way as the live model layer, so iterating
        # match_rocm_aiter_rope ∈ {True, False} would generate two
        # registrations that resolve to the same rope op under
        # ``VLLM_ROCM_USE_AITER_TRITON_ROPE=1`` and trip
        # ``check_and_add_duplicate_pattern``.
        match_rocm_aiter_rope = rocm_aiter_ops.is_triton_rotary_embed_enabled()

        # When ``_USE_LAYERNAME`` is enabled, layer_name is hoisted as an
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
        # ``torch._inductor.pattern_matcher.fx_to_pattern`` is invoked twice
        # in the matcher lifecycle: at ``register_replacement`` time it is
        # called with ``ignore_types=(int, float, list, torch.device,
        # torch.dtype)`` so int constants in the registered pattern (split
        # sizes, view dims, head_size, etc.) become ``Ignored()``. At apply
        # time (``pattern_matcher.py:1548`` in torch 2.10) it is called with
        # the default empty ``ignore_types``, which preserves the live FX
        # graph's concrete ``SymInt`` values. Without this wrapper the
        # apply-time fingerprint never matches the ``Ignored()`` slots in
        # the registered pattern -- yielding 0 matches for every layer.
        # Once vLLM moves to torch >= 2.11 this becomes unnecessary, but
        # the rest of this pass already supports both modes.
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
            "with AITER fused_qkv_split_qk_norm_rope_cache",
            self.matched_count,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, Qwen3NextQkNormRopeKvCachePattern)
