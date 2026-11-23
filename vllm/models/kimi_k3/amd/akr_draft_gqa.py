# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The release DSpark draft's attention inputs (`DFlashQwen3Model`, 64-dim GQA heads)
on `attention/norm_rope_insert_common`:

* `DQKV`    `qkv_norm_rope_insert`: each layer's q / k RMSNorm + NeoX RoPE + the K / V
  insert of its query block in one launch, in place of the compiled layer's norms,
  RoPE and the backend's cache update. The layer's class becomes a subclass whose
  forward calls the op `vllm.kimi_akr_draft_qkv_norm_rope_insert` (opaque to
  torch.compile; it reads the cache and slots of the layer when it runs) and hands
  the attention q alone.
* `DCTXKV`  `kv_norm_rope_insert_multilayer`: the context rows' K norm + RoPE + the
  K / V inserts of every layer in one launch after the fused K / V GEMM of
  `precompute_and_store_context_kv` (stock: copy, grouped norm, repeat, RoPE and
  one insert a layer). Wraps the model instance's method, so it runs before
  `VLLM_K31_BEAT_CTXKV`'s class-level wrap; a call it declines goes there.

Both need TRITON_ATTN-layout caches ([blocks, KV heads, block, K | V]: TRITON_ATTN,
KIMI_K3_FLYDSL_SWA_GQA), bf16 or per-tensor e4m3, no V scale. A call an entry
declines (`*_unsupported_reason`, host checks) runs the stock ops, logged once.
"""

from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_nonlayer
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

logger = init_logger(__name__)


def _lib():
    from amd_kernels.attention import norm_rope_insert_common

    return norm_rope_insert_common


def _impl_reason(attn) -> str | None:
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    impl = attn.impl
    if not isinstance(impl, TritonAttentionImpl):
        return f"draft attention {type(impl).__name__}"
    if impl.attn_type != AttentionType.DECODER:
        return f"attention type {impl.attn_type}"
    if impl._is_per_token_head_quant or impl.kv_cache_dtype not in (
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ):
        return f"KV cache dtype {impl.kv_cache_dtype}"
    if attn.kv_sharing_target_layer_name is not None:
        return "a layer sharing another layer's KV cache"
    if attn.attn_backend.forward_includes_kv_cache_update:
        return f"{attn.attn_backend.get_name()} updates the cache in its forward"
    return None


def _layer_reason(a) -> str | None:
    """Why a draft attention layer (`DFlashQwen3Attention`) stays stock."""
    rope = a.rotary_emb
    reason = _lib().rope_unsupported_reason(
        rope.head_size, rope.rotary_dim, rope.is_neox_style, rope.cos_sin_cache
    )
    if reason is not None:
        return reason
    if getattr(a, "v_scale", None) is not None:
        return "a V scale"
    for norm in (a.q_norm, a.k_norm):
        w = norm.weight
        if w.dtype != torch.bfloat16 or w.numel() != a.head_dim:
            return f"q / k norm weight {tuple(w.shape)} {w.dtype}"
    if a.q_norm.variance_epsilon != a.k_norm.variance_epsilon:
        return "q and k norms with different eps"
    return _impl_reason(a.attn)


# ------------------------------------------------------------------ query block
# key -> the layer (`DFlashQwen3Attention`) the op serves
_LAYERS: dict[int, nn.Module] = {}
# (key, cache pointer) -> the entry's reason (shapes and layout do not change)
_WHY: dict[tuple[int, int], str | None] = {}


def _qkv_stock(a, attn, qkv, positions, kv_cache, slots) -> torch.Tensor:
    """The compiled layer's stock ops (norms, RoPE, the backend's cache update)."""
    n = qkv.shape[0]
    q, k, v = qkv.split([a.q_size, a.kv_size, a.kv_size], dim=-1)
    q = a.q_norm(q.reshape(n, a.num_heads, a.head_dim)).view(n, a.q_size)
    k = a.k_norm(k.reshape(n, a.num_kv_heads, a.head_dim)).view(n, a.kv_size)
    q, k = a.rotary_emb(positions, q, k)
    if slots is not None:
        attn.impl.do_kv_cache_update(
            attn,
            k.view(n, a.num_kv_heads, a.head_dim),
            v.view(n, a.num_kv_heads, a.head_dim),
            kv_cache,
            slots,
        )
    return q.contiguous()


def _qkv_op(
    qkv: torch.Tensor, positions: torch.Tensor, key: int, layer_name: LayerNameType
) -> torch.Tensor:
    from vllm.model_executor.layers.attention.attention import get_attention_context

    a = _LAYERS[key]
    _, attn, kv_cache, slots = get_attention_context(_resolve_layer_name(layer_name))
    if slots is not None and kv_cache.numel() == 0:
        slots = None
    q = torch.empty(qkv.shape[0], a.q_size, dtype=qkv.dtype, device=qkv.device)
    args = (
        qkv,
        positions,
        a.q_norm.weight,
        a.k_norm.weight,
        a.rotary_emb.cos_sin_cache,
        a.q_norm.variance_epsilon,
        a.num_heads,
        a.num_kv_heads,
        q,
    )
    kw: dict[str, Any] = {}
    if slots is not None:
        kw = dict(
            kv_cache=kv_cache,
            slot_mapping=slots,
            k_scale=attn._k_scale,
            v_scale=attn._v_scale,
        )
    lib = _lib()
    ck = (key, kv_cache.data_ptr() if slots is not None else 0)
    if ck not in _WHY:
        _WHY[ck] = lib.qkv_norm_rope_insert_unsupported_reason(*args, **kw)
    if _WHY[ck] is not None or not positions.is_contiguous():
        akr_nonlayer.blocked("DQKV", _WHY[ck] or "non-contiguous positions")
        return _qkv_stock(a, attn, qkv, positions, kv_cache, slots)
    lib.qkv_norm_rope_insert(*args, **kw)
    return q


def _qkv_fake(
    qkv: torch.Tensor, positions: torch.Tensor, key: int, layer_name: LayerNameType
) -> torch.Tensor:
    return qkv.new_empty((qkv.shape[0], _LAYERS[key].q_size))


direct_register_custom_op(
    op_name="kimi_akr_draft_qkv_norm_rope_insert",
    op_func=_qkv_op,
    fake_impl=_qkv_fake,
)


class _AkrQkvForward:
    """`DFlashQwen3Attention.forward` with the fused q / k norm, RoPE and KV insert."""

    _akr_qkv_key: int
    qkv_proj: Any
    attn: Any
    o_proj: Any

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        qkv, _ = self.qkv_proj(hidden_states)
        q = torch.ops.vllm.kimi_akr_draft_qkv_norm_rope_insert(
            qkv, positions, self._akr_qkv_key, _encode_layer_name(self.attn.layer_name)
        )
        output, _ = self.o_proj(self.attn(q, None, None))
        return output


_SUBCLASSES: dict[type, type] = {}


def _fused_class(cls: type) -> type:
    sub = _SUBCLASSES.get(cls)
    if sub is None:
        sub = _SUBCLASSES[cls] = type(f"Akr{cls.__name__}", (_AkrQkvForward, cls), {})
    return sub


def _install_qkv(model) -> int:
    layers = [layer.self_attn for layer in model.model.layers]
    for a in layers:
        if akr_nonlayer.blocked("DQKV", _layer_reason(a)):
            return 0
    for a in layers:
        key = len(_LAYERS)
        _LAYERS[key] = a
        a._akr_qkv_key = key
        a.__class__ = _fused_class(type(a))
    return len(layers)


# ------------------------------------------------------------------ context K/V store
class ContextKv:
    """`precompute_and_store_context_kv` of one `DFlashQwen3Model` instance."""

    def __init__(self, m) -> None:
        self.m = m

    def stock(self, *args) -> None:
        # the class's method (with `VLLM_K31_BEAT_CTXKV`'s wrap when installed)
        type(self.m).precompute_and_store_context_kv(self.m, *args)

    def __call__(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        m = self.m
        n = context_states.shape[0]
        layers = m._attn_layers
        slots = context_slot_mapping
        if slots is not None and not isinstance(slots, (list, tuple)):
            slots = [slots] * len(layers)
        if n == 0 or slots is None or any(s is None for s in slots):
            return self.stock(context_states, context_positions, context_slot_mapping)
        nl, nkv, hd = m._num_attn_layers, m._num_kv_heads, m._head_dim
        caches = [a.kv_cache for a in layers]
        args = (
            context_positions,
            m._k_norm_weights,
            m._rope_cos_sin_cache,
            caches,
            list(slots),
            [a._k_scale for a in layers],
            [a._v_scale for a in layers],
            m._rms_norm_eps,
        )
        lib = _lib()
        probe = context_states.new_empty((n, nl * 2 * nkv * hd))
        reason = lib.kv_norm_rope_insert_multilayer_unsupported_reason(probe, *args)
        if akr_nonlayer.blocked("DCTXKV", reason):
            return self.stock(context_states, context_positions, context_slot_mapping)
        kv = m._project_context_kv_token_major(context_states, n, nl, nkv, hd)
        lib.kv_norm_rope_insert_multilayer(kv.view(n, -1), *args)


def _ctx_reason(model) -> str | None:
    m = model.model
    if not hasattr(m, "_num_attn_layers"):
        return "the context K/V buffers are not built"
    if getattr(m.layers[0].self_attn, "v_scale", None) is not None:
        return "a V scale"
    if m._fused_kv_weight is None:
        return "no fused bf16 K/V projection (quantized draft)"
    reason = _lib().rope_unsupported_reason(
        m._rope_head_size,
        m._rope_cos_sin_cache.shape[-1],
        m._rope_is_neox,
        m._rope_cos_sin_cache,
    )
    if reason is not None:
        return reason
    w = m._k_norm_weights
    if w.dtype != torch.bfloat16 or tuple(w.shape) != (m._num_attn_layers, m._head_dim):
        return f"K norm weights {tuple(w.shape)} {w.dtype}"
    for attn in m._attn_layers:
        reason = _impl_reason(attn)
        if reason is not None:
            return reason
    return None


def setup(spec, model) -> None:
    """From `akr_draft.setup` (after the weights load, before compile and capture)."""
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    if not isinstance(model.model, DFlashQwen3Model):
        return
    on_qkv, on_ctx = akr_nonlayer.switch("DQKV"), akr_nonlayer.switch("DCTXKV")
    if not (on_qkv or on_ctx):
        return
    if not _lib().available():
        akr_nonlayer.blocked("DQKV", "needs gfx950 with FlyDSL")
        return
    if on_qkv:
        n = _install_qkv(model)
        if n:
            logger.info(
                "amd_kernels drafter: fused q / k norm + RoPE + KV insert on %d layers",
                n,
            )
    if on_ctx and not akr_nonlayer.blocked("DCTXKV", _ctx_reason(model)):
        model.model.precompute_and_store_context_kv = ContextKv(model.model)  # type: ignore[method-assign]
        logger.info(
            "amd_kernels drafter: context K norm + RoPE + KV inserts of %d layers "
            "in one launch",
            model.model._num_attn_layers,
        )
