# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_K31_BEAT_CTXKV``: a DFlash-family draft's context K/V store
(``DFlashQwen3Model.precompute_and_store_context_kv``) runs its K norm, RoPE
and per-layer cache writes as one launch (``ops/ctx_kv_store``) after the
stock K/V projection. Off for anything the launch does not serve (quantized or
non-TRITON_ATTN caches, V scales, partial RoPE, a layer without a slot
mapping), which runs stock; why is logged once."""

import functools

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.levers import CTXKV, lever_on, wrap

logger = init_logger(__name__)

# Set on the model: None when the launch serves it, else why not.
_REASON = "_k31_ctxkv_why_not"


def install() -> None:
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    wrap(DFlashQwen3Model, "precompute_and_store_context_kv", _wrap_precompute)


def model_why_not(model) -> str | None:
    """Why the model's draft layers are outside the launch (None: inside)."""
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    if not hasattr(model, "_num_attn_layers"):
        return "context K/V buffers not built"
    if getattr(model.layers[0].self_attn, "v_scale", None) is not None:
        return "V scale"
    cache = model._rope_cos_sin_cache
    if cache.dim() != 2 or cache.shape[-1] != model._head_dim:
        return "partial RoPE"
    if model._rope_head_size != model._head_dim:
        return "RoPE head size differs from the head dim"
    if model._head_dim & (model._head_dim - 1) or model._num_kv_heads & (
        model._num_kv_heads - 1
    ):
        return "head dim or KV heads not a power of 2"
    if model._k_norm_weights.shape != (model._num_attn_layers, model._head_dim):
        return "K norm weights"
    for attn in model._attn_layers:
        impl = attn.impl
        if not isinstance(impl, TritonAttentionImpl):
            return f"draft attention {type(impl).__name__}"
        if impl.attn_type != AttentionType.DECODER:
            return f"attention type {impl.attn_type}"
        if impl.kv_cache_dtype != "auto" or impl._is_per_token_head_quant:
            return f"KV cache dtype {impl.kv_cache_dtype}"
    return None


def _why_not(model, context_slot_mapping) -> str | None:
    from vllm.models.kimi_k3.amd.ops import ctx_kv_store

    why = getattr(model, _REASON, "")
    if why == "":
        why = model_why_not(model)
        if hasattr(model, "_num_attn_layers"):
            setattr(model, _REASON, why)
    if why is not None:
        return why
    if isinstance(context_slot_mapping, (list, tuple)) and any(
        s is None for s in context_slot_mapping
    ):
        return "a layer without a slot mapping"
    return ctx_kv_store.layout_why_not(
        [a.kv_cache for a in model._attn_layers],
        model._num_kv_heads,
        model._head_dim,
    )


def _wrap_precompute(stock):
    @functools.wraps(stock)
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping=None,
    ) -> None:
        if context_slot_mapping is None or not lever_on(CTXKV):
            return stock(self, context_states, context_positions, context_slot_mapping)
        why = _why_not(self, context_slot_mapping)
        if why is not None:
            logger.info_once("Draft context K/V store launch off: %s", why)
            return stock(self, context_states, context_positions, context_slot_mapping)
        from vllm.models.kimi_k3.amd.ops import ctx_kv_store

        logger.info_once(
            "Draft context K/V store: one launch for %d layers",
            self._num_attn_layers,
        )
        L, nkv, hd = self._num_attn_layers, self._num_kv_heads, self._head_dim
        all_kv = self._project_context_kv_token_major(
            context_states, context_states.shape[0], L, nkv, hd
        )
        if all_kv.stride()[1:] != (2 * nkv * hd, nkv * hd, hd, 1):
            all_kv = all_kv.contiguous()
        if not isinstance(context_slot_mapping, (list, tuple)):
            context_slot_mapping = [context_slot_mapping] * L
        ctx_kv_store.store_context_kv(
            all_kv,
            context_positions,
            self._k_norm_weights,
            self._rope_cos_sin_cache,
            self._rms_norm_eps,
            self._rope_is_neox,
            [a.kv_cache for a in self._attn_layers],
            list(context_slot_mapping),
        )

    return precompute_and_store_context_kv
