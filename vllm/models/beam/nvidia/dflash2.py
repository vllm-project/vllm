# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA hooks for the common Beam DFlash2 model."""

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context

from ..common.dflash2 import (
    DFlash2BeamAttention,
    DFlash2BeamModel,
)
from ..common.dflash2 import (
    DFlash2BeamForCausalLM as CommonDFlash2BeamForCausalLM,
)
from .attention_ops import beam_qk_norm_rope
from .dflash2_context_ops import store_context_kv


class _NVIDIABackend:
    def forward_attention(
        self,
        attention: DFlash2BeamAttention,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if attention.v_scale is not None:
            v = v * attention.v_scale
        cache, slots = None, None
        mapping = get_forward_context().slot_mapping
        assert isinstance(mapping, dict)
        if mapping and attention.attn.kv_cache.dtype == q.dtype == torch.bfloat16:
            slots = mapping[attention.attn.layer_name]
            cache = attention.attn.kv_cache
            assert attention.attn.attn_backend.get_name() in (
                "FLASH_ATTN",
                "FLASHINFER",
            )
            assert attention.attn.impl.dcp_world_size == 1
            assert attention.attn.kv_sharing_target_layer_name is None
            assert cache.dtype == q.dtype == torch.bfloat16
            assert cache.ndim == 4
        rope = attention.rotary_emb
        q, k = beam_qk_norm_rope(
            q,
            k,
            attention.q_norm.weight,
            attention.k_norm.weight,
            attention.q_norm.variance_epsilon,
            positions,
            rope.cos_sin_cache,
            attention.head_dim,
            rope.rotary_dim,
            rope.is_neox_style,
            value=v,
            kv_cache=cache,
            slot_mapping=slots,
        )
        return (
            attention.attn(q, k, v) if cache is None else attention.attn(q, None, None)
        )

    def store_context_kv(
        self,
        model: DFlash2BeamModel,
        kv: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        if slot_mapping is None:
            return
        if isinstance(slot_mapping, (list, tuple)):
            slots = slot_mapping[0]
            if slots is None:
                assert all(s is None for s in slot_mapping)
                return
            assert all(
                s is not None
                and s.data_ptr() == slots.data_ptr()
                and s.shape == slots.shape
                and s.stride() == slots.stride()
                for s in slot_mapping
            )
        else:
            slots = slot_mapping
        first = model.layers[0].self_attn
        rotary = first.rotary_emb
        assert kv.dtype == torch.bfloat16
        assert rotary.rotary_dim == first.head_dim
        assert rotary.cos_sin_cache.dtype == kv.dtype
        assert slots.stride(0) == 1
        attns = [layer.self_attn.attn for layer in model.layers]
        caches = [attn.kv_cache for attn in attns]
        cache = caches[0]
        cache_dtype = attns[0].kv_cache_dtype
        fp8_dtype = {
            "fp8": torch.float8_e4m3fn,
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }.get(cache_dtype)
        for attn, layer_cache in zip(attns, caches):
            assert attn.attn_backend.get_name() in ("FLASH_ATTN", "FLASHINFER")
            assert attn.impl.dcp_world_size == 1
            assert attn.kv_sharing_target_layer_name is None
            assert attn.kv_cache_dtype == cache_dtype
            assert layer_cache.dtype == cache.dtype
            assert layer_cache.device == kv.device
            assert layer_cache.dtype == kv.dtype or fp8_dtype is not None
            assert layer_cache.ndim == 4 and layer_cache.stride(-1) == 1
            assert layer_cache.shape[1] == kv.shape[3]
            assert layer_cache.shape[-1] == 2 * kv.shape[-1]
            assert layer_cache.shape == cache.shape
            assert layer_cache.stride() == cache.stride()
        cache_pointers = tuple(layer_cache.data_ptr() for layer_cache in caches)
        scale_pointers = (
            tuple(
                scale.data_ptr()
                for attn in attns
                for scale in (attn._k_scale, attn._v_scale)
            )
            if fp8_dtype is not None
            else ()
        )
        pointers = cache_pointers + scale_pointers
        if getattr(model, "_context_cache_addresses", None) != pointers:
            model._context_cache_ptrs = torch.tensor(
                cache_pointers, dtype=torch.int64, device=kv.device
            )
            model._context_scale_ptrs = (
                torch.tensor(scale_pointers, dtype=torch.int64, device=kv.device)
                if scale_pointers
                else None
            )
            model._context_cache_addresses = pointers
        assert model._context_cache_ptrs.device == kv.device
        store_context_kv(
            kv,
            model._context_k_norm_weight,
            positions,
            rotary.cos_sin_cache,
            model._context_cache_ptrs,
            slots,
            cache.view(fp8_dtype) if fp8_dtype is not None else cache,
            first.k_norm.variance_epsilon,
            model._context_scale_ptrs,
        )


class DFlash2BeamForCausalLM(CommonDFlash2BeamForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            backend=_NVIDIABackend(),
        )


__all__ = ["DFlash2BeamForCausalLM"]
