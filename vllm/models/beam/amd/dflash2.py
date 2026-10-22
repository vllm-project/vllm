# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm hooks for the common Beam DFlash2 model."""

import torch

from vllm.config import VllmConfig

from ..common.dflash2 import (
    DFlash2BeamAttention,
    DFlash2BeamModel,
)
from ..common.dflash2 import (
    DFlash2BeamForCausalLM as CommonDFlash2BeamForCausalLM,
)
from .attention_ops import beam_qk_norm_rope
from .dflash2_context_ops import transform_context_kv


class _AMDBackend:
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
        )
        # ROCm backends own their cache layout and update path.
        return attention.attn(q, k, v)

    def store_context_kv(
        self,
        model: DFlash2BeamModel,
        kv: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        first = model.layers[0].self_attn
        rotary = first.rotary_emb
        assert kv.dtype == torch.bfloat16
        assert rotary.rotary_dim == first.head_dim
        assert rotary.cos_sin_cache.dtype == kv.dtype
        keys, values = transform_context_kv(
            kv,
            model._context_k_norm_weight,
            positions,
            rotary.cos_sin_cache,
            first.k_norm.variance_epsilon,
            first.v_scale if first.v_scale is not None else 1.0,
            rotary.is_neox_style,
        )
        if slot_mapping is None:
            return

        per_layer = isinstance(slot_mapping, (list, tuple))
        if per_layer and len(slot_mapping) != len(model.layers):
            raise ValueError(
                "DFlash2 context slot mappings must have one entry per draft layer."
            )
        for index, layer in enumerate(model.layers):
            slots = slot_mapping[index] if per_layer else slot_mapping
            if slots is None:
                continue
            attn = layer.self_attn.attn
            if not hasattr(attn.impl, "do_kv_cache_update"):
                raise NotImplementedError(
                    f"{attn.impl.__class__.__name__} cannot update the KV cache."
                )
            attn.impl.do_kv_cache_update(
                attn,
                keys[index],
                values[index],
                attn.kv_cache,
                slots,
            )


class DFlash2BeamForCausalLM(CommonDFlash2BeamForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, backend=_AMDBackend())


__all__ = ["DFlash2BeamForCausalLM"]
