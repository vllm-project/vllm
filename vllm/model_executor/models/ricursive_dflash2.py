# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 drafter for Ricursive Ripple, whose ``no_rope_layers`` are NoPE."""

import torch
from torch import nn

from .qwen3_dflash2 import (
    DFlash2Qwen3DecoderLayer,
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)


class _NoRope(nn.Module):
    """Identity rotation that keeps the attributes the fused context-K/V
    precompute reads from every layer's RoPE."""

    def __init__(self, rope: nn.Module) -> None:
        super().__init__()
        self.head_size = rope.head_size
        self.is_neox_style = rope.is_neox_style
        self.cos_sin_cache = rope.cos_sin_cache

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key


class RicursiveDFlash2DecoderLayer(DFlash2Qwen3DecoderLayer):
    def __init__(self, vllm_config, *, config, layer_idx: int, **kwargs) -> None:
        super().__init__(vllm_config, config=config, layer_idx=layer_idx, **kwargs)
        if layer_idx in config.no_rope_layers:
            self.self_attn.rotary_emb = _NoRope(self.self_attn.rotary_emb)


class RicursiveDFlash2Model(DFlash2Qwen3Model):
    decoder_layer_cls = RicursiveDFlash2DecoderLayer

    def _apply_context_rope(
        self, all_k: torch.Tensor, context_positions: torch.Tensor
    ) -> None:
        # Keep the fused RoPE call; restore the NoPE layers' keys afterwards.
        nope_layers = self.config.no_rope_layers
        nope_k = [all_k[i].clone() for i in nope_layers]
        super()._apply_context_rope(all_k, context_positions)
        for i, k in zip(nope_layers, nope_k):
            all_k[i].copy_(k)


class RicursiveDFlash2ForCausalLM(DFlash2Qwen3ForCausalLM):
    model_cls = RicursiveDFlash2Model
