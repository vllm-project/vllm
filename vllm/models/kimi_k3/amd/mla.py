# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD-specific MLA wrapper for Kimi-K3."""

from typing import cast

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper


class KimiK3NoPERotaryEmbedding(nn.Module):
    """Identity RoPE, so Kimi-K3 can reach the fused AITER MLA Q-prep kernel.

    Kimi-K3 MLA is NoPE (``mla_use_nope``), but the fused kernel that writes the
    KV cache and assembles the fp8 decode query also applies RoPE, and takes the
    cos/sin caches as required arguments. Feeding it ``cos = 1, sin = 0`` makes
    that rotation the identity.

    The cache is a single row. AITER's per-head and ``_opt`` decode kernels
    clamp ``pos`` into ``[0, cos_cache.size(0))``, so real positions can be
    passed through untouched -- at K3's ``max_position_embeddings`` of 1M a
    full-length constant cache would otherwise cost ~268 MB for a provable
    no-op. Its *general* decode kernel does not clamp, so the wrapper
    disables fusion for the configs that would select it.

    ``forward`` is never called: the K3 wrapper is NoPE and skips it.
    """

    is_neox_style: bool = True
    # Lets MLAAttention fuse on an impl that does not fuse every batch: the
    # wrapper's eager RoPE and the kernel's are both no-ops here.
    is_identity: bool = True

    def __init__(self, rotary_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        half = rotary_dim // 2
        cos_sin_cache = torch.cat(
            [torch.ones(1, half, dtype=dtype), torch.zeros(1, half, dtype=dtype)],
            dim=-1,
        )
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key


class KimiK3MultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    """Kimi-K3 MLA wrapper with eager AITER q/kv RMSNorm fusion."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_eager_qk_rmsnorm_fusion = bool(rocm_aiter_ops.is_enabled())
        # K3 is NoPE, so rotary_emb exists only to supply the fused kernel's
        # identity cos/sin; forward() below never applies it. A real RoPE here
        # would be silently dropped, so reject one outright.
        assert self.rotary_emb is None or isinstance(
            self.rotary_emb, KimiK3NoPERotaryEmbedding
        ), "Kimi-K3 MLA is NoPE; a positional rotary_emb would be ignored"

        # A single-row cos/sin cache is only safe on the AITER decode kernels
        # that clamp `pos` into it. Its general decode kernel does not, and
        # would read out of bounds. That kernel is chosen when the _opt
        # condition below fails -- K3 hits it at 3 heads/rank (TP32). The
        # _opt condition is independent of the KV block size and is also
        # satisfied by the per-head kernel's config, so it is decidable here.
        kernel_clamps_positions = (
            self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
            and self.kv_lora_rank * self.num_heads >= 2048
        )
        if not kernel_clamps_positions:
            self.mla_attn.impl.use_fused_qk_rope_cache = False

        # NoPE, so there is no eager RoPE to defer: this only decides whether
        # `positions` is forwarded. Fusing implies this, so the fused branch
        # can never find `positions` missing.
        self._defer_rope_to_fused_kernel = self.mla_attn.can_fuse_qk_rope_cache()

    def _normalize_q_kv(
        self,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_layernorm = cast(RMSNorm, self.q_a_layernorm)
        kv_layernorm = cast(RMSNorm, self.kv_a_layernorm)

        if self._use_eager_qk_rmsnorm_fusion and not torch.compiler.is_compiling():
            return torch.ops.vllm.fused_mla_dual_rms_norm(
                q_c,
                q_layernorm.weight,
                kv_c,
                kv_layernorm.weight,
                q_layernorm.variance_epsilon,
                kv_layernorm.variance_epsilon,
            )

        return q_layernorm(q_c), kv_layernorm(kv_c)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            q_proj_input, kv_c_normed = self._normalize_q_kv(q_c, kv_c)
            q_proj_layer = self.q_b_proj
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

        # Add head dim of 1 to k_pe.
        k_pe = k_pe.unsqueeze(1)

        q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        if self.dcp_q_replicate:
            heads *= q_proj_layer.group_size
        q = q.view(-1, heads, self.qk_head_dim)

        # NoPE: no rotation is applied here. `positions` is forwarded only so
        # the fused Q-prep kernel can index its identity cos/sin caches.
        fused_positions = positions if self._defer_rope_to_fused_kernel else None

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        q_dcp_replicated = None
        if self.dcp_q_replicate:
            q_dcp_replicated, q = q, q_proj_layer._local_view(q)

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            q_dcp_replicated=q_dcp_replicated,
            positions=fused_positions,
        )

        if self.g_proj is not None:
            attn_out = attn_out * self.g_proj(hidden_states)[0].sigmoid()

        return self.o_proj(attn_out)[0]
