# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

        # F3: fused RoPE + MLA KV-cache write gate (ROCm + aiter only).
        # Auto-enables when AITER has fused_qk_rope_concat_and_cache_mla.
        # No env var required — follows has_fused_rmsnorm_mxfp4_quant() pattern.
        self._f3_fusion_enabled: bool = False
        if current_platform.is_rocm():
            try:
                from vllm._aiter_ops import rocm_aiter_ops

                self._f3_fusion_enabled = bool(
                    rocm_aiter_ops.is_mla_enabled()
                    and rocm_aiter_ops.has_fused_rope_mla_kv_cache()
                )
            except Exception:
                pass  # aiter not available; stay False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

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
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self._f3_fusion_enabled and self.rotary_emb is not None:
            # F3: single Triton kernel — RoPE(q_pe, k_pe) + kv_cache write.
            # Runs here with PRE-RoPE tensors; replaces the separate rotary_emb
            # call and the do_kv_cache_update call inside mla_attn.
            from vllm._aiter_ops import rocm_aiter_ops
            from vllm.forward_context import get_forward_context

            fwd_ctx = get_forward_context()
            slot_mapping_dict = fwd_ctx.slot_mapping
            if isinstance(slot_mapping_dict, list):
                slot_mapping_dict = slot_mapping_dict[0]
            layer_slot_mapping = slot_mapping_dict.get(self.mla_attn.layer_name)
            if layer_slot_mapping is not None and self.mla_attn.kv_cache.numel() > 0:
                q_nope = q[..., : self.qk_nope_head_dim]
                q_pe_pre = q[..., self.qk_nope_head_dim :]
                kv_c = kv_c_normed.squeeze(1)  # [B, kv_lora_rank]
                cos_sin = self.rotary_emb.cos_sin_cache
                head_dim = self.qk_rope_head_dim
                cos_cache = cos_sin[:, :head_dim]
                sin_cache = cos_sin[:, head_dim:]
                rocm_aiter_ops.fused_rope_and_mla_kv_cache_write(
                    q_nope=q_nope,
                    q_pe=q_pe_pre,
                    kv_c=kv_c,
                    k_pe=k_pe.squeeze(1),
                    kv_cache=self.mla_attn.kv_cache,
                    q_out=q,
                    slot_mapping=layer_slot_mapping.flatten(),
                    k_scale=self.mla_attn._k_scale,
                    q_scale=self.mla_attn._k_scale,
                    positions=positions,
                    cos_cache=cos_cache,
                    sin_cache=sin_cache,
                    is_neox=self.rotary_emb.is_neox_style,
                )
                # kv_cache already updated by the fused kernel above.
                # do_kv_cache_update inside mla_attn will write the same data
                # again (redundant but correct); the duplicate write will be
                # removed in the follow-on PR when this flag defaults to True.
            else:
                # Fallback: slot_mapping unavailable or kv_cache empty
                q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                    positions, q[..., self.qk_nope_head_dim :], k_pe
                )
        elif self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]
