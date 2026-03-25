# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Debug flag - set to True to enable verbose debug output
_DEBUG_MLA = False

from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig


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

        # Extract RoPE caches for AITER fused kernels
        if self.rotary_emb is not None:
            # RoPE stores combined cos_sin_cache, need to split it
            # Format: [seq_len, rotary_dim] where first half is cos, second half is sin
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            rotary_dim = self.rotary_emb.rotary_dim
            half_dim = rotary_dim // 2
            self.cos_cache = cos_sin_cache[:, :half_dim]
            self.sin_cache = cos_sin_cache[:, half_dim:]
            self.is_neox_style = self.rotary_emb.is_neox_style
        else:
            self.cos_cache = None
            self.sin_cache = None
            self.is_neox_style = False

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
            # Pass RoPE caches for AITER fused kernels
            cos_cache=self.cos_cache,
            sin_cache=self.sin_cache,
            is_neox_style=self.is_neox_style,
        )

        self.prefix = prefix

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

        # DEBUG: Print k_pe BEFORE any RoPE
        if _DEBUG_MLA:
            print("\n[mla.py DEBUG] k_pe BEFORE RoPE (raw from projection):")
        if _DEBUG_MLA:
            print(f"  k_pe[0,0,:5]: {k_pe[0, 0, :5].cpu().float().numpy()}")

        # STEP 3: Apply RoPE (conditional based on fused kernel usage)
        # Skip RoPE only when AITER fused kernels will handle it
        # Fused kernels apply RoPE internally (decode) or in forward_impl (prefill)
        use_aiter_fused = (
            hasattr(self.mla_attn, "use_aiter_fused_decode")
            and self.mla_attn.use_aiter_fused_decode
        )

        if self.rotary_emb is not None and not use_aiter_fused:
            # UNFUSED PATH: Apply RoPE to ALL tokens here
            # This is the original behavior - handles prefill, decode, and mixed batches
            print("[mla.py] Path: UNFUSED - Applying RoPE here")
            if _DEBUG_MLA:
                print("\n[mla.py DEBUG] k_pe BEFORE RoPE (raw from projection):")
            if _DEBUG_MLA:
                print(f"  k_pe[0,0,:5]: {k_pe[0, 0, :5].cpu().float().numpy()}")
            if _DEBUG_MLA:
                print(f"  positions[0:5]: {positions[0:5].cpu().numpy()}")

            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions,
                q[..., self.qk_nope_head_dim :],  # Q PE part gets RoPE
                k_pe,  # K PE gets RoPE
            )
            # After this: Q and k_pe have RoPE applied for unfused path
            if _DEBUG_MLA:
                print("\n[mla.py] RoPE applied to Q_PE and K_PE (unfused path)")
            if _DEBUG_MLA:
                print("[mla.py DEBUG] k_pe AFTER RoPE:")
            if _DEBUG_MLA:
                print(f"  k_pe[0,0,:5]: {k_pe[0, 0, :5].cpu().float().numpy()}")
            if _DEBUG_MLA:
                print(
                    f"  q (RoPE part) [0,0,{self.qk_nope_head_dim}:{self.qk_nope_head_dim + 5}]: {q[0, 0, self.qk_nope_head_dim : self.qk_nope_head_dim + 5].cpu().float().numpy()}"
                )
        else:
            # FUSED PATH: Skip RoPE here, will be applied later
            print("[mla.py] Path: FUSED - RoPE will be applied in mla_attention.py")
            if _DEBUG_MLA:
                print("\n[mla.py] RoPE skipped (will be applied in mla_attention.py)")

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(
                hidden_states, q_c, positions, self.indexer_rope_emb
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        # STEP 4: Pass to mla_attention
        attn_out = self.mla_attn(
            q,  # Has RoPE if unfused, NO RoPE if fused
            kv_c_normed,
            k_pe,  # Has RoPE if unfused, NO RoPE if fused
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            positions=positions,
            use_aiter_fused=use_aiter_fused,
            rotary_emb=self.rotary_emb,
        )

        final_out = self.o_proj(attn_out)[0]
        return final_out
