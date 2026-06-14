# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig, get_current_vllm_config
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import _encode_layer_name


def _mla_rope_kvcache_fusion_enabled(
    rotary_emb: torch.nn.Module | None,
    qk_rope_head_dim: int,
    kv_cache_dtype: str,
    is_sparse: bool,
    model_dtype: torch.dtype,
    fuse_rope_kvcache_cat_mla: bool,
    calculate_kv_scales: bool,
) -> bool:
    """Decide whether the manual RoPE + MLA KV-cache-write fusion
    can be used.

    The allowlist starts from the former compiler-pass matcher coverage,
    then adds runtime safety checks needed by the manual call site: kernel
    dtype parity, unsupported cache layouts, sparse MLA, and dynamic
    KV-scale calculation.
    """
    if not fuse_rope_kvcache_cat_mla:
        return False
    if not current_platform.is_cuda_alike():
        return False
    if rotary_emb is None:
        return False
    # Exact-type allowlist: subclasses (Llama3/MRoPE/...) implement different
    # rotation math than the fused kernel.
    if type(rotary_emb) not in (RotaryEmbedding, DeepseekScalingRotaryEmbedding):
        return False
    # The fused kernel rotates the full rope head width.
    if not (rotary_emb.rotary_dim == rotary_emb.head_size == qk_rope_head_dim):
        return False
    # Kernel enforces cos_sin_cache dtype == activation dtype.
    if rotary_emb.cos_sin_cache.dtype != model_dtype:
        return False
    # Dynamic KV-scale calibration updates _k_scale inside
    # MLAAttention.forward (maybe_calc_kv_scales), i.e. *after* the fused
    # producer would have already quantized and written this layer's cache
    # row with the stale scale. Keep the unfused order in that case.
    if calculate_kv_scales:
        return False
    # fp8_ds_mla / sparse use cache layouts only the unfused path supports.
    return not (kv_cache_dtype == "fp8_ds_mla" or is_sparse)


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

        vllm_config = get_current_vllm_config()
        self._use_fused_rope_kv_cache = _mla_rope_kvcache_fusion_enabled(
            rotary_emb=self.rotary_emb,
            qk_rope_head_dim=self.qk_rope_head_dim,
            kv_cache_dtype=self.mla_attn.kv_cache_dtype,
            is_sparse=self.is_sparse,
            model_dtype=vllm_config.model_config.dtype,
            fuse_rope_kvcache_cat_mla=(
                vllm_config.compilation_config.pass_config.fuse_rope_kvcache_cat_mla
            ),
            calculate_kv_scales=self.mla_attn.calculate_kv_scales,
        )

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

        kv_cache_dummy_dep: torch.Tensor | None = None
        if self._use_fused_rope_kv_cache:
            # Manual RoPE + KV-cache-write fusion: rotates
            # q_pe/k_pe in place and writes this layer's (kv_c, k_pe) row to
            # the KV cache in one kernel. The fused kernel takes 2-D k_pe, so
            # this runs before the unsqueeze; the dummy output is threaded
            # into the attention op to preserve write -> attend ordering
            # under torch.compile.
            kv_cache_dummy_dep = torch.ops.vllm.fused_rope_unified_mla_kv_cache_update(
                positions,
                q[..., self.qk_nope_head_dim :],
                k_pe,
                kv_c_normed,
                self.rotary_emb.cos_sin_cache,
                self.rotary_emb.is_neox_style,
                self.mla_attn.kv_cache_dtype,
                self.mla_attn._k_scale,
                _encode_layer_name(self.mla_attn.layer_name),
            )
            # Add head dim of 1 to k_pe
            k_pe = k_pe.unsqueeze(1)
        else:
            # Add head dim of 1 to k_pe
            k_pe = k_pe.unsqueeze(1)

            if self.rotary_emb is not None:
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
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )

        return self.o_proj(attn_out)[0]
