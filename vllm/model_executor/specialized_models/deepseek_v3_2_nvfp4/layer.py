# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MLA attention and decoder layer for DeepSeek V3.2 on SM100 (Blackwell).

MLAAttention:
  KV cache update -> W_UK_T absorption -> sparse attn kernel -> W_UV up-proj
  MLAAttention kept only as a registration stub for KV cache / backend.

DecoderLayer:
  Single decoder layer: norm -> attn -> norm -> MoE/MLP.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

from .kernels import fused_norm_rope, fused_q
from .sparse_indexer import sparse_attn_indexer


def dsa(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[layer_name]
    attn = layer.attn
    mla = attn.mla_attn

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        output.zero_()
        return output

    mla_attn_metadata = attn_metadata.get(mla.layer_name)
    if mla_attn_metadata is None:
        output.zero_()
        return output

    num_actual_toks = mla_attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
    if num_actual_toks == 0:
        output.zero_()
        return output

    # Step 2. fused norm + rope + cache writes
    slot_mapping = None
    indexer_k_cache = None
    mla_kv_cache = None
    mla_k_scale = None
    idx_meta = attn_metadata.get(attn.indexer_k_cache.prefix)
    if idx_meta is not None:
        slot_mapping = idx_meta.slot_mapping  # type: ignore[attr-defined]
        indexer_k_cache = attn.indexer_k_cache.kv_cache
        mla_kv_cache = attn.mla_attn.kv_cache
        mla_k_scale = attn.mla_attn._k_scale

    q_c = fused_norm_rope(
        positions,
        q_c,
        attn.q_a_layernorm_weight,
        layer.rms_norm_eps,
        kv_c,
        attn.kv_a_layernorm_weight,
        attn.rms_norm_eps,
        k_pe,
        attn.rotary_emb.cos_sin_cache,
        index_k,
        attn.indexer_k_norm.weight,
        attn.indexer_k_norm.bias,
        attn.rms_norm_eps,
        attn.indexer_rope_emb.cos_sin_cache,
        attn.topk_indices_buffer,
        slot_mapping=slot_mapping,
        indexer_k_cache=indexer_k_cache,
        mla_kv_cache=mla_kv_cache,
        mla_kv_cache_dtype=attn.mla_attn.kv_cache_dtype,
        mla_k_scale=mla_k_scale,
    )

    # Step 3. q_c -> index_q, q
    step3_out = torch.mm(q_c, layer._fused_step3_q_w.T)
    index_q, q = step3_out.split(layer._q_split_sizes, dim=-1)
    index_q = index_q.view(-1, attn.index_n_heads, attn.index_head_dim)
    q = q.view(-1, attn.num_local_heads, attn.qk_head_dim)

    # Step 4. Q RoPE + W_UK_T absorption + FP8 packing
    q_nope, q_pe = q.split(
        [mla.qk_nope_head_dim, mla.qk_rope_head_dim],
        dim=-1,
    )
    q_nope = q_nope.transpose(0, 1)
    ql_nope = torch.bmm(q_nope, mla.W_UK_T)
    ql_nope = ql_nope.transpose(0, 1)

    index_q_fp8, index_weights, mqa_q = fused_q(
        positions,
        q_pe,
        attn.rotary_emb.cos_sin_cache,
        index_q,
        attn.indexer_rope_emb.cos_sin_cache,
        ql_nope,
        mla._q_scale,
        index_weights,
        attn.indexer_softmax_scale,
        attn.index_n_heads**-0.5,
    )

    # Steps 5-6. Sparse indexer + MLA sparse decode attention
    sparse_attn_indexer(
        attn.indexer_k_cache.prefix,
        attn.indexer_k_cache.kv_cache,
        index_q_fp8,
        index_weights,
        attn.topk_tokens,
        attn.index_head_dim,
        layer.max_model_len,
        layer.indexer_workspace_size,
        attn.topk_indices_buffer,
    )

    mqa_q = mqa_q[:num_actual_toks]
    kv_cache = mla.kv_cache
    if mla.kv_cache_dtype.startswith("fp8") and mla.kv_cache_dtype != "fp8_ds_mla":
        kv_cache = kv_cache.view(torch.float8_e4m3fn)
    attn_out, _ = mla.impl.forward_mqa(mqa_q, kv_cache, mla_attn_metadata, mla)
    x = attn_out.view(-1, mla.num_heads, mla.kv_lora_rank).transpose(0, 1)

    out = output[:num_actual_toks].view(-1, mla.num_heads, mla.v_head_dim)
    out = out.transpose(0, 1)
    torch.bmm(x, mla.W_UV, out=out)
    return output


def dsa_fake(
    positions: torch.Tensor,
    q_c: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del positions, q_c, kv_c, k_pe, index_k, index_weights, layer_name
    return output


direct_register_custom_op(
    op_name="monolithic_attn",
    op_func=dsa,
    fake_impl=dsa_fake,
    mutates_args=["output"],
    dispatch_key=current_platform.dispatch_key,
)


class DeepseekV32DecoderLayer(nn.Module):
    """
    Single decoder layer: norm -> attn -> norm -> MoE/MLP.
    Norms are raw weight + direct kernel call.
    Gate inlined as raw weight, experts kept as FusedMoE for quantization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        topk_indices_buffer: torch.Tensor,
        prefix: str = "",
    ) -> None:
        super().__init__()
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.layer_name = prefix
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.tp_size = get_tensor_model_parallel_world_size()

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.indexer_workspace_size = get_max_prefill_buffer_size(vllm_config)
        self.max_model_len = vllm_config.model_config.max_model_len

        # Use the regular vLLM RMSNorm modules so the compiler sees the
        # canonical residual-add + RMSNorm pattern.
        dtype = torch.get_default_dtype()
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )

        # Fused QKV A-projection lives inside self_attn namespace
        # for weight loading compatibility with original checkpoint paths
        from vllm.model_executor.models.deepseek_v2 import (
            DeepSeekV2FusedQkvAProjLinear,
        )

        self.self_attn = nn.Module()
        self.self_attn.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            config.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.fused_qkv_a_proj",
        )

        # MLA Attention
        self.attn = DeepseekV32MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=f"{prefix}.self_attn",
        )

        # MoE or Dense MLP
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        self.is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        from vllm.model_executor.models.deepseek_v2 import (
            DeepseekV2MLP,
            DeepseekV2MoE,
        )

        if self.is_moe:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Step 1. hidden_states -> q_c, kv_c, k_pe
        #                       -> index_k, index_weights
        out = self.self_attn.fused_qkv_a_proj(hidden_states)
        if isinstance(out, tuple):
            out = out[0]

        q_c, kv_c, k_pe = out.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        index_k, index_weights = torch.mm(
            hidden_states, self._fused_indexer_weights.T
        ).split(
            self._indexer_weights_split_sizes,
            dim=-1,
        )

        # Steps 2-6. Combined: fused norm/rope + Q projections + sparse MLA.
        mla = self.attn.mla_attn
        output_shape = (hidden_states.shape[0], mla.num_heads * mla.v_head_dim)
        output_dtype = mla.W_UV.dtype
        attn_out = torch.empty(
            output_shape,
            dtype=output_dtype,
            device=hidden_states.device,
        )
        attn_out = torch.ops.vllm.monolithic_attn(
            positions,
            q_c,
            kv_c,
            k_pe,
            index_k,
            index_weights,
            attn_out,
            self.layer_name,
        )

        hidden_states, _ = self.attn.o_proj(attn_out)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def fuse_indexer_weights(self) -> None:
        """Fuse Step 1 and Step 3 BF16 linears used by the inlined path.

        Call after model weights are loaded.
        """
        attn = self.attn
        wk = attn.indexer_wk.weight.data  # [128, 7168]
        wp = attn.indexer_weights_proj.weight.data  # [64, 7168]
        if wk.dtype != wp.dtype:
            raise ValueError(
                "Cannot fuse indexer weights: expected matching dtypes for "
                "indexer_wk and indexer_weights_proj."
            )
        self._fused_indexer_weights = nn.Parameter(
            torch.cat([wk, wp], dim=0),  # [192, 7168]
            requires_grad=False,
        )
        self._indexer_weights_split_sizes = [wk.shape[0], wp.shape[0]]

        wq_b = attn.indexer_wq_b.weight.data
        q_b = attn.q_b_proj.weight.data
        if wq_b.dtype != q_b.dtype:
            raise ValueError(
                "Cannot fuse Step 3 weights: expected matching dtypes for "
                "indexer_wq_b and q_b_proj."
            )
        self._fused_step3_q_w = nn.Parameter(
            torch.cat([wq_b, q_b], dim=0),
            requires_grad=False,
        )
        self._q_split_sizes = [wq_b.shape[0], q_b.shape[0]]


class DeepseekV32MLAAttention(nn.Module):
    """
    MLA attention for DeepSeek V3.2 targeting SM100.
    MLA forward fully inlined. MLAAttention kept only for KV cache
    registration and backend/impl initialization.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None,
        topk_indices_buffer: torch.Tensor,
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
        self.num_local_heads = num_heads // get_tensor_model_parallel_world_size()
        self.scaling = self.qk_head_dim**-0.5
        self.rms_norm_eps = config.rms_norm_eps

        # Q path
        self.q_a_layernorm_weight = nn.Parameter(
            torch.ones(q_lora_rank, dtype=torch.get_default_dtype())
        )
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )

        # KV path
        self.kv_a_layernorm_weight = nn.Parameter(
            torch.ones(kv_lora_rank, dtype=torch.get_default_dtype())
        )
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # Output projection (TP sync point)
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # RoPE
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # V3.2 Sparse Indexer (inlined)
        self.indexer_rope_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
        )
        self.topk_tokens = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.indexer_softmax_scale = config.index_head_dim**-0.5
        self.indexer_quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        self.indexer_wq_b = ReplicatedLinear(
            q_lora_rank,
            config.index_head_dim * config.index_n_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer.wq_b",
        )
        self.indexer_wk = ReplicatedLinear(
            hidden_size,
            config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer.wk",
        )
        self.indexer_k_norm = LayerNorm(config.index_head_dim, eps=1e-6)
        self.indexer_weights_proj = ReplicatedLinear(
            hidden_size,
            config.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.indexer.weights_proj",
        )

        idx_dim = config.index_head_dim
        indexer_cache_head_dim = idx_dim + idx_dim // 128 * 4
        self.indexer_k_cache = DeepseekV32IndexerCache(
            head_dim=indexer_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.indexer.k_cache",
            cache_config=cache_config,
        )
        self.indexer_op = SparseAttnIndexer(
            self.indexer_k_cache,
            self.indexer_quant_block_size,
            "ue8m0",
            self.topk_tokens,
            config.index_head_dim,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            self.topk_indices_buffer,
        )

        # MLAAttention stub: only for KV cache registration + backend init.
        # We never call its forward(); we inline everything below.
        class _IndexerProxy:
            def __init__(proxy_self):
                proxy_self.topk_indices_buffer = topk_indices_buffer
                proxy_self.indexer_op = self.indexer_op

        self._indexer_proxy = _IndexerProxy()
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mla_attn",
            use_sparse=True,
            indexer=self._indexer_proxy,
        )


def remap_weight_name(name: str) -> str:
    """Remap checkpoint names that differ from the module layout."""
    replacements = [
        (
            "self_attn.q_a_layernorm.weight",
            "attn.q_a_layernorm_weight",
        ),
        (
            "self_attn.kv_a_layernorm.weight",
            "attn.kv_a_layernorm_weight",
        ),
        ("self_attn.q_b_proj", "attn.q_b_proj"),
        ("self_attn.kv_b_proj", "attn.kv_b_proj"),
        ("self_attn.o_proj", "attn.o_proj"),
        ("self_attn.indexer.", "attn.indexer_"),
    ]
    for old, new in replacements:
        if old in name:
            return name.replace(old, new)
    return name
