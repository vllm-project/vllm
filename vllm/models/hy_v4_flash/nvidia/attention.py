# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer MLA projections with Flash's SWA/IndexPool selection."""

from copy import deepcopy

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.kernels.triton.fused_sigmoid_gate_mul import fused_sigmoid_gate_mul
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.hpc import hpc_gated_mla_gemm, hpc_gated_mla_supported
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import make_attention_sink
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.hy_v4.common.mrope import fused_mrope_norm, rope_into
from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend

from .compressed_indexer import FlashCompressedIndexer
from .swa import FlashSparseBackend, FlashSWABackend, FlashWindowIndexer


class HYV4FlashMLAAttention(MLAAttention):
    # FlashSWABackend indexes live pages and forces the paged FP8 kernel.
    supports_fp8_ds_mla_sliding_window = True


class HYV4FlashAttention(HYV4MLAAttention):
    q_a_layernorm: RMSNorm
    q_b_proj: ColumnParallelLinear
    kv_a_proj_with_mqa: ReplicatedLinear
    linear_gate: ColumnParallelLinear

    def __init__(self, vllm_config, config, layer_idx, topk_indices_buffer, prefix):
        nn.Module.__init__(self)
        self.is_window_layer = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.config = deepcopy(config)
        self.config.rope_parameters = config.rope_for_layer(layer_idx)
        self.hidden_size = config.hidden_size
        self.layer_idx = self.layer_id = layer_idx
        self.prefix = prefix
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.max_position_embeddings = vllm_config.model_config.max_model_len
        self.num_heads = (
            config.swa_num_attention_heads
            if self.is_window_layer
            else config.num_attention_heads
        )
        self.num_local_heads = self.num_heads // get_tensor_model_parallel_world_size()
        self.qk_nope_head_dim = (
            config.swa_qk_nope_head_dim
            if self.is_window_layer
            else config.qk_nope_head_dim
        )
        self.qk_rope_head_dim = (
            config.swa_qk_rope_head_dim
            if self.is_window_layer
            else config.qk_rope_head_dim
        )
        self.v_head_dim = (
            config.swa_v_head_dim if self.is_window_layer else config.v_head_dim
        )
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.is_sparse = self.create_indexer = True
        self.skip_topk = self.external_topk = False
        self.topk_snapshot = None
        self.gated_mla = config.gated_mla
        self.learnable_sink = config.learnable_sink
        capability = current_platform.get_device_capability()
        if capability is None or capability.major not in (9, 10, 12):
            raise NotImplementedError("Flash MLA requires Hopper or Blackwell GPUs")
        sparse_backend: type[AttentionBackend]
        swa_backend: type[AttentionBackend]
        if capability.major == 12:
            # RTX 50 / RTX PRO Blackwell and DGX Spark: FlashInfer's SM120 kernels
            from .sm120 import FlashSM120SparseBackend, FlashSM120SWABackend

            sparse_backend, swa_backend = FlashSM120SparseBackend, FlashSM120SWABackend
        else:
            sparse_backend, swa_backend = FlashSparseBackend, FlashSWABackend
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise ValueError("Flash sparse MLA requires latent512 + rotary64 KV")
        if (
            not config.learnable_sink
            or not config.gated_mla
            or config.gating_type != "elementwise"
        ):
            raise ValueError("Flash requires learnable sinks and elementwise gated MLA")
        if any(
            getattr(config, name, None) is not None
            for name in ("gate_lora_rank", "o_lora_rank")
        ):
            raise ValueError("Flash requires direct gate/output projections")
        if (
            getattr(config, "gate_groups", 1) != 1
            or getattr(config, "o_groups", 1) != 1
            or getattr(config, "full_nope", False)
            or not getattr(config, "qk_norm", True)
        ):
            raise ValueError("Flash requires normalized ungrouped RoPE MLA")
        quant = vllm_config.quant_config
        self.fused_qkv_a_proj = None
        self.q_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_a_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.linear_gate = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant,
            prefix=f"{prefix}.linear_gate",
        )
        self.use_hpc_gated_mla = hpc_gated_mla_supported(
            config.gating_type, self.linear_gate
        )
        has_post_proj_norm = config.sandwich_norm and (
            self.is_window_layer or not config.dsa_skip_attn_post_norm
        )
        # With a post-projection norm, the all-reduce is fused into it.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant,
            reduce_results=not has_post_proj_norm,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=self.config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rope_emb = self.rotary_emb
        self.indexer = (
            FlashWindowIndexer(
                config.sliding_window, topk_indices_buffer, f"{prefix}.attn"
            )
            if self.is_window_layer
            else FlashCompressedIndexer(
                vllm_config, self.config, topk_indices_buffer, f"{prefix}.indexer"
            )
        )
        self.learnable_sink_param = make_attention_sink(
            self.num_local_heads, torch.float32
        )
        self._force_sparse_mqa()
        self.mla_attn = HYV4FlashMLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=True,
            indexer=self.indexer,
            topk_indices_buffer=self.indexer.topk_indices_buffer,
            attn_backend=swa_backend if self.is_window_layer else sparse_backend,
            sliding_window=config.sliding_window if self.is_window_layer else None,
            sinks=self.learnable_sink_param,
        )
        self.post_proj_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if has_post_proj_norm
            else None
        )

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        q_c = self.q_a_proj(hidden_states)[0]
        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
        if positions.ndim == 2:
            q_c, kv_c, k_pe, rope_pos, cos_sin = fused_mrope_norm(
                self.rotary_emb,
                positions,
                q_c,
                kv,
                self.q_a_layernorm,
                self.kv_a_layernorm,
            )
            q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            q_pe = q[..., self.qk_nope_head_dim :]
            rope_into(rope_pos, q_pe, cos_sin, q_pe)
        else:
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            kv_c, k_pe = kv.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
            kv_c = self.kv_a_layernorm(kv_c)
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe.unsqueeze(1)
            )
        if llama_4_scaling is not None:
            q *= llama_4_scaling
        out = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._indexer_and_attn(hidden_states, q_c, positions, q, kv_c, k_pe, out)
        if self.use_hpc_gated_mla:
            assert hidden_states.is_contiguous() and out.is_contiguous()
            out = hpc_gated_mla_gemm(hidden_states, self.linear_gate.weight, out)
        else:
            out = fused_sigmoid_gate_mul(out, self.linear_gate(hidden_states)[0])
        out = self.o_proj(out)[0]
        if self.post_proj_norm is None:
            return out
        return fused_allreduce_rms_norm(out, None, self.post_proj_norm)[0]
