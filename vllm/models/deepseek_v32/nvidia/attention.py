# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import is_quantized_kv_cache

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadata,
    )


class DeepseekV32Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallel, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcasted to BF16 during loading to keep this fused.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        # fp8 naive cache: value in fp8 + fp32 scale per quant_block_size element.
        assert cache_config is not None, "DeepSeek V3.2 indexer requires cache_config"
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size,
        )

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        # Fused wk + weights_proj: one GEMM, then split.
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # RoPE (NeoX) can introduce extra leading dims; reshape back to flat.
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # Only quant q here; k quant is fused with cache insertion.
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return self.indexer_op(hidden_states, q_fp8, k, weights)


class DeepseekV32Attention(MLAAttention):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        hidden_size = config.hidden_size
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        q_lora_rank = config.q_lora_rank
        kv_lora_rank = config.kv_lora_rank
        num_heads = config.num_attention_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        num_local_heads = num_heads // tp_size
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        scaling = qk_head_dim**-0.5
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # DSA checkpoints may use plain ("default") or yarn-scaled RoPE.
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            scaling = scaling * mscale * mscale

        # DSA "shared indexer" pattern: only some layers carry an indexer; the
        # rest reuse the top-k written by the previous indexer layer into the
        # shared topk_indices_buffer. DeepSeek-V3.2 builds it on every layer
        # (index_topk_freq defaults to 1); GLM-5.2 uses index_topk_freq=4 so
        # only layers [0,1,2,6,10,...] (+ MTP) carry one.
        layer_id = extract_layer_index(prefix)
        index_topk_freq = getattr(config, "index_topk_freq", 1)
        index_topk_pattern = getattr(config, "index_topk_pattern", None)
        index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
        if index_topk_pattern is None:
            skip_topk = (
                max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
            )
        elif 0 <= layer_id < len(index_topk_pattern):
            skip_topk = index_topk_pattern[layer_id] == "S"
        else:
            skip_topk = False
        # MTP/nextn layers always build a full indexer (they toggle at runtime).
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = num_hidden_layers is not None and layer_id >= num_hidden_layers

        # Build kv_b_proj + indexer first; they are passed to MLAAttention.__init__
        # (which runs nn.Module.__init__ and registers them).
        kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        indexer = None
        if not skip_topk or is_mtp_layer:
            indexer = DeepseekV32Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                prefix=f"{prefix}.indexer",
            )

        # Set up the MLA engine (impl, KV cache, scales, backend, registration,
        # and process_weights_after_loading) via the MLAAttention base.
        super().__init__(
            num_heads=num_local_heads,
            scale=scaling,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=True,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.num_local_heads = num_local_heads
        self.qk_head_dim = qk_head_dim
        self.indexer = indexer
        # Runtime toggle for index_share_for_mtp_iteration: MTP draft step 0
        # computes the top-k, steps 1+ set this True to reuse it.
        self.skip_topk = False
        # Whether the paged KV cache must be viewed as fp8 before the attention
        # (per-tensor fp8; the fp8_ds_mla layout is read as uint8).
        self._fp8_kv_needs_view = (
            is_quantized_kv_cache(self.kv_cache_dtype)
            and self.kv_cache_dtype != "fp8_ds_mla"
        )
        # Whether the backend takes an fp8-quantized query (FlashInfer sparse)
        # vs the (ql_nope, q_pe) tuple (FlashMLA sparse).
        self._use_concat_quant = (
            is_quantized_kv_cache(self.kv_cache_dtype)
            and self.impl.supports_quant_query_input
        )

        # Remaining MLA projections (registered on this module).
        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        # Lightning indexer uses its own RoPE; interleave maps to non-NeoX.
        self.indexer_rope_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
        )

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        num_tokens = hidden_states.shape[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)  # (N, B, P)
        ql_nope = torch.bmm(q_nope, self.W_UK_T).transpose(0, 1)  # (B, N, L)

        # Lightning indexer writes the top-k indices into the shared buffer.
        # "Shared" layers (indexer is None) reuse the top-k from the previous
        # indexer layer already sitting in the buffer.
        if self.indexer is not None and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)  # type: ignore[operator]

        attn_latent = torch.empty(
            (num_tokens, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        self._sparse_attention(kv_c_normed, k_pe, ql_nope, q_pe, attn_latent)

        # V up-projection + output projection are metadata-independent GEMMs and
        # stay captured.
        output = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        self._v_up_proj(attn_latent, out=output)
        return self.o_proj(output)[0]

    @eager_break_during_capture
    def _sparse_attention(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_latent: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata: MLACommonMetadata | None
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw[self.layer_name]  # type: ignore[assignment]
        elif isinstance(attn_metadata_raw, list):
            # Speculative decoding: [0] is the base-model metadata dict.
            attn_metadata = attn_metadata_raw[0][self.layer_name]  # type: ignore[assignment]
        else:
            attn_metadata = attn_metadata_raw

        slot_mapping = forward_context.slot_mapping
        assert isinstance(slot_mapping, dict)
        self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            kv_c_normed,
            k_pe,
            self.kv_cache,
            slot_mapping.get(self.layer_name),
            self.kv_cache_dtype,
            self._k_scale,
        )

        if attn_metadata is None:
            # Profile / warmup: zero-fill for DP+EP determinism.
            attn_latent.zero_()
            return

        num_actual = attn_metadata.num_actual_tokens
        kv_cache = self.kv_cache
        if self._fp8_kv_needs_view:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)

        ql_nope = ql_nope[:num_actual]
        q_pe = q_pe[:num_actual]
        # FlashInfer sparse takes a single fp8-quantized query; FlashMLA sparse
        # takes the (ql_nope, q_pe) tuple and concatenates internally.
        mqa_q: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        if self._use_concat_quant:
            mqa_q = self._decode_concat_quant_fp8_op(ql_nope, q_pe, self._q_scale)
        else:
            mqa_q = (ql_nope, q_pe)

        attn_out, _ = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)  # type: ignore[attr-defined]
        attn_latent[:num_actual] = attn_out.view(
            num_actual, self.num_local_heads, self.kv_lora_rank
        )
