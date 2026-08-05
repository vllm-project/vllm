# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.deepseek_v32.attention import DeepseekV32Attention, DeepseekV32Indexer
from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
)


class DeepseekV32MLASparseBackend(ROCMAiterMLASparseBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        return [16, 32]


class DeepseekV32ROCmIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        return [16, 32]


class DeepseekV32ROCmIndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self):
        return DeepseekV32ROCmIndexerBackend


class DeepseekV32ROCmIndexer(DeepseekV32Indexer):
    indexer_cache_cls = DeepseekV32ROCmIndexerCache


class DeepseekV32MLAAttention(DeepseekV32Attention):
    require_fp8_kv_cache: bool = False
    indexer_cls = DeepseekV32ROCmIndexer

    def __init__(self, vllm_config, config, prefix, topk_indices_buffer=None):
        super().__init__(
            vllm_config,
            config,
            prefix,
            topk_indices_buffer,
            attn_backend=DeepseekV32MLASparseBackend,
        )

        self.indexer_op: SparseAttnIndexer | None = None
        if self.indexer is not None:
            self.indexer_op = SparseAttnIndexer(
                self.indexer.k_cache,
                self.indexer.quant_block_size,
                self.indexer.scale_fmt,
                self.indexer.topk_tokens,
                self.indexer.head_dim,
                self.indexer.max_model_len,
                self.indexer.max_total_seq_len,
                topk_indices_buffer,
                skip_k_cache_insert=True,
            )
        self._fp8_kv = is_quantized_kv_cache(self.kv_cache_dtype)
        self._fp8_kv_needs_view = self._fp8_kv and self.kv_cache_dtype != "fp8_ds_mla"

    def _compute_ql_nope(self, q_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)  # (N, tokens, P)

        if self.is_aiter_triton_fp4_bmm_enabled:
            from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

            ql_nope = batched_gemm_a16wfp4(
                q_nope, self.W_K, self.W_K_scale, transpose_bm=True, prequant=True
            )
        elif self.is_aiter_triton_fp8_bmm_enabled:
            from vllm._aiter_ops import rocm_aiter_ops

            ql_nope = rocm_aiter_ops.triton_fp8_bmm(
                q_nope, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
            )
        else:
            ql_nope = torch.bmm(q_nope, self.W_UK_T).transpose(0, 1)

        return ql_nope, q_pe

    def _run_indexer(
        self,
        q_c: torch.Tensor,
        index_q_fp8: torch.Tensor | None,
        index_weights_out: torch.Tensor | None,
    ) -> None:
        """Run the ROCm sparse indexer (forward_hip) if this layer has an indexer."""
        if self.indexer_op is not None:
            self.indexer_op.forward_hip(q_c, index_q_fp8, None, index_weights_out)

    def _build_q_for_attn(
        self,
        ql_nope: torch.Tensor,
        mqa_q: torch.Tensor,
        num_actual: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self._fp8_kv:
            # fp8 KV: mqa_q is the full [ql_nope; q_pe] packed as fp8.
            return mqa_q[:num_actual]

        return (ql_nope[:num_actual], mqa_q[:num_actual])

    def _compute_uv_out(
        self,
        attn_out: torch.Tensor,
        output: torch.Tensor,
        num_actual: int,
    ) -> None:
        x = attn_out.view(
            num_actual, self.num_local_heads, self.kv_lora_rank
        ).transpose(0, 1)  # (N, tokens, L)
        out_view = output[:num_actual].view(
            num_actual, self.num_local_heads, self.v_head_dim
        )

        if self.is_aiter_triton_fp4_bmm_enabled:
            from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

            batched_gemm_a16wfp4(
                x, self.W_V, self.W_V_scale, out_view, transpose_bm=True, prequant=True
            )
        elif self.is_aiter_triton_fp8_bmm_enabled:
            from vllm._aiter_ops import rocm_aiter_ops

            rocm_aiter_ops.triton_fp8_bmm(
                x,
                self.W_V,
                self.W_V_scale,
                group_size=128,
                transpose_bm=True,
                YQ=out_view,
            )
        else:
            torch.bmm(x, self.W_UV, out=out_view.transpose(0, 1))

    @eager_break_during_capture
    def _fused_attention(
        self,
        positions: torch.Tensor,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.layer_name)
        elif isinstance(attn_metadata_raw, list):
            attn_metadata = attn_metadata_raw[0].get(self.layer_name)
        else:
            attn_metadata = attn_metadata_raw

        slot_mapping = forward_context.slot_mapping
        assert isinstance(slot_mapping, dict)
        mla_slot = slot_mapping.get(self.layer_name)

        if self.indexer is not None:
            has_indexer = True
            indexer_k_norm_w = self.indexer.k_norm.weight
            indexer_k_norm_bias = self.indexer.k_norm.bias
            indexer_k_norm_eps = self.indexer.k_norm.eps
            indexer_k_rope_cos_sin_cache = self.indexer_rope_emb.cos_sin_cache
            indexer_k_cache = self.indexer.k_cache.kv_cache
            indexer_softmax_scale = self.indexer.softmax_scale
            indexer_n_head_scale = self.indexer.n_head**-0.5
        else:
            has_indexer = False
            indexer_k_norm_w = None
            indexer_k_norm_bias = None
            indexer_k_norm_eps = 1e-6
            indexer_k_rope_cos_sin_cache = None
            indexer_k_cache = None
            indexer_softmax_scale = 0.0
            indexer_n_head_scale = 0.0

        if attn_metadata is None:
            mla_kv_cache = None
            mla_k_scale = None
            indexer_k_cache = None
            mla_slot = None
        else:
            mla_kv_cache = self.kv_cache
            mla_k_scale = self._k_scale

        q_c = fused_norm_rope(
            positions,
            q_c,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
            kv_c,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon,
            k_pe,
            self.rotary_emb.cos_sin_cache,
            index_k,
            indexer_k_norm_w,
            indexer_k_norm_bias,
            indexer_k_norm_eps,
            indexer_k_rope_cos_sin_cache,
            self.topk_indices_buffer,
            slot_mapping=mla_slot,
            indexer_k_cache=indexer_k_cache,
            mla_kv_cache=mla_kv_cache,
            mla_kv_cache_dtype=self.kv_cache_dtype,
            mla_k_scale=mla_k_scale,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
        )

        ql_nope, q_pe = self._compute_ql_nope(q_c)

        if self.indexer is not None:
            index_q = self.indexer.wq_b(q_c)[0]
            index_q = index_q.view(-1, self.indexer.n_head, self.indexer.head_dim)
        else:
            index_q = None

        index_q_fp8, index_weights_out, mqa_q = fused_q(
            positions,
            q_pe,
            self.rotary_emb.cos_sin_cache,
            index_q,
            self.indexer_rope_emb.cos_sin_cache if has_indexer else None,
            ql_nope,
            self._q_scale,
            index_weights,
            indexer_softmax_scale,
            indexer_n_head_scale,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
            quantize_mqa=self._fp8_kv,
        )

        self._run_indexer(q_c, index_q_fp8, index_weights_out)

        if attn_metadata is None:
            output.zero_()
            return

        num_actual = attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
        kv_cache = self.kv_cache
        if self._fp8_kv_needs_view:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)

        q_for_attn = self._build_q_for_attn(ql_nope, mqa_q, num_actual)
        attn_out, _ = self.impl.forward_mqa(  # type: ignore[attr-defined]
            q_for_attn, kv_cache, attn_metadata, self
        )

        self._compute_uv_out(attn_out, output, num_actual)
