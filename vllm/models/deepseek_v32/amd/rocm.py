# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._aiter_ops import rocm_aiter_ops
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


# The aiter sparse kernels also handle 16 and 32, so extend rather than replace the
# base declaration -- narrowing it to [16, 32] made select_common_block_size silently
# downgrade a requested 64 to 32 via its largest-divisor fallback.
class DeepseekV32MLASparseBackend(ROCMAiterMLASparseBackend):
    """Sparse MLA backend with the extra block sizes the aiter kernels accept."""

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        """Base sizes plus the 16 and 32 the aiter kernels also handle."""
        return list(
            set(ROCMAiterMLASparseBackend.get_supported_kernel_block_sizes() + [16, 32])
        )


class DeepseekV32ROCmIndexerBackend(DeepseekV32IndexerBackend):
    """Indexer backend with the extra block sizes the aiter kernels accept."""

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        """Base sizes plus the 16 and 32 the aiter kernels also handle."""
        return list(
            set(DeepseekV32IndexerBackend.get_supported_kernel_block_sizes() + [16, 32])
        )


class DeepseekV32ROCmIndexerCache(DeepseekV32IndexerCache):
    """Indexer K cache that reports aiter's shuffled layout above block size 1."""

    def get_attn_backend(self):
        """Indexer backend for this cache."""
        return DeepseekV32ROCmIndexerBackend

    @property
    def uses_shuffled_layout(self) -> bool:
        """Whether the indexer K cache uses aiter's shuffled layout."""
        # aiter's gather/insert pair shuffles the cache above block size 1:
        # [n_blocks, blk/16, head_dim/16, 16, 16] instead of [n_blocks, blk, head_dim].
        return self.kv_cache.ndim == 3 and self.kv_cache.shape[1] != 1


class DeepseekV32ROCmIndexer(DeepseekV32Indexer):
    """Indexer wired to the ROCm K cache."""

    indexer_cache_cls = DeepseekV32ROCmIndexerCache


class DeepseekV32MLAAttention(DeepseekV32Attention):
    """DeepSeek-V3.2 / GLM-5.2 sparse MLA attention on ROCm."""

    indexer_cls = DeepseekV32ROCmIndexer

    def __init__(
        self,
        vllm_config,
        config,
        prefix,
        topk_indices_buffer=None,
        q_c_norm_buffer=None,
        kv_c_norm_buffer=None,
        mqa_q_buffer=None,
        q_index_buffer=None,
        index_weights_buffer=None,
    ):
        """Wire the ROCm sparse backend, indexer op and fusion toggles."""
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
        # Buffers are non-None only when the model enabled the aiter path; those
        # ops also need the packed fp8 q_out and cannot address fp8_ds_mla.
        self._use_aiter_qk_norm_rope = (
            q_c_norm_buffer is not None
            and kv_c_norm_buffer is not None
            and mqa_q_buffer is not None
            and q_index_buffer is not None
            and index_weights_buffer is not None
            and self._fp8_kv_needs_view
        )
        self._q_c_norm_buffer = q_c_norm_buffer
        self._kv_c_norm_buffer = kv_c_norm_buffer
        self._mqa_q_buffer = mqa_q_buffer
        self._q_index_buffer = q_index_buffer
        self._index_weights_buffer = index_weights_buffer
        # Set by the model via set_aiter_rope once the layers exist, so the
        # contiguous halves are split once rather than per layer.
        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None
        self._index_cos: torch.Tensor | None = None
        self._index_sin: torch.Tensor | None = None

    @property
    def _active_indexer(self) -> DeepseekV32Indexer | None:
        """The indexer when it should run this forward, else None."""
        # skip_topk is flipped at runtime by the MTP proposer (see mtp.py
        # set_skip_topk), so this cannot be cached into a flag.
        return None if self.skip_topk else self.indexer

    def set_aiter_rope(
        self,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        index_cos: torch.Tensor,
        index_sin: torch.Tensor,
    ) -> None:
        """Adopt the model's contiguous cos/sin halves for the aiter path."""
        self._rope_cos = rope_cos
        self._rope_sin = rope_sin
        self._index_cos = index_cos
        self._index_sin = index_sin

    def get_layer_forward_context(self):
        """This layer's (attn_metadata, mla_slot) view of the forward context."""
        forward_context = get_forward_context()
        raw = forward_context.attn_metadata
        if isinstance(raw, dict):
            attn_metadata = raw.get(self.layer_name)
        elif isinstance(raw, list):
            attn_metadata = raw[0].get(self.layer_name)
        else:
            attn_metadata = raw
        assert isinstance(forward_context.slot_mapping, dict)
        return attn_metadata, forward_context.slot_mapping.get(self.layer_name)

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """A-projections and indexer K weights, then fused attention and o_proj."""
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if (active_indexer := self._active_indexer) is not None:
            kw = active_indexer.wk_weights_proj(hidden_states)[0]
            index_k = kw[:, : active_indexer.head_dim]
            index_weights = kw[:, active_indexer.head_dim :]
        else:
            index_k = None
            index_weights = None

        num_tokens = hidden_states.shape[0]
        output = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._fused_attention(
            positions, q_c, kv_c, k_pe, index_k, index_weights, output
        )
        return self.o_proj(output)[0]

    def _norm_rope_and_cache(
        self,
        positions: torch.Tensor,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor | None,
        mla_slot: torch.Tensor | None,
        attn_metadata,
    ) -> torch.Tensor:
        """Norm q_c/kv_c, RoPE k_pe, cache [kv_c; k_pe] + indexer K -> normed q_c."""
        active_indexer = self._active_indexer
        # A profiling run has no caches allocated, so skip every per-token write.
        has_caches = attn_metadata is not None
        return fused_norm_rope(
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
            active_indexer.k_norm.weight if active_indexer is not None else None,
            active_indexer.k_norm.bias if active_indexer is not None else None,
            active_indexer.k_norm.eps if active_indexer is not None else 1e-6,
            self.indexer_rope_emb.cos_sin_cache if active_indexer is not None else None,
            self.topk_indices_buffer,
            slot_mapping=mla_slot if has_caches else None,
            indexer_k_cache=(
                active_indexer.k_cache.kv_cache
                if active_indexer is not None and has_caches
                else None
            ),
            indexer_cache_shuffled=(
                active_indexer.k_cache.uses_shuffled_layout
                if active_indexer is not None
                else False
            ),
            mla_kv_cache=self.kv_cache if has_caches else None,
            mla_kv_cache_dtype=self.kv_cache_dtype,
            mla_k_scale=self._k_scale if has_caches else None,
            has_indexer=active_indexer is not None,
            index_rope_interleave=self._index_rope_interleave,
        )

    def _project_q_index(self, q_c: torch.Tensor) -> torch.Tensor | None:
        """Indexer query projection [T, q_lora] -> [T, n_head, head_dim], or None."""
        active_indexer = self._active_indexer
        if active_indexer is None:
            return None
        q_index = active_indexer.wq_b(q_c)[0]  # [T, q_lora] -> [T, n_head*head_dim]
        return q_index.view(-1, active_indexer.n_head, active_indexer.head_dim)

    def _rope_and_pack_queries(
        self,
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        ql_nope: torch.Tensor,
        q_index: torch.Tensor | None,
        index_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """RoPE q_pe/q_index, quant q_index, pack [ql_nope; q_pe] into mqa_q."""
        active_indexer = self._active_indexer
        return fused_q(
            positions,
            q_pe,
            self.rotary_emb.cos_sin_cache,
            q_index,
            self.indexer_rope_emb.cos_sin_cache if active_indexer is not None else None,
            ql_nope,
            self._q_scale,
            index_weights,
            active_indexer.softmax_scale if active_indexer is not None else 0.0,
            active_indexer.n_head**-0.5 if active_indexer is not None else 0.0,
            has_indexer=active_indexer is not None,
            index_rope_interleave=self._index_rope_interleave,
            quantize_mqa=self._fp8_kv,
        )

    def _compute_w_uk_absorbed_ql_nope_and_q_pe(
        self, q_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project q_c per-head, split nope/rope, absorb W_UK into nope."""
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)  # [T, N, nope] -> [N, T, nope]

        # W_UK absorb: [N, T, nope] x [N, nope, kv_lora] -> [T, N, kv_lora]
        if self.is_aiter_triton_fp4_bmm_enabled:
            ql_nope = rocm_aiter_ops.batched_gemm_a16wfp4(
                q_nope, self.W_K, self.W_K_scale, transpose_bm=True, prequant=True
            )
        elif self.is_aiter_triton_fp8_bmm_enabled:
            ql_nope = rocm_aiter_ops.triton_fp8_bmm(
                q_nope, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
            )
        else:
            ql_nope = torch.bmm(q_nope, self.W_UK_T).transpose(0, 1)

        return ql_nope, q_pe

    def _run_indexer(
        self,
        q_c: torch.Tensor,
        q_index_fp8: torch.Tensor | None,
        index_weights_out: torch.Tensor | None,
    ) -> None:
        """Run the ROCm sparse indexer (forward_hip) if this layer has an indexer."""
        if self.indexer_op is not None:
            self.indexer_op.forward_hip(q_c, q_index_fp8, None, index_weights_out)

    def _select_q_for_forward_mqa(
        self,
        ql_nope: torch.Tensor,
        mqa_q: torch.Tensor,
        num_actual: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Slice Q to num_actual and pick the packed fp8 or bf16 tuple form."""
        if self._fp8_kv:
            # fp8 KV: mqa_q is the full [ql_nope; q_pe] packed as fp8.
            return mqa_q[:num_actual]

        return (ql_nope[:num_actual], mqa_q[:num_actual])

    def _absorb_w_uv_into_output(
        self,
        attn_out: torch.Tensor,
        output: torch.Tensor,
        num_actual: int,
    ) -> None:
        """Absorb W_UV into attn_out, writing [T, N*v_head_dim] into output."""
        x = attn_out.view(
            num_actual, self.num_local_heads, self.kv_lora_rank
        ).transpose(0, 1)  # [T, N, kv_lora] -> [N, T, kv_lora]
        out_view = output[:num_actual].view(
            num_actual, self.num_local_heads, self.v_head_dim
        )

        if self.is_aiter_triton_fp4_bmm_enabled:
            rocm_aiter_ops.batched_gemm_a16wfp4(
                x, self.W_V, self.W_V_scale, out_view, transpose_bm=True, prequant=True
            )
        elif self.is_aiter_triton_fp8_bmm_enabled:
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

    def _prepare_attn_inputs_for_common_triton(
        self,
        positions: torch.Tensor,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        mla_slot: torch.Tensor | None,
        attn_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared-Triton path: norm/RoPE/cache writes, then the MQA/indexer queries."""
        q_c = self._norm_rope_and_cache(
            positions, q_c, kv_c, k_pe, index_k, mla_slot, attn_metadata
        )
        ql_nope, q_pe = self._compute_w_uk_absorbed_ql_nope_and_q_pe(q_c)
        q_index = self._project_q_index(q_c)
        q_index_fp8, index_weights_out, mqa_q = self._rope_and_pack_queries(
            positions, q_pe, ql_nope, q_index, index_weights
        )
        return q_c, ql_nope, mqa_q, q_index_fp8, index_weights_out

    def _prepare_attn_inputs_for_aiter(
        self,
        positions: torch.Tensor,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        mla_slot: torch.Tensor | None,
        attn_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """AITER path: dual RMSNorm, indexer QK RoPE/quant, fused QK RoPE + cache."""
        assert self._rope_cos is not None, "set_aiter_rope was never called"
        active_indexer = self._active_indexer
        has_caches = attn_metadata is not None

        num_tokens = q_c.shape[0]
        q_c_normed = self._q_c_norm_buffer[:num_tokens]
        kv_c_normed = self._kv_c_norm_buffer[:num_tokens]
        rocm_aiter_ops.get_fused_mla_dual_rms_norm_out_op()(
            q_c_normed,
            kv_c_normed,
            q_c,
            self.q_a_layernorm.weight,
            kv_c,
            self.kv_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
            self.kv_a_layernorm.variance_epsilon,
        )
        q_c, kv_c = q_c_normed, kv_c_normed
        ql_nope, q_pe = self._compute_w_uk_absorbed_ql_nope_and_q_pe(q_c)
        q_index = self._project_q_index(q_c)

        # Unread placeholders: _run_indexer is guarded on the indexer being active.
        q_index_fp8 = self._q_index_buffer[:0]
        index_weights_out = self._index_weights_buffer[:0]
        if active_indexer is not None:
            assert q_index is not None and index_weights is not None
            assert self.topk_indices_buffer is not None
            # fused_norm_rope seeds -1 only on indexer layers; shared layers
            # deliberately reuse the previous indexer layer's top-k.
            self.topk_indices_buffer[: positions.shape[0]].fill_(-1)
            q_index_fp8 = self._q_index_buffer[:num_tokens]
            index_weights_out = self._index_weights_buffer[:num_tokens]
            if has_caches:
                rocm_aiter_ops.get_indexer_qk_rope_quant_and_cache_op()(
                    q_index,
                    q_index_fp8,
                    index_weights,
                    index_weights_out,
                    index_k,
                    active_indexer.k_cache.kv_cache,
                    mla_slot,
                    active_indexer.k_norm.weight,
                    active_indexer.k_norm.bias,
                    positions,
                    self._index_cos,
                    self._index_sin,
                    active_indexer.k_norm.eps,
                    active_indexer.quant_block_size,
                    active_indexer.scale_fmt,
                    active_indexer.softmax_scale * active_indexer.n_head**-0.5,
                    active_indexer.k_cache.uses_shuffled_layout,
                    self.indexer_rope_emb.is_neox_style,
                )
            else:
                # Profiling run: the op early-returns on slot < 0, so seed what
                # _run_indexer reads rather than leave the previous layer's rows.
                q_index_fp8.zero_()
                index_weights_out.zero_()

        mqa_q = self._mqa_q_buffer[: ql_nope.shape[0]]
        if has_caches:
            rocm_aiter_ops.get_fused_qk_rope_concat_and_cache_mla_op()(
                ql_nope,
                q_pe,
                kv_c,
                k_pe,
                # aiter rejects the raw uint8 cache; the enable gate guarantees fp8.
                self.kv_cache.view(torch.float8_e4m3fn),
                mqa_q,
                mla_slot,
                self._k_scale,
                self._q_scale,
                positions,
                self._rope_cos,
                self._rope_sin,
                self.rotary_emb.is_neox_style,
                True,  # MLA packs [ql_nope; q_pe] nope-first
            )
        return q_c, ql_nope, mqa_q, q_index_fp8, index_weights_out

    def _prepare_attn_inputs(
        self,
        positions: torch.Tensor,
        q_c: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        mla_slot: torch.Tensor | None,
        attn_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route query/cache preparation to the selected fusion path."""
        if self._use_aiter_qk_norm_rope:
            return self._prepare_attn_inputs_for_aiter(
                positions,
                q_c,
                kv_c,
                k_pe,
                index_k,
                index_weights,
                mla_slot,
                attn_metadata,
            )
        return self._prepare_attn_inputs_for_common_triton(
            positions, q_c, kv_c, k_pe, index_k, index_weights, mla_slot, attn_metadata
        )

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
        """Eager region: caches, queries, sparse indexer, MQA attention, W_UV."""
        attn_metadata, mla_slot = self.get_layer_forward_context()

        q_c, ql_nope, mqa_q, q_index_fp8, index_weights_out = self._prepare_attn_inputs(
            positions, q_c, kv_c, k_pe, index_k, index_weights, mla_slot, attn_metadata
        )

        if self._active_indexer is not None:
            self._run_indexer(q_c, q_index_fp8, index_weights_out)

        if attn_metadata is None:
            output.zero_()
            return

        num_actual = attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
        kv_cache = self.kv_cache
        if self._fp8_kv_needs_view:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)

        q_for_attn = self._select_q_for_forward_mqa(ql_nope, mqa_q, num_actual)
        attn_out, _ = self.impl.forward_mqa(  # type: ignore[attr-defined]
            q_for_attn, kv_cache, attn_metadata, self
        )

        self._absorb_w_uv_into_output(attn_out, output, num_actual)
