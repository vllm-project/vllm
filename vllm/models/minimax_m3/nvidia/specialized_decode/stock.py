# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every stage on the ops the stock layers run: the stock-op chain.

Each stage below runs exactly the ops `MiniMaxM3DecoderLayer` runs for a pure
decode step of up to `MAX_ROWS` rows, in the same order and with the same
arguments, split at the stage boundaries of `stages.py`. Run alone, the chain
reproduces the stock decode; it is also the last implementation of every
stage, which a kernel stage falls back to for the steps it does not support.

Two places split one stock call in two:

- `MiniMaxM3IndexerMSAImpl.forward` is `StockIndexScore` (its decode
  branch) followed by `StockPageTopK` (its unified top-k); executor steps
  have no prefill rows, so its prefill branch never runs.
- `MiniMaxM3MoE.forward` computes the router logits, then runs the experts;
  `StockAllReduceNorm.post_attention` computes the logits and `StockMoE`
  the experts.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

from .stages import (
    AllReduceNormStage,
    BoundaryOutput,
    DenseDecodeStage,
    DenseMLPStage,
    FrontOutput,
    FrontStage,
    IndexScoreStage,
    MoEStage,
    OProjStage,
    PageTopKStage,
    SparseDecodeStage,
    Stages,
)

if TYPE_CHECKING:
    from ..model import (
        MiniMaxM3Attention,
        MiniMaxM3DecoderLayer,
        MiniMaxM3MLP,
        MiniMaxM3Model,
        MiniMaxM3MoE,
        MiniMaxM3SparseAttention,
    )
    from .stages import LayerInput, PendingFFN


class StockFront(FrontStage):
    """`MiniMaxM3SparseAttention.forward` / `MiniMaxM3Attention.forward`
    up to the attention: `qkv_proj` and
    `fused_minimax_m3_qknorm_rope_kv_insert`. Executor steps stay below the
    fused qkv epilogue's 512 rows, so it never runs."""

    def run(
        self,
        attn: "MiniMaxM3SparseAttention | MiniMaxM3Attention",
        hidden: "LayerInput",
        positions: torch.Tensor,
    ) -> FrontOutput:
        from ..model import MiniMaxM3SparseAttention

        if not isinstance(attn, MiniMaxM3SparseAttention):
            return self._dense(attn, hidden, positions)
        if isinstance(hidden, QuantizedActivation):
            num_tokens = hidden.orig_shape[0]
            dtype = hidden.orig_dtype
        else:
            num_tokens = hidden.shape[0]
            dtype = hidden.dtype
        qkv, _ = attn.qkv_proj(hidden)
        return self.run_qkv(attn, qkv, num_tokens, dtype, positions)

    @staticmethod
    def run_qkv(
        attn: "MiniMaxM3SparseAttention",
        qkv: torch.Tensor,
        num_tokens: int,
        dtype: torch.dtype,
        positions: torch.Tensor,
    ) -> FrontOutput:
        """The sparse front after `qkv_proj`: q/k norms, RoPE and the KV /
        index-K cache insert of `qkv` (a kernel QKV projection reuses it)."""
        slot_mappings = get_forward_context().slot_mapping
        assert isinstance(slot_mappings, dict)
        main_slot_mapping = slot_mappings[attn.layer_name]
        index_slot_mapping = slot_mappings[attn.indexer.index_cache.prefix]
        q = attn._allocate_query(num_tokens, dtype, qkv.device)
        q_is_fp8 = q.dtype == torch.float8_e4m3fn
        index_q = torch.empty(
            (num_tokens, attn.index_q_size),
            dtype=attn.indexer.index_cache.query_dtype,
            device=qkv.device,
        )
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.rotary_emb.cos_sin_cache,
            positions,
            attn.num_heads,
            attn.num_kv_heads,
            attn.rotary_emb.rotary_dim,
            attn.q_norm.variance_epsilon,
            attn.index_q_norm.weight,
            attn.index_k_norm.weight,
            attn.num_idx_heads,
            main_slot_mapping,
            index_slot_mapping,
            attn.kv_cache,
            attn.indexer.index_cache.kv_cache,
            attn.kv_cache.size(2),  # paged-cache block size
            None if q_is_fp8 else q,
            index_q,
            attn.kv_cache_dtype,
            q_fp8_out=q if q_is_fp8 else None,
            q_fp8_scale=attn._q_scale_float,
            kv_k_scale=attn._k_scale if attn.use_nvfp4_kv else None,
            kv_v_scale=attn._v_scale if attn.use_nvfp4_kv else None,
            write_qkv=not q_is_fp8,
        )
        return FrontOutput(q=q, index_q=index_q)

    @staticmethod
    def _dense(
        attn: "MiniMaxM3Attention", hidden: "LayerInput", positions: torch.Tensor
    ) -> FrontOutput:
        qkv, _ = attn.qkv_proj(hidden)
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.rotary_emb.cos_sin_cache,
            positions,
            attn.num_heads,
            attn.num_kv_heads,
            attn.rotary_emb.rotary_dim,
            attn.q_norm.variance_epsilon,
            kv_cache_dtype="auto",
        )
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        return FrontOutput(q=q, k=k, v=v)


class StockIndexScore(IndexScoreStage):
    """The decode branch of `MiniMaxM3IndexerMSAImpl.forward`: the CuTe DSL
    (or Triton, for wide query tiles) decode score into the unified score
    buffer."""

    def run(
        self, attn: "MiniMaxM3SparseAttention", index_q: torch.Tensor
    ) -> torch.Tensor:
        from vllm.models.minimax_m3.common.ops.index_topk import (
            minimax_m3_index_decode_score,
        )
        from vllm.models.minimax_m3.nvidia.ops import (
            minimax_m3_index_decode_score_cutedsl,
        )

        impl = attn.indexer.impl
        md = _indexer_metadata(attn)
        num_tokens = md.num_actual_tokens
        nd = md.num_decode_tokens
        d = md.decode
        assert d is not None and nd == num_tokens, "executor steps are pure decode"
        unified_scores = md.unified_scores
        assert unified_scores is not None
        q = index_q[:num_tokens].view(-1, impl.num_index_heads, impl.index_head_dim)
        decode_score = (
            minimax_m3_index_decode_score_cutedsl
            if impl.num_index_heads * d.max_decode_query_len <= 32
            else minimax_m3_index_decode_score
        )
        decode_score(
            q[:nd],
            impl.index_cache.kv_cache,
            d.block_table,
            d.seq_lens,
            d.max_seq_len,
            impl.init_blocks,
            impl.local_blocks,
            impl.num_kv_heads,
            d.decode_query_len,
            d.max_decode_query_len,
            score_out=unified_scores[:nd].transpose(0, 1),
        )
        return unified_scores


class StockPageTopK(PageTopKStage):
    """The unified top-k of `MiniMaxM3IndexerMSAImpl.forward`:
    `fmha_sm100.sparse_topk_select` into the shared top-k buffer."""

    def run(
        self, attn: "MiniMaxM3SparseAttention", scores: torch.Tensor
    ) -> torch.Tensor:
        from vllm.third_party.fmha_sm100.api import sparse_topk_select

        impl = attn.indexer.impl
        md = _indexer_metadata(attn)
        num_tokens = md.num_actual_tokens
        buf = impl.topk_indices_buffer
        assert buf is not None
        sparse_topk_select(
            scores,
            impl.topk_blocks,
            num_valid_pages=md.topk_num_valid_pages,
            force_begin_blocks=impl.init_blocks,
            force_end_blocks=impl.local_blocks,
            output=buf[:num_tokens],
            max_score_layout="THK",
        )
        return buf[:num_tokens]


class StockSparseDecode(SparseDecodeStage):
    """`attn.impl.forward`: the MSA sparse decode (CUTLASS, or Triton where
    CUTLASS has no plan) over the shared top-k buffer."""

    def run(
        self,
        attn: "MiniMaxM3SparseAttention",
        q: torch.Tensor,
        topk: torch.Tensor,
    ) -> torch.Tensor:
        # The stock impl reads the top-k from attn.topk_indices_buffer, which ``topk``
        # views.
        assert attn.topk_indices_buffer is not None
        assert topk.data_ptr() == attn.topk_indices_buffer.data_ptr()
        output = torch.empty(
            (q.shape[0], attn.q_size), dtype=torch.bfloat16, device=q.device
        )
        return attn.impl.forward(attn, q, attn.kv_cache, output)


class StockDenseDecode(DenseDecodeStage):
    """`MiniMaxM3Attention.attn`: the platform's attention backend."""

    def run(self, attn: "MiniMaxM3Attention", front: FrontOutput) -> torch.Tensor:
        return attn.attn(front.q, front.k, front.v)


class StockOProj(OProjStage):
    def run(
        self,
        attn: "MiniMaxM3SparseAttention | MiniMaxM3Attention",
        attn_out: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = attn.o_proj(attn_out)
        return output


class StockAllReduceNorm(AllReduceNormStage):
    """`forward_input_norm`, the post-attention
    `fused_allreduce_gemma_rms_norm` (+ the MoE router) and the final norm
    of `MiniMaxM3Model.forward`."""

    def input(
        self,
        layer: "MiniMaxM3DecoderLayer",
        pending: "PendingFFN",
        residual: torch.Tensor | None,
    ) -> BoundaryOutput:
        hidden, residual = layer.forward_input_norm(pending, residual)
        return BoundaryOutput(hidden=hidden, residual=residual)

    def post_attention(
        self,
        layer: "MiniMaxM3DecoderLayer",
        partial: torch.Tensor,
        residual: torch.Tensor,
    ) -> BoundaryOutput:
        hidden, residual = fused_allreduce_gemma_rms_norm(
            partial, residual, layer.post_attention_layernorm
        )
        router_logits = None
        if layer.is_moe_layer:
            router_logits, _ = layer.block_sparse_moe.gate(hidden)
        return BoundaryOutput(
            hidden=hidden, residual=residual, router_logits=router_logits
        )

    def final(
        self,
        model: "MiniMaxM3Model",
        pending: "PendingFFN",
        residual: torch.Tensor,
    ) -> BoundaryOutput:
        # The last layer's MoE keeps its own finalize (the final norm feeds the LM head
        # in BF16), so nothing is pending a finalize here.
        assert isinstance(pending, torch.Tensor)
        if model.fuse_final_norm_allreduce:
            hidden, residual = fused_allreduce_gemma_rms_norm(
                pending, residual, model.norm
            )
        else:
            hidden, residual = model.norm(pending, residual)
        return BoundaryOutput(hidden=hidden, residual=residual)


class StockMoE(MoEStage):
    """`MiniMaxM3MoE.forward` (or `forward_unfinalized`) after the router:
    the shared expert and the routed experts on the configured NVFP4 MoE
    backend."""

    def run(
        self,
        moe: "MiniMaxM3MoE",
        hidden: torch.Tensor,
        router_logits: torch.Tensor | None,
        defer_finalize: bool,
        push: bool = False,
    ) -> "PendingFFN":
        # can_push is False: the executor never asks the stock MoE to push.
        assert not push, "the stock MoE does not push"
        if router_logits is None:
            router_logits, _ = moe.gate(hidden)
        if defer_finalize or moe.defers_finalize(hidden.shape[0]):
            # As forward_unfinalized: the runner's custom op returns tensors only, so
            # run its body directly.
            shared_output, routed = _unpack(
                moe.experts._forward_impl(hidden, router_logits, hidden)
            )
            assert shared_output is not None
            assert isinstance(routed, UnfinalizedMoEOutput)
            out = MoEOutput(routed=routed, shared_output=shared_output)
            if defer_finalize:
                return out
            # The experts defer at these rows, but the consumer (a kernel front fusing
            # the next input boundary) takes a finalized BF16 partial: the top-k
            # reduction and the shared add here.
            return _finalize_moe_output(out, hidden.shape[0])
        return moe.experts(hidden_states=hidden, router_logits=router_logits)


def _finalize_moe_output(out: MoEOutput, rows: int) -> torch.Tensor:
    """The un-reduced BF16 FFN partial of an unfinalized MoE output: the
    weighted top-k sum of the GEMM2 rows (fp32), times the routed scale, plus
    the shared expert."""
    r = out.routed
    assert isinstance(r, UnfinalizedMoEOutput)
    idx = r.expanded_idx_to_permuted_idx.reshape(rows, -1).long()
    w = r.expert_weights.reshape(rows, -1).float().masked_fill(idx < 0, 0.0)
    g = r.gemm2_permuted.index_select(0, idx.clamp(min=0).reshape(-1)).float()
    x = (g.view(rows, idx.shape[1], -1) * w.unsqueeze(-1)).sum(1)
    x = x * out.routed_scaling_factor
    if out.shared_output is not None:
        x = x + out.shared_output.float()
    return x.to(torch.bfloat16)


class StockDenseMLP(DenseMLPStage):
    def run(self, mlp: "MiniMaxM3MLP", hidden: torch.Tensor) -> torch.Tensor:
        return mlp(hidden)


def _indexer_metadata(attn: "MiniMaxM3SparseAttention"):
    from vllm.models.minimax_m3.nvidia.indexer_msa import (
        MiniMaxM3IndexerMSAMetadata,
    )

    attn_metadata = get_forward_context().attn_metadata
    assert isinstance(attn_metadata, dict)
    md = attn_metadata[attn.indexer.index_cache.prefix]
    assert isinstance(md, MiniMaxM3IndexerMSAMetadata)
    return md


def stock_stages() -> Stages:
    """The stock-op chain: one stock implementation per stage."""

    def one(stage) -> Sequence:
        return (stage,)

    return Stages(
        front=one(StockFront()),
        index_score=one(StockIndexScore()),
        page_topk=one(StockPageTopK()),
        sparse_decode=one(StockSparseDecode()),
        dense_decode=one(StockDenseDecode()),
        o_proj=one(StockOProj()),
        allreduce_norm=one(StockAllReduceNorm()),
        moe=one(StockMoE()),
        dense_mlp=one(StockDenseMLP()),
    )
