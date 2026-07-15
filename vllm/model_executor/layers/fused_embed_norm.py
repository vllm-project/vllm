# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated input embedding + its fused gather/norm kernels.

Groups the ``VLLM_REPLICATE_EMBED`` path in one place: the replicated embedding
module, its factory (with the fallback to vocab-parallel), and the two Triton
fusions the full on-rank table unlocks --

  * ``fused_embed_norm``: gather + a chained RMSNorm (e.g. the first decoder
    layer's ``input_layernorm``), and
  * ``fused_embed_eh_norm``: gather + pos-0 zeroing + enorm/hnorm + cat, the
    embed/previous-hidden input norm for a speculative (MTP/eagle) depth layer
    (the replicated-table analogue of the model-local ``fused_eh_norm``, which
    takes precomputed embeds).

Self-contained (no model-local imports) so it can live under ``layers/``.
"""

import torch

import vllm.envs as envs
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


class ReplicatedEmbedding(torch.nn.Embedding):
    """Fully-replicated input token embedding for GLM-5.2 / DeepSeek-V32.

    The full [num_embeddings, embedding_dim] table lives on every TP rank and
    the forward is a local lookup with NO all-reduce (unlike
    VocabParallelEmbedding, which shards the vocab and all-reduces the output).
    The full on-rank table also enables the fused gather+norm kernels
    (``fused_embed_norm`` / ``fused_eh_norm`` GATHER mode). The weight has no
    ``weight_loader`` attr, so it loads via ``default_weight_loader`` (a
    shape-checked full-tensor copy). The only addition over
    ``torch.nn.Embedding`` is the int32->int64 index cast that ``F.embedding``
    requires (vLLM feeds int32 input_ids).
    """

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return super().forward(input_.long())


def make_input_embedding(
    num_embeddings: int,
    embedding_dim: int,
    *,
    params_dtype: torch.dtype | None = None,
    quant_config=None,
    prefix: str = "",
    tie_word_embeddings: bool = False,
):
    """Input token embedding with an optional replicated escape hatch.

    With ``VLLM_REPLICATE_EMBED=1`` use a fully-replicated ``ReplicatedEmbedding``
    to unlock the fused gather+norm path (and, at TP>1, skip the embedding
    all-reduce), at the cost of a full table per rank at TP>1 (no extra memory at
    TP=1, where vocab-parallel is already unsharded). The replicated table is
    always the raw (unquantized) ``params_dtype``; ``quant_config`` is accepted
    only for the vocab-parallel fallback (embeddings are unquantized regardless).
    Assumes an untied embedding -- a replicated, unsharded table cannot be tied
    to a vocab-parallel ``ParallelLMHead``, so tied word embeddings are rejected.
    Otherwise falls back to the default TP-sharded ``VocabParallelEmbedding``
    (byte-identical to before).
    """
    if envs.VLLM_REPLICATE_EMBED:
        assert not tie_word_embeddings, (
            "VLLM_REPLICATE_EMBED is unsupported with tied word embeddings "
            "(the replicated table cannot tie to a vocab-parallel lm_head)"
        )
        emb = ReplicatedEmbedding(
            num_embeddings,
            embedding_dim,
            dtype=params_dtype or torch.get_default_dtype(),
        )
        emb.weight.requires_grad_(False)
        return emb
    return VocabParallelEmbedding(
        num_embeddings,
        embedding_dim,
        params_dtype=params_dtype,
        quant_config=quant_config,
        prefix=prefix,
    )


@triton.jit
def _fused_embed_norm_kernel(
    ids_ptr,  # [T] token ids
    table_ptr,  # [V, H] embedding table (full vocab, replicated on-rank)
    table_stride_0,
    out_ptr,  # [T, H] gathered embedding (the residual stream)
    normed_ptr,  # [T, H] rmsnorm(out, chain_w) (HAS_NORM only)
    chain_w_ptr,  # [H] next norm weight (HAS_NORM only)
    eps,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_NORM: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    off = tl.arange(0, BLOCK)
    mask = off < H
    row = tl.load(ids_ptr + tok).to(tl.int64)
    x = tl.load(table_ptr + row * table_stride_0 + off, mask=mask, other=0.0)
    tl.store(out_ptr + tok * H + off, x, mask=mask)
    if HAS_NORM:
        w = tl.load(chain_w_ptr + off, mask=mask)
        y = _rms_norm(x, w, eps, H).to(normed_ptr.dtype.element_ty)
        tl.store(normed_ptr + tok * H + off, y, mask=mask)


# Base model fusion
def fused_embed_norm(
    input_ids: torch.Tensor,
    embed_table: torch.Tensor,
    chain_weight: torch.Tensor | None = None,
    eps: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused embedding row gather (``embed_table[input_ids]``).

    Requires the full vocab on-rank (replicated embedding). When
    ``chain_weight`` is given, also emits ``rmsnorm(gathered, chain_weight)``
    (the first decoder layer's ``input_layernorm``) as a second output in the
    same launch, so the returned pair is ``(residual, normed_input)``. Bit-exact
    vs a plain gather followed by an ``RMSNorm``.
    """
    assert embed_table.ndim == 2, embed_table.shape
    ids = input_ids.view(-1)
    (t,) = ids.shape
    h = embed_table.shape[1]
    if chain_weight is not None:
        assert chain_weight.shape == (h,), (chain_weight.shape, h)
    out = torch.empty((t, h), dtype=embed_table.dtype, device=embed_table.device)
    normed = torch.empty_like(out) if chain_weight is not None else None
    if t > 0:
        block = triton.next_power_of_2(h)
        _fused_embed_norm_kernel[(t,)](
            ids,
            embed_table,
            embed_table.stride(0),
            out,
            normed if normed is not None else out,
            chain_weight if chain_weight is not None else embed_table,
            eps,
            h,
            block,
            HAS_NORM=chain_weight is not None,
            num_warps=min(32, max(4, block // 512)),
        )
    if normed is not None:
        return out, normed
    return out


@triton.jit
def _fused_embed_eh_norm_kernel(
    pos_ptr,
    ids_ptr,  # [T] token ids
    table_ptr,  # [V, H] embedding table (full vocab, replicated on-rank)
    table_stride,
    prev_ptr,  # [T, H] previous-step hidden
    prev_stride,
    enorm_w_ptr,
    hnorm_w_ptr,
    eps,
    out_ptr,  # [T, 2H]
    out_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MTP input fusion with a folded embedding gather: gather
    ``table[ids]``, zero it at position 0, RMSNorm(embed) with enorm and
    RMSNorm(prev_hidden) with hnorm, written side-by-side into ``out`` ([N, 2H])
    ready for the eh_proj GEMM. Replaces embedding lookup + where + 2x RMSNorm +
    cat. Requires the full table on-rank (replicated embedding)."""
    tok = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < H

    pos = tl.load(pos_ptr + tok)
    row = tl.load(ids_ptr + tok).to(tl.int64)
    e = tl.load(table_ptr + row * table_stride + off, mask=mask, other=0.0)
    e = tl.where(pos == 0, 0.0, e.to(tl.float32))
    ew = tl.load(enorm_w_ptr + off, mask=mask)
    e_normed = _rms_norm(e, ew, eps, H)
    tl.store(out_ptr + tok * out_stride + off, e_normed, mask=mask)

    p = tl.load(prev_ptr + tok * prev_stride + off, mask=mask, other=0.0)
    hw = tl.load(hnorm_w_ptr + off, mask=mask)
    p_normed = _rms_norm(p, hw, eps, H)
    tl.store(out_ptr + tok * out_stride + H + off, p_normed, mask=mask)


# MTP fusion
def fused_embed_eh_norm(
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    embed_table: torch.Tensor,
    previous_hidden: torch.Tensor,
    enorm_w: torch.Tensor,
    hnorm_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused ``cat([enorm(masked embed_table[ids]), hnorm(prev_hidden)])`` -> [N, 2H].

    Folds the embedding row gather into the MTP eh-norm launch; requires the full
    table on-rank (replicated embedding). Bit-exact vs gathering ``embed_table[
    input_ids]`` and passing it to the model-local ``fused_eh_norm``.
    """
    assert previous_hidden.ndim == 2 and embed_table.ndim == 2
    n, h = previous_hidden.shape
    assert positions.shape == (n,) and input_ids.view(-1).shape == (n,)
    assert embed_table.shape[1] == h, (embed_table.shape, h)
    assert enorm_w.shape == (h,) and hnorm_w.shape == (h,)
    out = torch.empty(
        n, 2 * h, dtype=previous_hidden.dtype, device=previous_hidden.device
    )
    _fused_embed_eh_norm_kernel[(n,)](
        positions,
        input_ids,
        embed_table,
        embed_table.stride(0),
        previous_hidden,
        previous_hidden.stride(0),
        enorm_w,
        hnorm_w,
        eps,
        out,
        out.stride(0),
        h,
        triton.next_power_of_2(h),
    )
    return out
