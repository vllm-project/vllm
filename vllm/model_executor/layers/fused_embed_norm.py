# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated input embedding + its fused gather/norm kernels.

Groups the ``VLLM_REPLICATE_EMBED`` path in one place: the embedding factory,
the predicate that says whether the fusions apply, and the two Triton fusions
the full on-rank table unlocks --

  * ``fused_embed_norm``: gather + a chained RMSNorm (e.g. the first decoder
    layer's ``input_layernorm``), and
  * ``fused_embed_eh_norm``: gather + pos-0 zeroing + enorm/hnorm + cat, the
    embed/previous-hidden input norm for a speculative (MTP/eagle) depth layer
    (the replicated-table analogue of the model-local ``fused_eh_norm``, which
    takes precomputed embeds).

Self-contained (no model-local imports) so it can live under ``layers/``.
"""

from dataclasses import dataclass
from typing import Any

import torch
import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


def make_input_embedding(
    num_embeddings: int,
    embedding_dim: int,
    *,
    params_dtype: torch.dtype | None = None,
    quant_config=None,
    prefix: str = "",
    tie_word_embeddings: bool = False,
) -> VocabParallelEmbedding:
    """Input token embedding with an optional replicated escape hatch.

    ``VLLM_REPLICATE_EMBED=1`` builds the embedding with ``disable_tp``: the full
    table lives on every rank and the lookup is a local gather with no mask and
    no all-reduce, which unlocks the fused gather+norm path. The cost is a full
    table per rank at TP>1 (no extra memory at TP=1, where vocab-parallel is
    already unsharded). A replicated, unsharded table cannot be tied to a
    vocab-parallel ``ParallelLMHead``, so tied word embeddings are rejected at
    TP>1 (at TP=1 ``disable_tp`` is a no-op and tying still works).
    """
    disable_tp = envs.VLLM_REPLICATE_EMBED
    if disable_tp and tie_word_embeddings:
        assert get_tensor_model_parallel_world_size() == 1, (
            "VLLM_REPLICATE_EMBED is unsupported with tied word embeddings "
            "(the replicated table cannot tie to a vocab-parallel lm_head)"
        )
    return VocabParallelEmbedding(
        num_embeddings,
        embedding_dim,
        params_dtype=params_dtype,
        quant_config=quant_config,
        prefix=prefix,
        disable_tp=disable_tp,
    )


def has_full_vocab_on_rank(embedding: torch.nn.Module) -> bool:
    """Whether ``embedding.weight`` is the whole vocab as a plain [V, H] table.

    The fused gather kernels index the table directly, so they need every row
    on-rank (``disable_tp``, or any TP=1 run) and an unquantized weight.
    """
    return getattr(embedding, "tp_size", 0) == 1 and isinstance(
        getattr(embedding, "quant_method", None), UnquantizedEmbeddingMethod
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


class FusedEmbedNormKernel(VllmTritonJitKernel["FusedEmbedNormKernel.CompileKey"]):
    kernel = staticmethod(_fused_embed_norm_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        ids_dtype: torch.dtype
        table_dtype: torch.dtype
        table_stride: int
        hidden_size: int
        block_size: int
        has_norm: bool
        num_warps: int

    def dispatch(
        self,
        *,
        ids_dtype: torch.dtype,
        table_dtype: torch.dtype,
        table_stride: int,
        hidden_size: int,
        has_norm: bool,
    ) -> CompileKey:
        block_size = triton.next_power_of_2(hidden_size)
        return self.CompileKey(
            ids_dtype=ids_dtype,
            table_dtype=table_dtype,
            table_stride=triton_scalar_specialization_rep(table_stride),
            hidden_size=hidden_size,
            block_size=block_size,
            has_norm=has_norm,
            num_warps=min(32, max(4, block_size // 512)),
        )

    def get_warmup_keys(self, **kwargs: Any) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(**kwargs)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        table = TritonWarmupTensor(
            compile_key.table_dtype,
            shape=(1, compile_key.hidden_size),
        )
        return dict(
            input_ids=TritonWarmupTensor(compile_key.ids_dtype),
            embed_table=table,
            chain_weight=(table if compile_key.has_norm else None),
            eps=0.0,
            _outputs=(
                table,
                table if compile_key.has_norm else None,
            ),
        )

    @kernel_launcher
    def __call__(
        self,
        input_ids: torch.Tensor,
        embed_table: torch.Tensor,
        chain_weight: torch.Tensor | None = None,
        eps: float = 0.0,
        *,
        _outputs: tuple[Any, Any] | None = None,
    ) -> LaunchSpec:
        ids = input_ids if _outputs is not None else input_ids.view(-1)
        t = ids.shape[0]
        h = embed_table.shape[1]
        if chain_weight is not None:
            assert chain_weight.shape == (h,), (chain_weight.shape, h)
        if _outputs is None:
            out = torch.empty(
                (t, h), dtype=embed_table.dtype, device=embed_table.device
            )
            normed = torch.empty_like(out) if chain_weight is not None else None
        else:
            out, normed = _outputs
        block = triton.next_power_of_2(h)
        outputs = (out, normed) if normed is not None else out
        return (t,) if t else None, dict(
            ids_ptr=ids,
            table_ptr=embed_table,
            table_stride_0=embed_table.stride(0),
            out_ptr=out,
            normed_ptr=normed if normed is not None else out,
            chain_w_ptr=chain_weight if chain_weight is not None else embed_table,
            eps=eps,
            H=h,
            BLOCK=block,
            HAS_NORM=chain_weight is not None,
            num_warps=min(32, max(4, block // 512)),
        ), outputs


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
    return _FUSED_EMBED_NORM_KERNEL(input_ids, embed_table, chain_weight, eps)


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


class FusedEmbedEhNormKernel(
    VllmTritonJitKernel["FusedEmbedEhNormKernel.CompileKey"]
):
    kernel = staticmethod(_fused_embed_eh_norm_kernel)

    @dataclass(frozen=True)
    class CompileKey:
        ids_dtype: torch.dtype
        table_dtype: torch.dtype
        hidden_dtype: torch.dtype
        hidden_size: int
        block_size: int
        table_stride: int
        hidden_stride: int
        output_stride: int

    def dispatch(
        self,
        *,
        ids_dtype: torch.dtype,
        table_dtype: torch.dtype,
        hidden_dtype: torch.dtype,
        hidden_size: int,
    ) -> CompileKey:
        return self.CompileKey(
            ids_dtype=ids_dtype,
            table_dtype=table_dtype,
            hidden_dtype=hidden_dtype,
            hidden_size=hidden_size,
            block_size=triton.next_power_of_2(hidden_size),
            table_stride=triton_scalar_specialization_rep(hidden_size),
            hidden_stride=triton_scalar_specialization_rep(hidden_size),
            output_stride=triton_scalar_specialization_rep(2 * hidden_size),
        )

    def get_warmup_keys(self, **kwargs: Any) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(**kwargs)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        hidden = TritonWarmupTensor(
            compile_key.hidden_dtype,
            shape=(1, compile_key.hidden_size),
        )
        return dict(
            positions=TritonWarmupTensor(torch.int64),
            input_ids=TritonWarmupTensor(compile_key.ids_dtype),
            embed_table=TritonWarmupTensor(
                compile_key.table_dtype,
                shape=(1, compile_key.hidden_size),
            ),
            previous_hidden=hidden,
            enorm_w=hidden,
            hnorm_w=hidden,
            eps=0.0,
            _output=TritonWarmupTensor(
                compile_key.hidden_dtype,
                shape=(1, 2 * compile_key.hidden_size),
            ),
        )

    @kernel_launcher
    def __call__(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        embed_table: torch.Tensor,
        previous_hidden: torch.Tensor,
        enorm_w: torch.Tensor,
        hnorm_w: torch.Tensor,
        eps: float,
        *,
        _output: Any | None = None,
    ) -> LaunchSpec:
        n, h = previous_hidden.shape
        out = _output
        if out is None:
            out = torch.empty(
                n, 2 * h, dtype=previous_hidden.dtype, device=previous_hidden.device
            )
        return (n,), dict(
            pos_ptr=positions,
            ids_ptr=input_ids,
            table_ptr=embed_table,
            table_stride=embed_table.stride(0),
            prev_ptr=previous_hidden,
            prev_stride=previous_hidden.stride(0),
            enorm_w_ptr=enorm_w,
            hnorm_w_ptr=hnorm_w,
            eps=eps,
            out_ptr=out,
            out_stride=out.stride(0),
            H=h,
            BLOCK=triton.next_power_of_2(h),
        ), out


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
    return _FUSED_EMBED_EH_NORM_KERNEL(
        positions,
        input_ids,
        embed_table,
        previous_hidden,
        enorm_w,
        hnorm_w,
        eps,
    )


_FUSED_EMBED_NORM_KERNEL = FusedEmbedNormKernel()
_FUSED_EMBED_EH_NORM_KERNEL = FusedEmbedEhNormKernel()
