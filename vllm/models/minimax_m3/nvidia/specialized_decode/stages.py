# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-layer stages of the specialized decode.

A decoder layer of the executor is a chain of stages; each stage has one
interface below and any number of implementations. `stock.py` implements
every stage with the ops the stock layers run; a kernel replaces one stage at a
time by implementing its interface (see `AGENTS.md`).

Shapes are per TP rank. `rows` is the step's token count,
`c * (K + 1)` for `c` requests and `K` DSpark draft tokens (padded rows
included), at most `MAX_ROWS`. At TP4 for MiniMax-M3.1-Flash:

======================  =====================================================
`hidden`              6144
`q_heads`             16 (64 / TP), `head_dim` 128, `q_size` 2048
`kv_heads`            1 (4 / TP), `kv_size` 128
`index_heads`         1 (= `kv_heads`), `index_dim` 128
`topk_blocks`         16 blocks of 128 tokens (= KV block size)
`max_k_tiles`         8192 (the indexer's unified score buffer)
`experts`             128 routed, top 4; expert intermediate 768 (3072 / TP)
`dense_intermediate`  3072 (12288 / TP), layers 0-2
======================  =====================================================

Every stage runs inside a FULL CUDA graph capture: it may not synchronize with
the host or read GPU values back, and every decision it makes must depend only
on host state shared by all TP ranks (`rows`, module attributes, attention
metadata host fields). Stages that communicate (`AllReduceNormStage`) must
issue the same collectives on every rank.

Fused kernel stages. Some kernel implementations cover more than their own
slot, or hand a private layout to a partner stage instead of the tensor the
interface below names:

- a `FrontStage` may fuse the input boundary (`fuses_input`): the executor
  then calls `run_fused` with the previous FFN's pending output and residual
  instead of `AllReduceNormStage.input` + `run`;
- an `OProjStage` may return a pending object only its partner
  `AllReduceNormStage.post_attention` consumes (the o_proj + boundary pair);
- a `PageTopKStage` may return a private block table only its partner
  `SparseDecodeStage` reads;
- a `MoEStage` may push its output straight into the next front's exchange
  (`can_push`) and return a `PushedFFN`.

`Stage.compatible` states which partners an implementation needs; `select`
demotes every implementation whose partners are not selected (down to the
stock one, which composes with everything), so every mix it returns is one
that is correct. The executor asks `fuses_input` / `polls_push` /
`can_push` before each boundary, so a push is only ever made to a front that
consumes it.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import torch

from . import STAGE_NAMES, CacheDtypes

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )

    from ..model import (
        MiniMaxM3Attention,
        MiniMaxM3DecoderLayer,
        MiniMaxM3MLP,
        MiniMaxM3Model,
        MiniMaxM3MoE,
        MiniMaxM3SparseAttention,
    )
    from .runtime.decode import DecodeStep

    LayerInput: TypeAlias = torch.Tensor | QuantizedActivation
    PendingFFN: TypeAlias = torch.Tensor | MoEOutput | "PushedFFN"


class PushedFFN(NamedTuple):
    """An FFN output its stage already pushed into the exchange the next
    layer's (input-fusing) front polls. Only a `FrontStage` whose
    `polls_push` holds for that layer may receive it."""

    # [rows, hidden] BF16: the same un-reduced partial (reading it does not consume the
    # push).
    partial: torch.Tensor


class FrontOutput(NamedTuple):
    """The attention inputs a front stage leaves for the next stages."""

    # [rows, q_size]: E4M3 with the layer's ``_q_scale_float`` when the CUTLASS sparse
    # decode serves the layer (``impl.use_cutlass_decode``), else BF16. QK-normed and
    # RoPE'd. Dense layers: BF16, a view into the qkv output.
    q: torch.Tensor
    # Sparse layers: [rows, index_heads * index_dim], index-normed and RoPE'd: E4M3 for
    # an fp8 index cache, BF16 for a BF16 one. None on dense layers.
    index_q: torch.Tensor | None = None
    # Dense layers only: [rows, kv_size] BF16 K (normed, RoPE'd) and V, views into the
    # qkv output; the dense decode inserts them into the KV cache.
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None


class BoundaryOutput(NamedTuple):
    """A sublayer boundary's result: the next sublayer's input and residual."""

    # [rows, hidden]: BF16, or an MXFP8 ``QuantizedActivation`` (E4M3 data [rows,
    # hidden] + swizzled E8M0 scales) that ``qkv_proj`` reads as is.
    hidden: "LayerInput"
    # [rows, hidden] BF16: the all-reduced residual stream after this boundary.
    residual: torch.Tensor
    # Post-attention boundaries of MoE layers: [rows, experts] FP32 router logits of
    # ``hidden``, when the boundary computes them. None otherwise.
    router_logits: torch.Tensor | None = None


class Stage(ABC):  # noqa: B024 (the stage interfaces below add the abstract methods)
    """What every stage shares."""

    # Short stage name, for logs and test ids (e.g. "front").
    name: str = ""
    # Most rows this implementation runs; larger steps take the next implementation of
    # the stage (finally the stock one, which has no limit).
    max_rows: int | None = None
    # The (KV, index-K) cache dtypes this implementation runs, or None for all; other
    # deployments take the next implementation of the stage.
    cache_dtypes: frozenset[CacheDtypes] | None = None

    def runs_with(self, dtypes: CacheDtypes) -> bool:
        """Whether this implementation runs over caches of `dtypes`; decided
        once, when the executor is built."""
        return self.cache_dtypes is None or dtypes in self.cache_dtypes

    def supports(self, rows: int) -> bool:
        """Whether this implementation runs a step of `rows` rows. Host-only
        and alike on every TP rank."""
        return self.max_rows is None or rows <= self.max_rows

    def warm_up(self, rows: Sequence[int]) -> None:  # noqa: B027
        """Compile and tune for every row count in `rows` (each one this
        stage `supports`), before memory profiling and CUDA graph capture.
        Collective when the stage is; the stock stages need nothing."""

    def compatible(self, step: "StepStages") -> bool:
        """Whether this implementation can run beside the other stages of
        `step` (host-only, alike on every rank). Implementations that hand a
        private layout to a partner stage return False unless that partner is
        selected; the stock stages compose with every implementation."""
        return True

    def begin_step(self, step: "DecodeStep", positions: torch.Tensor) -> None:  # noqa: B027
        """Called once per step (`step.rows` rows, `positions` [rows]
        int64), before the first layer, on every selected implementation
        (e.g. to build step metadata the kernels share). Runs inside the
        capture: graph-safe ops only."""


class FrontStage(Stage):
    """`qkv_proj` + per-head QK norm + partial RoPE (+ the index branch's
    norm and RoPE) + the KV and index-K cache insert of sparse layers."""

    name = "front"

    def fuses_input(self, layer: "MiniMaxM3DecoderLayer") -> bool:
        """Whether this implementation also runs `layer`'s input boundary
        (the previous FFN's all-reduce, residual add, input norm): then the
        executor calls `run_fused` instead of `AllReduceNormStage.input` +
        `run`, and leaves the previous MoE's finalize deferred only when
        `fuses_finalize`."""
        return False

    def polls_push(self, layer: "MiniMaxM3DecoderLayer") -> bool:
        """Whether `run_fused` on `layer` takes a `PushedFFN` (the
        previous MoE pushed its output into this front's exchange)."""
        return False

    def fuses_finalize(self, layer: "MiniMaxM3DecoderLayer") -> bool:
        """Whether `run_fused` on `layer` also takes a `MoEOutput` whose
        routed output is unfinalized (it runs the previous MoE's top-k
        reduction and shared add itself), so that MoE may defer."""
        return False

    def run_fused(
        self,
        layer: "MiniMaxM3DecoderLayer",
        pending: "PendingFFN",
        residual: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[FrontOutput, torch.Tensor]:
        """The input boundary and `run` in one: `pending` is the previous
        FFN's un-reduced [rows, hidden] BF16 output, a `MoEOutput` when
        `fuses_finalize`, or a `PushedFFN` when `polls_push`; `residual`
        [rows, hidden] BF16. Returns the front's output and the residual stream
        after the boundary (what `AllReduceNormStage.input` returns as
        `residual`)."""
        raise NotImplementedError(f"{type(self).__name__} does not fuse the input")

    @abstractmethod
    def run(
        self,
        attn: "MiniMaxM3SparseAttention | MiniMaxM3Attention",
        hidden: "LayerInput",
        positions: torch.Tensor,
    ) -> FrontOutput:
        """`hidden`: [rows, hidden] BF16 or MXFP8 (the input norm's output).
        `positions`: [rows] int64.

        Sparse layers write the step's K/V into `attn.kv_cache` (BF16 or fp8
        E4M3, 128-token blocks, in the MiniMax-M3 sparse backend's layout) and its
        index K into `attn.indexer.index_cache.kv_cache` (BF16 or fp8 E4M3)
        at the
        forward context's slot mappings for `attn.layer_name` and the index
        cache prefix."""


class IndexScoreStage(Stage):
    """The lightning indexer's decode scores: per query row and index head,
    the max score of every KV block of the request's context."""

    name = "index_score"

    @abstractmethod
    def run(
        self, attn: "MiniMaxM3SparseAttention", index_q: torch.Tensor
    ) -> torch.Tensor:
        """`index_q`: [rows, index_heads * index_dim], E4M3 or BF16 as the
        index cache.

        Returns the scores [rows, index_heads, max_k_tiles] FP32: the indexer
        metadata's `unified_scores` (pre-filled with -inf by the metadata
        build), with every causally valid block's tile written. The page top-k
        reads it next."""


class PageTopKStage(Stage):
    """The top `topk_blocks` KV blocks per query row and index head, always
    including the init and local blocks."""

    name = "page_topk"

    @abstractmethod
    def run(
        self, attn: "MiniMaxM3SparseAttention", scores: torch.Tensor
    ) -> torch.Tensor:
        """`scores`: [rows, index_heads, max_k_tiles] FP32.

        Returns the selected block ids [rows, index_heads, topk_blocks] int32,
        -1 past the context: a view of the model's shared
        `topk_indices_buffer` (the sparse decode of the same layer reads it).
        An implementation whose `compatible` requires a specific
        `SparseDecodeStage` may instead return its own table for it."""


class SparseDecodeStage(Stage):
    """Block-sparse GQA decode over the selected blocks of the KV cache."""

    name = "sparse_decode"

    @abstractmethod
    def run(
        self,
        attn: "MiniMaxM3SparseAttention",
        q: torch.Tensor,
        topk: torch.Tensor,
    ) -> torch.Tensor:
        """`q`: [rows, q_size] E4M3 or BF16 (`FrontOutput.q`).
        `topk`: [rows, index_heads, topk_blocks] int32.

        Returns the attention output [rows, q_size] BF16."""


class DenseDecodeStage(Stage):
    """Full GQA decode of the dense-attention layers (0-2), including their
    KV-cache insert."""

    name = "dense_decode"

    @abstractmethod
    def run(self, attn: "MiniMaxM3Attention", front: FrontOutput) -> torch.Tensor:
        """`front`: q [rows, q_size], k and v [rows, kv_size], BF16.

        Returns the attention output [rows, q_size] BF16."""


class OProjStage(Stage):
    """`o_proj`: the row-parallel MXFP8 output projection, un-reduced."""

    name = "o_proj"

    @abstractmethod
    def run(
        self,
        attn: "MiniMaxM3SparseAttention | MiniMaxM3Attention",
        attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """`attn_out`: [rows, q_size] BF16.

        Returns this rank's partial sum [rows, hidden] BF16; the next
        `AllReduceNormStage.post_attention` reduces it. An implementation
        whose `compatible` requires its partner `AllReduceNormStage` may
        return a pending object instead, which that partner launches with
        the residual (the fused o_proj + boundary pair)."""


class AllReduceNormStage(Stage):
    """The sublayer boundaries: the TP all-reduce of the previous sublayer's
    partial output, the residual add and the Gemma RMSNorm (and, after
    attention on MoE layers, the FP32 router GEMM)."""

    name = "allreduce_norm"

    @abstractmethod
    def input(
        self,
        layer: "MiniMaxM3DecoderLayer",
        pending: "PendingFFN",
        residual: torch.Tensor | None,
    ) -> BoundaryOutput:
        """Into `layer`'s attention. `pending`: the embedding [rows, hidden]
        BF16 when `residual` is None (layer 0); otherwise the previous FFN's
        un-reduced output [rows, hidden] BF16, or its `MoEOutput` whose top-k
        finalize was deferred to this boundary. `residual`: [rows, hidden]
        BF16."""

    @abstractmethod
    def post_attention(
        self,
        layer: "MiniMaxM3DecoderLayer",
        partial: torch.Tensor,
        residual: torch.Tensor,
    ) -> BoundaryOutput:
        """Into `layer`'s FFN. `partial`: `o_proj`'s [rows, hidden] BF16
        (or its partner o_proj's pending object). Returns BF16 `hidden`;
        `router_logits` [rows, experts] FP32 on MoE layers if this stage
        computes them (else the MoE stage does)."""

    @abstractmethod
    def final(
        self,
        model: "MiniMaxM3Model",
        pending: "PendingFFN",
        residual: torch.Tensor,
    ) -> BoundaryOutput:
        """The final norm after the last layer: `hidden` [rows, hidden] BF16
        for the LM head, `residual` the residual stream after the last
        layer."""


class MoEStage(Stage):
    """The MoE FFN: shared expert + NVFP4 routed experts (sigmoid routing with
    the correction bias, top 4 of 128, renormalized and scaled)."""

    name = "moe"

    def can_push(self, moe: "MiniMaxM3MoE", rows: int) -> bool:
        """Whether `run(..., push=True)` is possible for `moe` at
        `rows`: the output can be pushed into the next front's exchange."""
        return False

    @abstractmethod
    def run(
        self,
        moe: "MiniMaxM3MoE",
        hidden: torch.Tensor,
        router_logits: torch.Tensor | None,
        defer_finalize: bool,
        push: bool = False,
    ) -> "PendingFFN":
        """`hidden`: [rows, hidden] BF16. `router_logits`: [rows, experts]
        FP32, or None to compute them here (or route from the boundary's own
        hand-off, for a kernel MoE behind its kernel boundary).

        Returns the un-reduced FFN output [rows, hidden] BF16, or with
        `defer_finalize` a `MoEOutput` whose top-k finalize the next
        boundary fuses with its all-reduce (an implementation may ignore the
        hint and return the tensor), or with `push` (only passed when
        `can_push` and the next front `polls_push`) a `PushedFFN`."""


class DenseMLPStage(Stage):
    """The dense SwiGLU-OAI MLP of layers 0-2 (MXFP8 `gate_up_proj` and
    `down_proj`)."""

    name = "dense_mlp"

    @abstractmethod
    def run(self, mlp: "MiniMaxM3MLP", hidden: torch.Tensor) -> torch.Tensor:
        """`hidden`: [rows, hidden] BF16. Returns the un-reduced output
        [rows, hidden] BF16."""


class Stages(NamedTuple):
    """The implementations of every stage the executor runs, each a list
    tried in order: the first that `supports` a step's rows runs it. The
    last of each list is the stock stage, which supports every step."""

    front: Sequence[FrontStage]
    index_score: Sequence[IndexScoreStage]
    page_topk: Sequence[PageTopKStage]
    sparse_decode: Sequence[SparseDecodeStage]
    dense_decode: Sequence[DenseDecodeStage]
    o_proj: Sequence[OProjStage]
    allreduce_norm: Sequence[AllReduceNormStage]
    moe: Sequence[MoEStage]
    dense_mlp: Sequence[DenseMLPStage]

    def for_cache_dtypes(self, dtypes: CacheDtypes) -> "Stages":
        """The implementations that run over caches of `dtypes`."""
        return Stages(
            *([stage for stage in impls if stage.runs_with(dtypes)] for impls in self)  # type: ignore[arg-type]
        )

    def select(self, rows: int) -> "StepStages":
        """The implementation of every stage for a step of `rows` rows: per
        stage the first that `supports` the rows, then, while any selected
        implementation is not `compatible` with the others, the first such
        one (in stage order) is replaced by the next supporting implementation
        of its stage. Implementations only move down their lists and the
        stock ones are compatible with everything, so this ends, and the
        result is deterministic (the same on every rank)."""
        candidates = [
            [stage for stage in impls if stage.supports(rows)] for impls in self
        ]
        for impls, found in zip(self, candidates):
            if not found:
                raise ValueError(
                    f"no implementation of stage {impls[-1].name!r} supports "
                    f"{rows} rows"
                    if impls
                    else "a stage without implementations"
                )
        index = [0] * len(candidates)
        while True:
            step = StepStages(
                *(found[i] for found, i in zip(candidates, index))  # type: ignore[arg-type]
            )
            demoted = False
            for slot, stage in enumerate(step):
                if stage.compatible(step):
                    continue
                if index[slot] + 1 >= len(candidates[slot]):
                    raise ValueError(
                        f"no implementation of stage {stage.name!r} for {rows} "
                        "rows is compatible with the other stages"
                    )
                index[slot] += 1
                demoted = True
                break
            if not demoted:
                return step

    def all(self) -> list[Stage]:
        """Every implementation of every stage, once."""
        seen: dict[int, Stage] = {}
        for impls in self:
            for stage in impls:
                seen.setdefault(id(stage), stage)
        return list(seen.values())


class StepStages(NamedTuple):
    """One implementation per stage: the chain of one step."""

    front: FrontStage
    index_score: IndexScoreStage
    page_topk: PageTopKStage
    sparse_decode: SparseDecodeStage
    dense_decode: DenseDecodeStage
    o_proj: OProjStage
    allreduce_norm: AllReduceNormStage
    moe: MoEStage
    dense_mlp: DenseMLPStage

    def distinct(self) -> list[Stage]:
        """Every selected implementation, once (an object may fill several
        slots)."""
        seen: dict[int, Stage] = {}
        for stage in self:
            seen.setdefault(id(stage), stage)
        return list(seen.values())


assert StepStages._fields == Stages._fields == STAGE_NAMES
