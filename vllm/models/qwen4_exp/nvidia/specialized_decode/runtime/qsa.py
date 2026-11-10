# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QSA layers of the specialized decode's steps: their sparse attention
and, in small steps, their indexer's block select.

Why. A QSA layer's sparse attention (`Qwen4ExpQSAImpl.forward`) zero-fills
its whole output (a kernel of its own, about 1 us a layer at one to eight
requests) and then runs the stock Triton split-K kernel and, with more than
one split, its merge (`ops/qsa.py`). Those kernels write every value of the
step's rows themselves: the split-K kernel stores every head of its row and
KV head (no early exit; a row without selected positions, a padding row
among them, stores zeros) when it runs one split, and the merge stores every
(row, head) when it runs more. The fill of the step's rows is dead.

At 128 rows and more at TP1 (32 requests and more, four rows a request, two
KV heads) the stock kernels are also thousands of one-warp CTAs and tens of
megabytes of FP32 partials a layer; the row attention of `..qsa` (one CTA
per row and KV head, no partials) computes the same operator about 18%
sooner there. With fewer CTAs than `MIN_CTAS` (two a SM) the stock kernels
keep more SMs busy and stay.

What changes. `QsaRows` holds the QSA layers among some modules whose main
K/V cache is BF16: the target's decoder layers for the executor
(`SpecializedDecode`), the MTP draft model's layers for the draft passes that
take the fused front (`DraftFront`). Within `QsaRows.active(rows)` each of
those layers has a `DecodeQsaImpl` as its `impl`: the stock
implementation's K/V cache update, then the stock kernels without the fill
of the step's rows, or `qsa_row_attention` instead when the launch would
have at least `MIN_CTAS` CTAs (the step's rows times the layer's KV heads
on the rank: 128 rows at TP1, 256 at TP4) and the kernel takes the operands
(`row_attention_unsupported`). In steps of at most `SELECT_ROWS` rows,
each of those layers whose indexer has the fused prepare and the deployed
selection (512 blocks of 4 tokens) also has a `DecodeIndexer` as its
indexer's `forward`: the stock decode branch (the fused prepare, the score
kernel's FP32 logits) with `qsa_select_expand` (`..qsa`, one cluster of
CTAs a row) in place of the stock top-k and the index expansion. Outside the
block the layers are as built.

Output. The row attention also writes every value of the step's rows,
padding rows included (request -1 or no selected position: zeros, as stock
writes them), so with either kernel `DecodeQsaImpl` zero-fills only the
rows past the step's, if the output has any. Padding rows and the rows past
the step's are then bitwise what the stock forward leaves there.

When. The executor and the draft front enter the block only while they
capture a FULL CUDA graph of a pure decode step, so the choice is static per
captured shape: the graphs of steps that large replay the row attention, the
other decode graphs the stock kernels without the fill, every other graph
and every eager forward the stock forward. The swap changes the layers'
`impl` in place, so it assumes one thread runs the forwards and nothing
traces them, as `PlannedGemms.active` does: the deployment check rejects
micro-batching and `torch.compile` of the model.

Numerics. Without the fill the stock kernels write the stock forward's
output bitwise. The CuTe select selects the stock top-k's set of blocks
(of the blocks tied with the 512th score, lowest block index first when at
most 4,096 scores share its quarter octave, in CTA, thread and key order
otherwise, where the stock top-k takes them in the order of its atomics) and
writes the stock expansion's row bitwise when a row has at most 512 visible
blocks; with more, its blocks' order in the row is its own, as the stock
top-k's is the order of its atomics, so the attention sums them in another
order. The row attention computes
stock's operator with stock's rounding points (BF16 probabilities in P V,
FP32 sums, the attention output rounded to BF16 before the gate) but sums in
another order: its outputs are not bitwise equal to stock's, and its error
against an FP32 reference is stock's
(`tests/models/qwen4_exp/specialized_decode/test_qsa_attention.py`).

Warm-up. `QsaRows.warm_up` compiles and runs the row attention once for
each distinct (query heads per KV head, selection width) of its layers, and
the select once for each cluster shape of the steps it serves; the
executor and the draft front call it when they warm up their own kernels,
before memory profiling and every CUDA graph capture, so no capture compiles
it. The kernel takes the cache's page size, strides and the block table's
width at launch.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger

if TYPE_CHECKING:
    from ....common.qsa import Qwen4ExpQSAMetadata
    from ...qsa import Qwen4ExpQSAAttention, Qwen4ExpQSAImpl
    from ...qsa_indexer import QSAIndexer

logger = init_logger(__name__)

# The fewest CTAs (rows times the rank's KV heads) of a launch that takes the
# row attention: two a SM, one wave; TP1 at 32 requests with three draft
# tokens. GB300, TP1 (two KV heads), CUDA graphs of 12 layers over real
# 65,536-token selections, us a call (row attention against stock): 77.75
# against 94.5 at 128 rows (256 CTAs), 57.0 against 52.0 at 64 (128 CTAs),
# 56.5 against 30.75 at 32.
MIN_CTAS = 256

# The most rows of a step whose QSA indexers take the CuTe select. GB300, us a
# layer-pass from the score kernel's end to the select's end at 65,536-token
# contexts (stock top-k and expansion against the CuTe select): 11.25 against
# 7.75 at 4 rows, 11.00 against 10.50 at 16, 12.25 against 18.25 at 64.
SELECT_ROWS = 32


class DecodeQsaImpl:
    """A QSA layer's attention implementation within `QsaRows.active`: the
    stock kernels, or the row attention (`row`) where the kernel takes the
    operands, without the zero fill of the step's rows.

    Args:
        stock: The layer's own implementation.
        row: Whether the launch takes the row attention (`MIN_CTAS`).

    """

    def __init__(self, stock: "Qwen4ExpQSAImpl", row: bool) -> None:
        self.stock = stock
        self.row = row

    def do_kv_cache_update(
        self,
        layer: nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """The stock K/V cache update."""
        self.stock.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def forward(
        self,
        layer: nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "Qwen4ExpQSAMetadata",
        output: torch.Tensor,
        *,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
    ) -> torch.Tensor:
        """`Qwen4ExpQSAImpl.forward` without its zero fill of the step's
        rows, with the row attention in place of the stock kernels when
        `row` is set and `qsa_row_attention` takes the operands; the stock
        forward for an empty step, a layer without its selection buffer
        (`topk_indices_buffer`) or a cache that is not BF16.

        Args:
            layer: The QSA layer (its `topk_indices_buffer` holds the packed
                selection).
            query: The queries `[tokens, heads, head_size]`.
            kv_cache: The layer's main K/V cache.
            attn_metadata: The layer's attention metadata.
            output: The attention output (the step's rows written, the rest
                zero-filled).
            token_to_req: Each row's request.
            use_prefill_config: The stock kernel's configuration choice.
            output_gate: The output gate logits.

        """
        # Imported here: CuTe DSL kernels (CUDA builds only).
        from ...ops.qsa import qsa_sparse_paged_attention
        from ..qsa import qsa_row_attention, row_attention_unsupported

        stock = self.stock
        rows = attn_metadata.num_actual_tokens
        indices = getattr(layer, "topk_indices_buffer", None)
        if rows == 0 or indices is None or kv_cache.dtype != torch.bfloat16:
            return stock.forward(
                layer,
                query,
                kv_cache,
                attn_metadata,
                output,
                token_to_req=token_to_req,
                use_prefill_config=use_prefill_config,
                output_gate=output_gate,
            )
        key_cache, value_cache = kv_cache.transpose(1, 2).split(stock.head_size, dim=-1)
        # Either kernel writes every value of the step's rows (module docstring).
        if output.shape[0] > rows:
            output[rows:].zero_()
        if self.row:
            missing = row_attention_unsupported(
                query[:rows],
                key_cache,
                value_cache,
                indices[:rows],
                attn_metadata.block_table,
                output[:rows],
                output_gate[:rows],
            )
            if not missing:
                qsa_row_attention(
                    query[:rows],
                    key_cache,
                    value_cache,
                    indices[:rows],
                    attn_metadata.block_table,
                    token_to_req[:rows],
                    output[:rows],
                    output_gate=output_gate[:rows],
                )
                return output
            logger.warning_once(
                "Qwen4Exp specialized decode: a %d-row QSA attention keeps the "
                "stock kernels (the row attention needs %s).",
                rows,
                missing,
            )
        qsa_sparse_paged_attention(
            query[:rows],
            key_cache,
            value_cache,
            indices[:rows],
            attn_metadata.block_table,
            token_to_req[:rows],
            use_prefill_config,
            output[:rows],
            output_gate=output_gate[:rows],
        )
        return output


def _decode_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    decode_query_len: int,
) -> torch.Tensor:
    """The stock decode score kernel's FP32 logits `[rows, columns]`, as
    `qsa_select_paged_decode` launches it before its top-k."""
    # Imported here: the stock indexer kernels (CUDA builds only).
    from vllm.triton_utils import triton

    from ...ops.qsa_indexer import (
        _DECODE_BLOCK_N,
        _decode_tiles_per_program,
        _qsa_mqa_paged_uniform_kernel,
    )

    num_requests = q.shape[0] // decode_query_len
    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    tiles_per_program = _decode_tiles_per_program(num_requests, columns)
    grid = (
        num_requests,
        triton.cdiv(columns, _DECODE_BLOCK_N * tiles_per_program),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        BLOCK_N=_DECODE_BLOCK_N,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        num_warps=1 if k_cache.dtype == torch.float8_e4m3fn else 2,
    )
    return logits


class DecodeIndexer:
    """A QSA indexer's `forward` within `QsaRows.active` for steps of at
    most `SELECT_ROWS` rows: in a pure decode step with the fused prepare,
    the stock prepare and score kernel, then `qsa_select_expand` in place of
    the stock top-k and expansion; the stock forward otherwise.

    Args:
        indexer: The indexer.

    """

    def __init__(self, indexer: "QSAIndexer") -> None:
        self.indexer = indexer

    def __call__(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        attn: "Qwen4ExpQSAAttention",
        qkv: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """`QSAIndexer.forward` (same arguments and results), with the CuTe
        select in a pure decode step.

        Args:
            projected_qk: The indexer's query and raw key projections.
            positions: The rope positions.
            out: The packed selection buffer.
            attn: The QSA layer.
            qkv: The layer's main projections.
            slot_mapping: The main K/V cache slots.

        """
        # Imported here: CuTe DSL kernels (CUDA builds only).
        from ...ops.qsa_prepare import qsa_prepare
        from ..qsa import PACKED_WIDTH, qsa_select_expand

        indexer = self.indexer
        metadata = indexer._metadata()
        if (
            metadata is None
            or out is None
            or qkv is None
            or slot_mapping is None
            or indexer.skip_topk
            or not attn.use_fused_qsa_prepare
        ):
            return type(indexer).forward(
                indexer,
                projected_qk,
                positions,
                out,
                attn=attn,
                qkv=qkv,
                slot_mapping=slot_mapping,
            )
        raw, compressed = metadata
        rows = raw.num_actual_tokens
        decodes = compressed.num_decodes
        query_len = compressed.decode_query_len
        compressed_cache = indexer.compressed_key_cache.kv_cache
        if (
            rows == 0
            or compressed.num_decode_tokens != rows
            or decodes * query_len != rows
            or out.shape != (rows, PACKED_WIDTH)
            or compressed_cache.shape[1] % 4
        ):
            return type(indexer).forward(
                indexer,
                projected_qk,
                positions,
                out,
                attn=attn,
                qkv=qkv,
                slot_mapping=slot_mapping,
            )
        # The stock forward's fused prepare, called as it calls it.
        projected_q, raw_keys = projected_qk[:rows].split(
            (
                indexer.index_n_heads * indexer.index_head_dim,
                indexer.index_kv_heads * indexer.index_head_dim,
            ),
            dim=-1,
        )
        ring = indexer.raw_key_cache
        q = projected_q.new_empty(
            rows,
            indexer.index_n_heads,
            indexer.index_head_dim,
            dtype=indexer.indexer_dtype,
        )
        main_kv_cache = attn.kv_cache.transpose(1, 2)
        if attn.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            main_kv_cache = main_kv_cache.view(torch.float8_e4m3fn)
        main_outputs = qsa_prepare(
            projected_q,
            raw_keys,
            positions[..., :rows],
            indexer.rotary_emb.cos_sin_cache,
            indexer.q_layernorm.weight,
            indexer.k_layernorm.weight,
            indexer.q_layernorm.variance_epsilon,
            q,
            ring.kv_cache,
            raw.slot_mapping,
            raw.block_table,
            raw.query_start_loc,
            raw.logical_positions,
            compressed_cache,
            compressed.slot_mapping,
            compressed.k_work_metadata,
            compress_ratio=indexer.compress_ratio,
            mrope_section=getattr(indexer.rotary_emb, "mrope_section", None),
            rope_pos_offset=(
                ring.rope_position_offset
                if ring.rope_position_cache is not None
                else None
            ),
            main_qkv=qkv[:rows],
            main_q_norm_weight=attn.q_norm.weight,
            main_k_norm_weight=attn.k_norm.weight,
            main_eps=attn.q_norm.variance_epsilon,
            main_kv_cache=main_kv_cache,
            main_slot_mapping=slot_mapping[:rows],
            main_k_scale=attn._k_scale_float,
            main_v_scale=attn._v_scale_float,
        )
        visible = compressed.visible_blocks[:rows]
        logits = _decode_logits(
            q, compressed_cache, compressed.block_table[:decodes], visible, query_len
        )
        qsa_select_expand(logits, visible, compressed.logical_positions[:rows], out)
        return out, main_outputs


class QsaRows:
    """The QSA layers with a BF16 main K/V cache among some modules.

    Args:
        layers: The layers.

    """

    def __init__(
        self, layers: list["Qwen4ExpQSAAttention"], max_model_len: int | None = None
    ) -> None:
        self.layers = layers
        self.max_model_len = max_model_len

    @classmethod
    def build(cls, modules: Iterable[nn.Module]) -> "QsaRows":
        """The QSA layers in `modules` whose main K/V cache is BF16, with the
        current configuration's model length."""
        # Imported here: the QSA layer module imports the model module.
        from vllm.config import get_current_vllm_config_or_none

        from ...qsa import Qwen4ExpQSAAttention

        layers: list[Qwen4ExpQSAAttention] = []
        seen: set[int] = set()
        for module in modules:
            for child in module.modules():
                if id(child) in seen or not isinstance(child, Qwen4ExpQSAAttention):
                    continue
                seen.add(id(child))
                if child.kv_cache_torch_dtype == torch.bfloat16:
                    layers.append(child)
        config = get_current_vllm_config_or_none()
        return cls(layers, config.model_config.max_model_len if config else None)

    def selecting(self) -> list["Qwen4ExpQSAAttention"]:
        """The layers whose indexer can take the CuTe select: a fused prepare,
        the deployed selection (512 blocks of 4 tokens) and a model length
        whose rows hold at most `MAX_BLOCKS` blocks."""
        # Imported here: CuTe DSL kernels (CUDA builds only).
        from ..qsa import MAX_BLOCKS, RATIO, TOKENS

        if self.max_model_len is None or self.max_model_len > RATIO * MAX_BLOCKS:
            return []
        return [
            layer
            for layer in self.layers
            if layer.use_fused_qsa_prepare
            and layer.indexer.compress_ratio == RATIO
            and layer.indexer.token_topk == TOKENS
        ]

    @contextmanager
    def active(self, rows: int) -> Iterator[None]:
        """Within the block the layers take a `DecodeQsaImpl`: the row
        attention where the launch for a step of `rows` rows has at least
        `MIN_CTAS` CTAs (`rows` times the layer's KV heads), the stock
        kernels without the fill of the step's rows otherwise; with at most
        `SELECT_ROWS` rows the `selecting` layers' indexers take a
        `DecodeIndexer`.

        Args:
            rows: The step's rows.

        """
        saved = [(layer, layer.impl) for layer in self.layers]
        indexers = (
            [layer.indexer for layer in self.selecting()] if rows <= SELECT_ROWS else []
        )
        if not saved:
            yield
            return
        try:
            for indexer in indexers:
                # An instance attribute: nn.Module.__call__ looks it up first.
                indexer.forward = DecodeIndexer(indexer)  # type: ignore[method-assign]
            row_layers = 0
            for layer, impl in saved:
                row = rows * layer.num_kv_heads >= MIN_CTAS
                row_layers += row
                layer.impl = DecodeQsaImpl(impl, row)  # type: ignore[assignment]
            if row_layers:
                logger.info_once(
                    "Qwen4Exp specialized decode: the %d-row decode graphs run "
                    "the QSA row attention in %d layers.",
                    rows,
                    row_layers,
                )
            yield
        finally:
            for layer, impl in saved:
                layer.impl = impl
            for indexer in indexers:
                indexer.__dict__.pop("forward", None)

    def warm_up(self) -> list[tuple[int, ...]]:
        """Compile and run the row attention once for each distinct (query
        heads per KV head, selection width) of the layers, on an empty
        selection, and the select once for each cluster shape of steps of up
        to `SELECT_ROWS` rows, on an empty row; returns the attention's
        shapes and the select's cluster shapes."""
        # Imported here: CuTe DSL kernels (CUDA builds only).
        from ..qsa import (
            PACKED_WIDTH,
            default_variant,
            qsa_row_attention,
            qsa_select_expand,
        )

        shapes: dict[tuple[int, int], Qwen4ExpQSAAttention] = {}
        for layer in self.layers:
            topk = layer.indexer.packed_output_width - 1
            shapes.setdefault((layer.num_heads // layer.num_kv_heads, topk), layer)
        with torch.inference_mode():
            for (_group, topk), layer in shapes.items():
                device = next(layer.parameters()).device
                heads, kv_heads, dim = (
                    layer.num_heads,
                    layer.num_kv_heads,
                    layer.head_dim,
                )
                cache = torch.zeros(
                    1, kv_heads, 16, 2 * dim, dtype=torch.bfloat16, device=device
                )
                key_cache, value_cache = cache.transpose(1, 2).split(dim, dim=-1)
                query = torch.zeros(1, heads, dim, dtype=torch.bfloat16, device=device)
                indices = torch.zeros(1, topk + 1, dtype=torch.int32, device=device)
                qsa_row_attention(
                    query,
                    key_cache,
                    value_cache,
                    indices,
                    torch.zeros(1, 1, dtype=torch.int32, device=device),
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.empty_like(query),
                    output_gate=torch.zeros_like(query),
                )
            variants: list[tuple[int, ...]] = []
            selecting = self.selecting()
            if selecting:
                device = next(selecting[0].parameters()).device
                for rows in range(1, SELECT_ROWS + 1):
                    variant = default_variant(rows)
                    if variant in variants:
                        continue
                    variants.append(variant)
                    qsa_select_expand(
                        torch.zeros(1, 4, dtype=torch.float32, device=device),
                        torch.zeros(1, dtype=torch.int32, device=device),
                        torch.zeros(1, dtype=torch.int64, device=device),
                        torch.empty(1, PACKED_WIDTH, dtype=torch.int32, device=device),
                        variant,
                    )
            if shapes or variants:
                torch.accelerator.synchronize(device)
        return [*shapes, *variants]
