# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PLE layer's lookup in the executor's steps at TP > 1: the Lamport row
gather (`..ple.gather`) instead of the int8 all-reduce, copy and dequant.

The PLE layer (`Qwen4ExpPLELayer`, layer 1 of Qwen3.8-Flash-Next) embeds each
token as 16 rows of an FP8 n-gram table in pinned host memory, split along its
rows among the TP ranks. One layer ahead, `start_prefetch` writes the step's
n-gram ids into the module's `_prefetch_ids` and, on a side stream, the rows
this rank owns into the table's `_prefetch_buffer` (zeros for the others).
At the layer, the stock `finalize_prefetch` joins the side stream,
all-reduces the FP8 bytes over the ETP group (NCCL, about 14 us at TP4),
copies them into a new `[tokens, 2560]` tensor, and the layer dequantizes it
(`rows.to(bf16) * scale.to(bf16)`, three more kernels): about 27 us a step
before the layer's projection.

`PleGathers.active` swaps, for the executor's steps (the block is entered
while a FULL graph of a pure decode step is captured), the table's
`finalize_prefetch` for one that joins the side stream and runs
`PleRowGather` on the prefetch buffer and ids into a persistent BF16 buffer,
and the layer's `_dequantize_embeddings` for one that passes BF16 rows
through (and dequantizes anything else as before): 3.6 to 4.0 us a step, the
stock dequantized rows bit for bit with negative zeros as positive zeros, the
same on every rank. Steps of more rows than the gather was built for keep the
stock finalize. Outside the block the modules are as built.

Padded projection. The layer's `kv_proj` (`[12800, 2560]`, replicated)
multiplies the gathered rows next. At 4 to 8 rows cuBLAS picks a split-K
kernel and a reduction (13.4 us on GB300), at 16 rows a single kernel (11.6
us). Within the block, the projection's quant method (`PaddedProjection`)
therefore multiplies the first `PAD_ROWS` (16) rows of the gather's buffer
when the step has fewer, and returns the step's rows of the product: the rows
past the step's hold zeros or earlier steps' rows, and a GEMM's output row
depends on its own input row only. cuBLAS sums K in another order at 16 rows,
so these rows are within BF16 rounding of the stock projection's, not bitwise
equal.

Liveness. The gather polls for the other ranks' rows, so nothing that waits on
the ranks may hold SMs beside it (`..ple.gather`): the side stream's lookup
has been joined before the gather starts, and the step's other kernels run on
the main stream before or after it.

Generations. A rank that owns no row of a call does not hold the other ranks
back, so a gather's call `n + 2` may reuse call `n`'s slots only once
every rank has run a collective after call `n + 1` (`..ple.gather`). A
step calls each site's gather once, at its PLE layer, and every later decoder
layer of the step ends with an all-reduce over the TP group (the boundary
kernel or vLLM's), which every rank must enter before any rank completes it;
`warm_up` makes one call, and the steps' collectives follow it.

Only tables the gather serves take it: pinned-host FP8 tables split over an
ETP group equal to the TP group (no data parallelism in the ETP group), the
layer without sequence parallelism. Others keep the stock path (logged once).
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from ....common.ngram_embedding import (
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
)

if TYPE_CHECKING:
    from ...model import Qwen4ExpDecoderLayer
    from ...ngram_embedding import Qwen4ExpNGramEmbedding
    from ...ple_layer import Qwen4ExpPLELayer
    from ..ple.gather import PleRowGather

logger = init_logger(__name__)

# Rows the projection multiplies in steps of fewer (module docstring).
PAD_ROWS = 16


def gather_unsupported(layer: "Qwen4ExpPLELayer") -> str | None:
    """Why the PLE `layer`'s lookup cannot take the row gather, or None."""
    ngram = layer.ple_embedding
    table = ngram.ngram_embedding
    if not isinstance(table, Qwen4ExpPLEPinnedHostEmbedding):
        return f"the table is a {type(table).__name__}, not a pinned-host one"
    if not isinstance(table.embedding_method, Qwen4ExpPLEFp8EmbeddingMethod):
        return "the table is not FP8"
    if table.tp_size == 1 or table.parallel_group is None:
        return "the table is not split"
    if table.etp_data_parallel_size != 1 or not table.reduce_results:
        return "the table's group spans data parallelism or reduce-scatters"
    if ngram.use_sequence_parallel:
        return "the layer is sequence parallel"
    if ngram.head_dim % 8:
        return f"the head dimension {ngram.head_dim} is not a multiple of 8"
    return None


class PaddedProjection:
    """The PLE projection's quant method within `PleGathers.active`: steps
    of fewer than `rows` rows that come straight from the gather's buffer
    multiply its first `rows` rows (module docstring, Padded projection).

    Args:
        stock: The projection's own method.
        buffer: The gather's output buffer, at least `rows` rows.
        rows: Rows to multiply.

    """

    def __init__(
        self, stock: QuantizeMethodBase, buffer: torch.Tensor, rows: int
    ) -> None:
        assert buffer.shape[0] >= rows
        self.stock = stock
        self.buffer = buffer
        self.rows = rows

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`x @ layer.weight.T (+ bias)`, through the padded rows when `x`
        is the leading rows of the buffer."""
        rows = x.shape[0]
        if (
            bias is None
            and x.dim() == 2
            and rows < self.rows
            and x.data_ptr() == self.buffer.data_ptr()
            and x.shape[1] == self.buffer.shape[1]
            and x.is_contiguous()
        ):
            return self.stock.apply(layer, self.buffer[: self.rows], None)[:rows]
        return self.stock.apply(layer, x, bias)


class PleSite:
    """One PLE layer the gather serves, and its gather.

    Args:
        layer: The PLE layer.
        gather: Its row gather (built over the table's group).
        out: The gather's BF16 output, `[max(max tokens, PAD_ROWS),
            ple_embed_dim]`, zeros at first.

    """

    def __init__(
        self, layer: "Qwen4ExpPLELayer", gather: "PleRowGather", out: torch.Tensor
    ) -> None:
        self.layer = layer
        self.ngram: Qwen4ExpNGramEmbedding = layer.ple_embedding
        self.table = self.ngram.ngram_embedding
        assert isinstance(self.table, Qwen4ExpPLEPinnedHostEmbedding)
        self.gather = gather
        self.out = out
        self.heads = self.ngram.ngram_heads

    def finalize(
        self, stock: Callable[..., torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        """The table's `finalize_prefetch` within `PleGathers.active`."""

        def finalize_prefetch(total_tokens: int) -> torch.Tensor:
            """The step's rows gathered and dequantized (module docstring);
            the stock finalize for steps the gather does not take."""
            if not 0 < total_tokens * self.heads <= self.gather.max_rows:
                return stock(total_tokens=total_tokens)
            table = self.table
            assert table._prefetch_stream is not None
            assert table._prefetch_buffer is not None
            torch.cuda.current_stream().wait_stream(table._prefetch_stream)
            out = self.out[:total_tokens]
            self.gather(
                table._prefetch_buffer[:total_tokens],
                self.ngram._prefetch_ids[:total_tokens],
                out,
            )
            return out

        return finalize_prefetch

    def dequantize(
        self, stock: Callable[[torch.Tensor, torch.dtype], torch.Tensor]
    ) -> Callable[[torch.Tensor, torch.dtype], torch.Tensor]:
        """The layer's `_dequantize_embeddings` within
        `PleGathers.active`: gathered rows are already in the output dtype."""

        def dequantize_embeddings(
            embeddings: torch.Tensor, output_dtype: torch.dtype
        ) -> torch.Tensor:
            """`embeddings` if already `output_dtype`, else the stock
            dequantization."""
            if embeddings.dtype == output_dtype:
                return embeddings
            return stock(embeddings, output_dtype)

        return dequantize_embeddings


class PleGathers:
    """The PLE layers among the executor's layers that take the row gather.

    Args:
        sites: The layers and their gathers.

    """

    def __init__(self, sites: list[PleSite]) -> None:
        self.sites = sites

    @classmethod
    def build(
        cls, layers: list["Qwen4ExpDecoderLayer"], max_tokens: int
    ) -> "PleGathers":
        """The gathers of the PLE layers among `layers` for steps of up to
        `max_tokens` tokens. Collective over each table's group: every rank
        builds the same gathers (the decisions read the config only)."""
        sites: list[PleSite] = []
        for decoder_layer in layers:
            layer = decoder_layer.ple
            if layer is None:
                continue
            reason = gather_unsupported(layer)
            if reason is not None:
                if layer.ple_embedding.ngram_embedding.tp_size > 1:
                    logger.info_once(
                        "The Qwen4Exp specialized decode keeps the stock PLE "
                        "lookup of %s: %s.",
                        layer.prefix,
                        reason,
                    )
                continue
            # Imported here: CuTe DSL kernels (CUDA builds only).
            from ..ple.gather import PleRowGather

            ngram = layer.ple_embedding
            table = ngram.ngram_embedding
            assert table.parallel_group is not None
            device = table.weight_scale.device
            gather = PleRowGather(
                table.parallel_group.device_group,
                ngram.head_dim,
                max_tokens * ngram.ngram_heads,
                table.shard_indices.org_vocab_start_index,
                table.shard_indices.org_vocab_end_index,
                float(table.weight_scale.reshape(-1)[0]),
                device,
                trigger_early=True,
            )
            out = torch.zeros(
                max(max_tokens, PAD_ROWS),
                ngram.ngram_heads * ngram.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            sites.append(PleSite(layer, gather, out))
        return cls(sites)

    @contextmanager
    def active(self) -> Iterator[None]:
        """Within the block, the layers' lookups take the gather and their
        projections the padded rows."""
        projections = [
            (site.layer.kv_proj, site.layer.kv_proj.quant_method) for site in self.sites
        ]
        try:
            for site, (projection, method) in zip(self.sites, projections):
                # Instance attributes that shadow the classes' methods.
                table, layer = site.table, site.layer
                table.finalize_prefetch = site.finalize(  # type: ignore[method-assign]
                    table.finalize_prefetch
                )
                layer._dequantize_embeddings = site.dequantize(  # type: ignore[method-assign,assignment]
                    layer._dequantize_embeddings
                )
                assert method is not None
                # PaddedProjection stands in for the quant method (duck-typed).
                padded = PaddedProjection(method, site.out, PAD_ROWS)
                projection.quant_method = padded  # type: ignore[assignment]
            yield
        finally:
            for site, (projection, method) in zip(self.sites, projections):
                # The instance attributes shadowed the classes' methods.
                site.table.__dict__.pop("finalize_prefetch", None)
                site.layer.__dict__.pop("_dequantize_embeddings", None)
                projection.quant_method = method

    def warm_up(self) -> int:
        """Run every gather once on zeros (collective over each table's
        group); returns how many."""
        for site in self.sites:
            ngram = site.ngram
            device = site.out.device
            rows = ngram.ngram_heads
            lookup = torch.zeros(
                1, rows, ngram.head_dim, dtype=torch.float8_e4m3fn, device=device
            )
            # The same ids on every rank, as every call needs: row 0, which
            # the first rank owns.
            ids = torch.zeros(1, rows, dtype=torch.int64, device=device)
            site.gather(lookup, ids, site.out[:1])
        if self.sites:
            torch.accelerator.synchronize(self.sites[0].out.device)
        return len(self.sites)
