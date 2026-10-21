# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A decode step's token embedding from a full copy of the table on every rank.

Under tensor parallelism vLLM shards the token embedding table by vocabulary
(``VocabParallelEmbedding``): each rank looks up the tokens its shard holds,
writes zeros for the others, and an all-reduce sums the ranks' rows. The
DeepSeek-V4.1 backbone then copies the embedding into its ``hc_mult``
hyper-connection streams ([n, H] -> [n, hc, H]) at the first layer. At a
few rows both are fixed costs on the critical path of every step.

``ReplicatedEmbedding`` keeps a full copy of the table on every rank: the
ranks broadcast their shards into it once, when the decode executor is built
(1.32 GB a GPU for DeepSeek-V4.1-Flash), which is before the KV cache is
sized, so the memory profile counts it. A captured decode step then looks its
tokens up in one launch of ``StreamEmbedding`` that writes the embedding (for
the target's first layer, whose pre-mix reads it) and its streams, and needs
no collective. A token outside the vocabulary gets zeros, as the sharded
lookup's sum gives it, so the result equals the all-reduced embedding (up to
the sign of a zero, which the sum makes positive).

The copy serves the table as it was when the executor was built. Loading
weights again rewrites the shards in place but not the copy, so a deployment
that updates weights at runtime must not use the specialized decode.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import make_fake_stream, make_ptr
from torch import nn

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

from ...ops.cute_dsl.primitives import current_cu_stream

THREADS = 256
VECTOR_BYTES = 16
MAX_CTAS = 1024


class StreamEmbedding:
    """Rows of a ``[vocab, H]`` BF16 table into an ``[n, H]`` embedding and
    its ``[n, hc_mult, H]`` streams, by 16-byte vectors; a token outside
    ``[0, vocab)`` gets zeros."""

    def __init__(self, hidden: int, hc_mult: int) -> None:
        assert hidden * 2 % VECTOR_BYTES == 0
        self.row_vectors = hidden * 2 // VECTOR_BYTES
        self.hc_mult = hc_mult

    @cute.jit
    def __call__(
        self,
        ids_ptr: cute.Pointer,
        table_ptr: cute.Pointer,
        embedding_ptr: cute.Pointer,
        streams_ptr: cute.Pointer,
        rows: Int32,
        vocab: Int32,
        num_ctas: Int32,
        stream: cuda.CUstream,
    ):
        """Every tensor as int32 words; a row's vector ``v`` is the column
        ``v`` of four words."""
        row_vectors = self.row_vectors
        self.kernel(
            cute.make_tensor(ids_ptr, cute.make_layout((rows,))),
            cute.make_tensor(table_ptr, cute.make_layout((4, row_vectors, vocab))),
            cute.make_tensor(embedding_ptr, cute.make_layout((4, row_vectors, rows))),
            cute.make_tensor(
                streams_ptr, cute.make_layout((4, row_vectors, self.hc_mult, rows))
            ),
            rows,
            vocab,
            num_ctas,
        ).launch(grid=(num_ctas, 1, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        ids: cute.Tensor,
        table: cute.Tensor,
        embedding: cute.Tensor,
        streams: cute.Tensor,
        rows: Int32,
        vocab: Int32,
        num_ctas: Int32,
    ):
        """Thread t of CTA c takes the (row, vector) items c * THREADS + t,
        then every num_ctas * THREADS after it."""
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        vector = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128
        )
        words = cute.make_rmem_tensor((4,), Int32)
        row_vectors = self.row_vectors
        for item in cutlass.range(
            cta * THREADS + thread, rows * row_vectors, num_ctas * THREADS, unroll=1
        ):
            row = item // row_vectors
            v = item - row * row_vectors
            token = ids[row]
            words.fill(0)
            if token >= 0:  # noqa: SIM102
                if token < vocab:
                    cute.copy(vector, table[None, v, token], words)
            cute.copy(vector, words, embedding[None, v, row])
            for c in cutlass.range_constexpr(self.hc_mult):
                cute.copy(vector, words, streams[None, v, c, row])


def _pointer(address: int, align: int = VECTOR_BYTES):
    """A CuTe DSL global-memory pointer to int32 words."""
    return make_ptr(Int32, address, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _kernel(device_index: int, hidden: int, hc_mult: int):
    """``StreamEmbedding`` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            StreamEmbedding(hidden, hc_mult),
            _pointer(16, 4),
            _pointer(16),
            _pointer(16),
            _pointer(16),
            Int32(1),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


def embed_streams(
    ids: torch.Tensor,
    table: torch.Tensor,
    embedding: torch.Tensor,
    streams: torch.Tensor,
) -> None:
    """Write ``table``'s rows of ``ids`` (int32 ``[n]``) into ``embedding``
    (``[n, H]``) and ``streams`` (``[n, hc, H]``), on the current stream;
    zeros for an id outside the table."""
    rows, hc_mult, hidden = streams.shape
    assert ids.dtype == torch.int32 and ids.shape == (rows,) and ids.is_contiguous()
    assert table.dtype == torch.bfloat16 and table.shape[1] == hidden
    assert table.is_contiguous() and streams.is_contiguous()
    assert streams.dtype == torch.bfloat16
    assert embedding.shape == (rows, hidden) and embedding.is_contiguous()
    assert embedding.dtype == torch.bfloat16
    if rows == 0:
        return
    items = rows * (hidden * 2 // VECTOR_BYTES)
    num_ctas = min((items + THREADS - 1) // THREADS, MAX_CTAS)
    _kernel(table.device.index, hidden, hc_mult)(
        _pointer(ids.data_ptr(), 4),
        _pointer(table.data_ptr()),
        _pointer(embedding.data_ptr()),
        _pointer(streams.data_ptr()),
        Int32(rows),
        Int32(table.shape[0]),
        Int32(num_ctas),
        current_cu_stream(),
    )


class ReplicatedEmbedding:
    """A full copy of a vocabulary-parallel embedding table on every TP rank,
    and a decode step's lookup from it."""

    @classmethod
    def create(cls, embed: nn.Module, hc_mult: int) -> "ReplicatedEmbedding | None":
        """The copy of ``embed``'s table, or None when ``embed`` is not the
        plain sharded BF16 lookup this replaces. The decision reads only the
        configuration, which every TP rank shares; building the copy is
        collective."""
        if not isinstance(embed, VocabParallelEmbedding):
            return None
        weight = embed.weight
        indices = embed.shard_indices
        if (
            embed.tp_size == 1
            or not embed.reduce_results
            or embed.parallel_group is not None
            or not isinstance(embed.quant_method, UnquantizedEmbeddingMethod)
            or weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or not weight.is_contiguous()
            or weight.shape[1] * 2 % VECTOR_BYTES
            or indices.added_vocab_end_index != indices.added_vocab_start_index
        ):
            return None
        return cls(embed, hc_mult)

    def __init__(self, embed: VocabParallelEmbedding, hc_mult: int) -> None:
        group = get_tp_group()
        weight = embed.weight.data
        indices = embed.shard_indices
        bounds = torch.tensor(
            [indices.org_vocab_start_index, indices.org_vocab_end_index],
            dtype=torch.int64,
            device=weight.device,
        )
        ranges = group.all_gather(bounds, dim=0).view(group.world_size, 2).tolist()
        vocab = embed.org_vocab_size
        covered = sorted((start, end) for start, end in ranges if end > start)
        assert (
            covered[0][0] == 0
            and covered[-1][1] == vocab
            and all(a[1] == b[0] for a, b in zip(covered, covered[1:]))
        ), f"vocabulary shards {ranges} do not tile [0, {vocab})"
        self.table = weight.new_empty(vocab, weight.shape[1])
        for source, (start, end) in enumerate(ranges):
            rows = self.table[start:end]
            if source == group.rank_in_group:
                rows.copy_(weight[: end - start])
            group.broadcast(rows, src=source)
        self.hc_mult = hc_mult
        # Compile and load the kernel now: steps use it only in captures.
        self(torch.zeros(1, dtype=torch.int32, device=weight.device))
        torch.accelerator.synchronize(weight.device)

    def accepts(self, ids: torch.Tensor) -> bool:
        """Whether ``__call__`` takes ``ids``: a non-empty int32 vector."""
        return (
            ids.dtype == torch.int32
            and ids.dim() == 1
            and ids.shape[0] > 0
            and ids.is_contiguous()
        )

    def __call__(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The embedding ``[n, H]`` and its streams ``[n, hc, H]`` of the
        tokens ``ids``."""
        rows, hidden = ids.shape[0], self.table.shape[1]
        streams = self.table.new_empty(rows, self.hc_mult, hidden)
        embedding = self.table.new_empty(rows, hidden)
        embed_streams(ids, self.table, embedding, streams)
        return embedding, streams
