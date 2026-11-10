# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The token embedding of the specialized decode at TP > 1: a replicated table
instead of the vocab-parallel lookup and its all-reduce.

The stock token embedding (`VocabParallelEmbedding`) splits the table along
the vocabulary: each TP rank holds a quarter of its rows at TP4, looks up the
step's tokens it owns (zeros for the others, one fused kernel) and all-reduces
the result over the TP group. A decode step embeds four times: the target's
tokens and, in the three MTP draft passes, the drafter's, which shares the
target's embedding module (vLLM shares it with every MTP drafter). Each
all-reduce takes 3.5 to 5.5 us on GB300 and is the step's first collective,
for 20 to 24 us a step of lookups and all-reduces at TP4 (4-GPU CUDA graph
microbenchmark).

`ReplicatedEmbedding` all-gathers the shards once, when the executor is
built, into a full BF16 table on every rank (1.27 GB for Qwen3.8-Flash-Next's
248,320 x 2560 table), points the stock module's shard at that table's slice
(so the rank pays 0.95 GB more, not 1.27 GB), and routes the module's forward
to the same fused lookup kernel over the full vocabulary range: no
all-reduce, 6 us a step for the four lookups. It does so only while a CUDA
graph is being captured: the FULL graphs of decode steps and of the drafter,
and any PIECEWISE graph that captures the embedding (the same rows, without
the collective); every eager forward (prefill, mixed batches, profiling)
keeps the stock lookup and all-reduce. The model state builds the executor
before memory profiling, so the KV cache shrinks by the table's extra memory.

Numerics: every row is the stock one bit for bit, except that the stock
all-reduce turns a negative zero into a positive zero (one rank's value plus
the other ranks' zeros) and the replicated lookup keeps the table's sign.

Every rank decides alike: whether a CUDA graph is being captured is the same
on every rank, and nothing is exchanged.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

logger = init_logger(__name__)


def replication_unsupported(embed: torch.nn.Module) -> str | None:
    """Why `embed` cannot take the replicated table, or None."""
    if not isinstance(embed, VocabParallelEmbedding):
        return f"the embedding is a {type(embed).__name__}"
    if embed.tp_size == 1:
        return "TP 1 has no all-reduce"
    if embed.parallel_group is not None or not embed.reduce_results:
        return "the embedding is not reduced over the TP group"
    if not isinstance(embed.quant_method, UnquantizedEmbeddingMethod):
        return "the embedding is quantized"
    if not embed.use_fused_embedding:
        return "the embedding does not take the fused lookup"
    if embed.num_added_embeddings:
        return "the embedding has added vocabulary"
    weight = embed.weight
    if (
        weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or not weight.is_contiguous()
        or not weight.is_cuda
    ):
        return "the embedding weight is not a contiguous BF16 CUDA matrix"
    return None


class ReplicatedEmbedding:
    """The full table of a vocab-parallel token embedding on every rank, and
    the module's forward over it while a CUDA graph is captured (module
    docstring). Construction is collective over the TP group.

    Args:
        embed: The target's `VocabParallelEmbedding` (TP > 1, unquantized
            BF16, no added vocabulary: `replication_unsupported`).

    """

    def __init__(self, embed: VocabParallelEmbedding) -> None:
        reason = replication_unsupported(embed)
        assert reason is None, reason
        self.embed = embed
        group = get_tp_group()
        shard = embed.weight.data
        rows = embed.num_embeddings_per_partition
        assert shard.shape[0] == rows and group.world_size == embed.tp_size
        self.table = torch.empty(
            embed.tp_size * rows, shard.shape[1], dtype=shard.dtype, device=shard.device
        )
        dist.all_gather_into_tensor(self.table, shard, group=group.device_group)
        start = embed.tp_rank * rows
        own = self.table[start : start + rows]
        assert torch.equal(own, shard)
        # The module's shard becomes a view of the table: the old storage is
        # freed, and every reader of the weight (the LM head when tied) sees
        # the same values.
        embed.weight.data = own
        self.vocab = embed.org_vocab_size
        self._stock: Callable[[torch.Tensor], torch.Tensor] = embed.forward
        # The instance attribute shadows the class's forward for this module.
        embed.forward = self.forward  # type: ignore[method-assign]

    @property
    def extra_bytes(self) -> int:
        """Bytes the table adds to the rank's shard."""
        rows = self.embed.num_embeddings_per_partition
        return (self.table.shape[0] - rows) * self.table.shape[1] * 2

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """The module's output for token ids `input_`: the replicated
        lookup while a CUDA graph is captured, the stock lookup and all-reduce
        otherwise."""
        if not torch.cuda.is_current_stream_capturing():
            return self._stock(input_)
        ids = input_ if input_.dim() == 1 else input_.reshape(-1)
        out = ops.vocab_parallel_embedding(
            ids, self.table, 0, self.vocab, 0, self.vocab, self.vocab
        )
        return out if input_.dim() == 1 else out.view(*input_.shape, out.shape[-1])


def replicate_embedding(embed: torch.nn.Module) -> ReplicatedEmbedding | None:
    """The replicated table of the target's token embedding `embed` at TP >
    1, or None (logged once) when the module does not take it. Collective over
    the TP group: every rank decides from the same config."""
    reason = replication_unsupported(embed)
    if reason is not None:
        if getattr(embed, "tp_size", 1) > 1:
            logger.info_once(
                "The Qwen4Exp specialized decode keeps the stock token embedding: %s.",
                reason,
            )
        return None
    assert isinstance(embed, VocabParallelEmbedding)
    replicated = ReplicatedEmbedding(embed)
    logger.info_once(
        "The Qwen4Exp specialized decode replicates the token embedding "
        "(%d x %d, %.2f GB more a rank) for the lookups in CUDA graphs.",
        replicated.table.shape[0],
        replicated.table.shape[1],
        replicated.extra_bytes / 1e9,
    )
    return replicated
