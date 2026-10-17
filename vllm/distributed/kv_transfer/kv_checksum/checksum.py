# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connector-agnostic checksums of KV cache blocks.

A checksum covers one block position of one KV cache group, over the
unsharded model (all layers of the group, all KV heads):

1. A slice is the KV of one layer and one KV head in the block: the K and V
   of the block's tokens for that head. Laid out token by token and split
   into little-endian 8-byte pieces, its checksum is::

       s = sum_j x_j * (2 * j + 1)  mod 2**64

   where piece ``x_j`` is piece ``p`` of the token at position ``t`` in the
   request, ``j = t * P + p`` with ``P`` pieces per token.

2. The block's checksum adds up the slices of every layer and head, each
   times its own odd factor::

       c = sum_{layer, head} m * s  mod 2**64,  m = 2 * (layer * H + head) + 1

   where ``H = MAX_KV_HEADS_PER_LAYER``, a fixed bound on any layer's KV
   heads, so that ``m`` does not depend on the model or how it is split.
   Layers are numbered by the layer number in their names and heads
   globally across TP; several caches of one layer (e.g. an attention's KV
   and its indexer's) number their heads consecutively, in layer name order.

Odd factors make any change to a single piece change ``c``, and distinct
factors make misplaced data (swapped tokens, blocks, heads or layers) change
it too. Numbering pieces from the request's start makes the checksums of
consecutive blocks add up to that of the tokens they hold, so a producer and
a consumer with different block sizes compare sums over the same tokens.

A slice always lies whole on one TP rank, and each worker weighs its slices
by their global layer and head, whatever its shard or memory layout. The
parts from all TP ranks and PP stages therefore sum to ``c``, and a producer
and a consumer can compare one value per (group, block position) regardless
of their TP sizes, PP sizes or KV cache layouts.

Replicated KV (MLA, or GQA with more TP ranks than KV heads) is tracked per
replica: a partial checksum is keyed by ``(replicas, replica_index)`` of the
layers it covers, so each consumer replica can be checked on its own.

Mamba state is not per token: a group's block is the request's running
state, at position 0. A layer whose TP split is known
(``MambaSpec.state_splits``) has one slice per global row of its states,
the TP-split dimension, whose pieces are its 2- or 4-byte elements read
unsigned; other Mamba layers have one slice per TP shard, which only
producers and consumers of equal TP sizes can compare.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if TYPE_CHECKING:
    from vllm.config import VllmConfig

MAX_KV_HEADS_PER_LAYER = 1 << 16
"""``H`` in the slice factor; groups with more KV heads per layer are not
checksummed."""

ReplicaKey = tuple[int, int]
"""``(replicas, replica_index)`` of the layers a partial checksum covers."""


@dataclass
class KVChecksumBlocks:
    """The blocks of one request to checksum.

    Attributes:
        num_tokens: Tokens whose KV the blocks hold; slots of the last block at
            or past this count are not checksummed.
        groups: Per KV cache group id, the position of the first block and the
            ids of the blocks at consecutive positions from there.

    """

    num_tokens: int
    groups: dict[int, tuple[int, list[int]]]


@dataclass
class KVChecksumScheduled:
    """The requests to checksum, by request id, set by the scheduler each step."""

    reqs_to_send: dict[str, KVChecksumBlocks] = field(default_factory=dict)
    """Scheduled to be checksummed after this step's forward, when their KV is final."""

    reqs_to_recv: dict[str, KVChecksumBlocks] = field(default_factory=dict)
    """Scheduled to be checksummed once the KV load started this step finishes."""


@dataclass
class KVChecksumGroup:
    """One worker's partial checksums of one request's blocks in one group.

    Attributes:
        start: Position of the first row.
        replica_keys: The replica key of each column.
        values: ``(num_positions, len(replica_keys))`` uint64 partial checksums.

    """

    start: int
    replica_keys: list[ReplicaKey]
    values: np.ndarray


@dataclass
class KVChecksumRecord:
    """One worker's checksums for one step.

    Attributes:
        checksums: Partial checksums per request and group. Every request the
            worker processed this step has an entry, empty if none of its
            groups has layers here, so the scheduler can tell which workers
            reported it.
        skipped_groups: Groups with layers on this worker that it cannot
            checksum.
        fingerprints: Per transfer group, the sum of a hash of each of this
            worker's layers in it (name, type and dtype). Summed over PP stages it
            identifies the group's layers.

    """

    pp_rank: int
    tp_rank: int
    checksums: dict[str, dict[int, KVChecksumGroup]]
    skipped_groups: frozenset[int]
    fingerprints: dict[int, int]
    received: dict[str, bytes] = field(default_factory=dict)
    """Producer checksums of loads, per request, received by connectors that
    carry them in their own messages; reported by the first worker only (PP
    and TP rank 0), as every worker receives them."""


def num_valid_tokens(position: int, block_size: int, num_tokens: int) -> int:
    """How many of the first ``num_tokens`` tokens block ``position`` holds."""
    return min(block_size, max(0, num_tokens - position * block_size))


def kv_checksum_platform_supported() -> bool:
    """Whether this platform can run the KV checksum kernel."""
    return HAS_TRITON and (
        current_platform.is_cuda_alike() or current_platform.is_xpu()
    )


def kv_checksum_enabled(vllm_config: "VllmConfig") -> bool:
    """Whether KV checksums are enabled; ``VllmConfig`` rejects setups that do
    not support them."""
    kv_transfer_config = vllm_config.kv_transfer_config
    return kv_transfer_config is not None and kv_transfer_config.enable_kv_checksum
