# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Data classes for MooncakeStoreConnector."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import cast

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import (
    KeyMetadata,
    RankLocalStoreLayout,
    StoreLayout,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
)

logger = init_logger(__name__)


class BlobBlockHashes(Sequence[BlockHash]):
    """Lazy view over a flat buffer of fixed-size block hashes to avoid the overhead
    of materializing all hashes upfront.
    """

    def __init__(self, blob: memoryview, hash_len: int):
        self._blob = blob
        self._hash_len = hash_len
        self._n = len(blob) // hash_len if hash_len else 0

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._n))]
        if idx < 0:
            idx += self._n
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        off = idx * self._hash_len
        return BlockHash(self._blob[off : off + self._hash_len])


class _CompactChunkHashList(BlockHashListWithBlockSize):
    """View that keys each ``block_size`` chunk by the last constituent
    ``hash_block_size`` hash instead of concatenating all of them.

    The engine chains block hashes (each hash folds in the previous one), so the
    final sub-block hash of a chunk already uniquely identifies the whole chunk
    and its prefix. Using it keeps a Mooncake key at a single hash digest
    regardless of the ``block_size`` / ``hash_block_size`` ratio, instead of
    growing the key linearly with it (e.g. 64x for ``block_size=256``,
    ``hash_block_size=4``).
    """

    def __init__(
        self,
        block_hashes: Sequence[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        # Accept any indexable sequence (e.g. the lazy ``BlobBlockHashes``), not
        # just ``list``; the base only indexes/sizes it.
        assert target_block_size % hash_block_size == 0
        self.block_hashes = block_hashes  # type: ignore[assignment]
        self.scale_factor = target_block_size // hash_block_size

    def _get_value_at(self, idx: int) -> BlockHash:
        return self.block_hashes[idx * self.scale_factor + self.scale_factor - 1]


def chunk_hashes_for_block_size(
    block_hashes: Sequence[BlockHash],
    hash_block_size: int,
    block_size: int,
) -> Sequence[BlockHash]:
    """Map ``hash_block_size``-granular block hashes to one compact hash per
    ``block_size`` chunk (the chunk's last sub-hash). Returns ``block_hashes``
    unchanged when the two sizes are equal.
    """
    if block_size == hash_block_size:
        return block_hashes
    # Structurally a Sequence[BlockHash] (indexable + sized); the base class
    # just isn't declared as one.
    return cast(
        "Sequence[BlockHash]",
        _CompactChunkHashList(block_hashes, hash_block_size, block_size),
    )


class ChunkedTokenDatabase:
    """Enumerates Store chunks and their prefix hashes."""

    def __init__(
        self,
        metadata: KeyMetadata,
        chunk_size: int,
        hash_block_size: int | None = None,
        store_layout: StoreLayout | None = None,
    ):
        self.metadata = metadata
        self.chunk_size = chunk_size
        self.hash_block_size = hash_block_size or chunk_size
        if self.chunk_size % self.hash_block_size != 0:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be a multiple of "
                f"hash_block_size ({self.hash_block_size})"
            )
        self.store_layout = store_layout or RankLocalStoreLayout(
            metadata, chunk_size, self.hash_block_size
        )

    @property
    def kv_caches_base_addr(self) -> list[int]:
        return self._rank_local_layout().kv_caches_base_addr

    @property
    def block_len(self) -> list[int]:
        return self._rank_local_layout().block_len

    def _rank_local_layout(self) -> RankLocalStoreLayout:
        if not isinstance(self.store_layout, RankLocalStoreLayout):
            raise RuntimeError("This operation requires a rank-local Store layout")
        return self.store_layout

    def key_for(self, chunk_hash: BlockHash) -> str:
        return self._rank_local_layout().key_for(0, chunk_hash)

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        self._rank_local_layout().set_kv_caches_base_addr(kv_caches_base_addr)

    def set_block_len(self, block_len: list[int]):
        self._rank_local_layout().set_block_len(block_len)

    def prepare_value_for_block(self, block_id: int) -> tuple[list[int], list[int]]:
        """Return addresses and sizes for one physical block slot."""
        return self._rank_local_layout().prepare_value_for_block(block_id)

    def process_tokens(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        mask_num: int = 0,
        *,
        chunk_mask: list[bool] | None = None,
        put_step: int = 1,
        put_step_rank: int = 0,
    ) -> Iterable[tuple[int, int, BlockHash]]:
        """Process tokens and yield (start_idx, end_idx, block_hash) tuples.

        When there are fewer KV heads than TP ranks, chunks are distributed
        across TP ranks to avoid duplicate load/store. The assignment keys off
        the absolute ``chunk_id`` so a given chunk always lands on the same
        rank regardless of where the processed suffix begins.

        Args:
            token_len: Total number of tokens. Must be hash-block aligned and
                covered by ``block_hashes`` when hashes are present.
            block_hashes: Block hashes computed at ``hash_block_size`` granularity.
                A chunk spanning multiple hashes is keyed by its last hash.
            mask_num: Number of tokens to skip from the beginning.
            chunk_mask: Optional mask over chunks after ``mask_num``.
            put_step: Stride for distributing chunks across ranks.
            put_step_rank: ``chunk_id % put_step`` value this rank stores.

        """
        assert put_step > 0
        if not block_hashes:
            return
        assert token_len % self.hash_block_size == 0
        assert token_len // self.hash_block_size <= len(block_hashes)
        start_chunk = max(0, cdiv(mask_num, self.chunk_size))
        max_chunks = cdiv(token_len, self.chunk_size)
        if chunk_mask is not None:
            max_chunks = min(max_chunks, start_chunk + len(chunk_mask))
        for chunk_id in range(start_chunk, max_chunks):
            if chunk_mask is not None and not chunk_mask[chunk_id - start_chunk]:
                continue
            if chunk_id % put_step != put_step_rank:
                continue
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, token_len)
            h = block_hashes[end_idx // self.hash_block_size - 1]
            yield start_idx, end_idx, h


@dataclass(frozen=True)
class TailKeyBoundary:
    """Hash boundary used to key a group's tail block in the store.

    Attributes:
        group_id: KV-cache group containing the tail block.
        num_tokens: Token boundary whose prefix hash identifies the matched
            stored block. The loader uses
            ``block_hashes[num_tokens // hash_block_size - 1]`` instead of the
            hash implied by ``MooncakeLookupResult.hit_length``. This changes
            only the load key, not the reusable prefix.

    """

    group_id: int
    num_tokens: int


@dataclass
class MooncakeLookupResult:
    """Lookup result used to build the subsequent load request.

    Attributes:
        hit_length: Longest prefix that every KV-cache group can reuse after
            their individual cache hits converge.
        tail_key_boundaries: Hash boundary used to store each cache group's
            tail block when ``hit_length`` does not identify its store key.
            There is one entry per group for every nonzero hit.

    """

    hit_length: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


@dataclass
class LoadSpec:
    """Specification for loading KV cache from external store."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    can_load: bool
    token_len: int = 0
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


@dataclass
class RequestTracker:
    """Tracks per-request state across scheduler ticks."""

    req_id: str
    token_len: int
    allocated_block_ids: tuple[list[int], ...]
    num_saved_tokens: int = 0
    token_ids: list[int] | None = None
    has_pending_offload: bool = False
    # Snapshot of the prefill range length at tracker creation time.
    # For a fresh request this is len(prompt). For a resumed-from-preemption
    # request it includes previously-generated tokens, which are re-prefilled.
    prefill_end_tokens: int = 0

    def reset(self) -> None:
        self.token_len = 0
        self.allocated_block_ids = ()
        self.num_saved_tokens = 0
        self.token_ids = None
        self.has_pending_offload = False
        self.prefill_end_tokens = 0

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
    ) -> None:
        # Backward-compat: accept a single list (broadcast to single group).
        if isinstance(new_block_ids, list):
            new_block_ids = (new_block_ids,)
        if len(new_block_ids) != len(self.allocated_block_ids):
            raise ValueError(
                f"Group count mismatch: tracker has "
                f"{len(self.allocated_block_ids)} groups, update has "
                f"{len(new_block_ids)}"
            )
        for existing, new in zip(self.allocated_block_ids, new_block_ids, strict=True):
            if new:
                existing.extend(new)


@dataclass
class ReqMeta:
    """Per-request metadata for store put/get operations."""

    req_id: str
    token_len_chunk: int
    block_ids: tuple[list[int], ...]
    block_hashes: list[BlockHash]

    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    current_event: torch.cuda.Event | None = None

    token_ids: list[int] | None = None
    # Absolute request offset represented by token_ids[0].
    token_ids_start: int = 0
    num_prompt_tokens: int | None = None
    # Identifies this store job for the engine's lifetime. A request id cannot
    # serve that purpose: it is reused once a preempted request resumes, so it
    # would release the wrong job's blocks.
    store_job_id: int | None = None
    # Core-provided (group_id, block_id, boundary_tokens) mamba "align"
    # boundary states. A block-aligned entry is a committed boundary snapshot;
    # a non-aligned entry is the sub-block CoW tail. The store-job reference
    # keeps each exact block alive until every worker rank finishes the job.
    boundary_state_offloads: list[tuple[int, int, int]] | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
    ) -> "ReqMeta | None":
        """Create ReqMeta from a RequestTracker."""
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        token_ids_start = tracker.num_saved_tokens
        chunk_boundary = cdiv(token_ids_start + 1, block_size) * block_size
        num_tokens_to_save = input_token_len // block_size * block_size

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        # A loading step saves new tokens on subsequent scheduled steps.
        if load_spec is not None and load_spec.can_load:
            skip_save = True
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        token_ids = None
        if tracker.token_ids and not skip_save:
            # Scheduler tracking continues while this job is handled by an
            # asynchronous worker, so metadata must own a stable snapshot.
            token_ids = tracker.token_ids[token_ids_start:num_tokens_to_save]

        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            load_spec = None

        logger.debug(
            "request:%s, meta save spec:%s, meta load spec:%s",
            tracker.req_id,
            not skip_save,
            load_spec,
        )
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            token_ids=token_ids,
            token_ids_start=token_ids_start,
            num_prompt_tokens=tracker.prefill_end_tokens,
        )


@dataclass
class MooncakeStoreWorkerMetadata(KVConnectorWorkerMetadata):
    """Maps ``ReqMeta.store_job_id`` to the number of ranks done with that job."""

    completed_saves: dict[int, int] = field(default_factory=dict)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "MooncakeStoreWorkerMetadata":
        assert isinstance(other, MooncakeStoreWorkerMetadata)
        for store_job_id, count in other.completed_saves.items():
            self.completed_saves[store_job_id] = (
                self.completed_saves.get(store_job_id, 0) + count
            )
        return self


class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker."""

    def __init__(
        self,
        unfinished_request_ids: set[str],
        preempted_req_ids: set[str],
    ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)
