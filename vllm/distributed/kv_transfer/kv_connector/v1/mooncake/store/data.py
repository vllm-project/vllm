# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Data classes for MooncakeStoreConnector."""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import cast

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    derive_mamba_conv_split,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import is_non_overlapping_and_dense
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

logger = init_logger(__name__)


def _schema_fingerprint(parts: tuple[object, ...]) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


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


@dataclass
class KeyMetadata:
    """Metadata for constructing pool keys."""

    model_name: str
    tp_rank: int
    pcp_rank: int
    dcp_rank: int
    pp_rank: int
    group_id: int = 0
    # Optional namespace prepended to every key. Lets separate deployments
    # share one Mooncake master without colliding on identical block hashes.
    # Empty (the default) keeps keys byte-identical to the unprefixed format.
    cache_prefix: str = ""
    # Complete namespace for an opt-in Store payload format. Empty keeps
    # historical keys byte-identical.
    store_namespace: str = ""


@dataclass(order=True)
class PoolKey:
    """Key for addressing KV cache blocks in the distributed store."""

    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.key_metadata.cache_prefix,
                self.key_metadata.model_name,
                self.key_metadata.store_namespace,
                self.key_metadata.tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.key_metadata.group_id,
                self.chunk_hash,
            )
        )

    @staticmethod
    def build_prefix(
        key_metadata: KeyMetadata,
        *,
        tp_rank: int | None = None,
        pcp_rank: int | None = None,
        dcp_rank: int | None = None,
        pp_rank: int | None = None,
    ) -> str:
        """Return the stable prefix for a Mooncake pool key."""
        prefix = f"{key_metadata.cache_prefix}@" if key_metadata.cache_prefix else ""
        return (
            f"{prefix}"
            f"{key_metadata.model_name}"
            f"{key_metadata.store_namespace}"
            f"@tp_rank:{key_metadata.tp_rank if tp_rank is None else tp_rank}"
            f"@pcp{key_metadata.pcp_rank if pcp_rank is None else pcp_rank}"
            f"@dcp{key_metadata.dcp_rank if dcp_rank is None else dcp_rank}"
            f"@pp_rank:{key_metadata.pp_rank if pp_rank is None else pp_rank}"
            f"@group:{key_metadata.group_id}"
        )

    @staticmethod
    def build_key_string(key_prefix: str, chunk_hash: str) -> str:
        return f"{key_prefix}@{chunk_hash}"

    def to_string(self) -> str:
        return self.build_key_string(
            self.build_prefix(self.key_metadata), self.chunk_hash
        )


StoreShardId = int
# Physical ranks encoded in a rank-local key: (TP, PCP, DCP, PP).
RankNamespace = tuple[int, int, int, int]


class StoreLayout:
    """Store payload layout for one KV-cache group on one local TP rank."""

    store_format = "rank_local"

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
    ) -> None:
        self.metadata = metadata
        self.block_size = block_size
        self.hash_block_size = hash_block_size

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        """Store objects contributed or consumed by this local rank."""
        raise NotImplementedError

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        raise NotImplementedError

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        """Prefixes that must exist for a logical block to be reusable."""
        raise NotImplementedError

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
        num_blocks: int,
    ) -> None:
        """Build descriptors from the local KV cache tensors."""
        raise NotImplementedError

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: list[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        """Build Store multi-buffer descriptors for logical chunks."""
        raise NotImplementedError


class RankLocalStoreLayout(StoreLayout):
    """Rank-local Store payload layout."""

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
    ) -> None:
        super().__init__(metadata, block_size, hash_block_size)
        self._key_prefix = PoolKey.build_prefix(metadata)
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        return (0,)

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        assert shard_id == 0
        return PoolKey.build_key_string(self._key_prefix, chunk_hash.hex())

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        return tuple(
            PoolKey.build_prefix(
                self.metadata,
                tp_rank=tp_rank,
                pcp_rank=pcp_rank,
                dcp_rank=dcp_rank,
                pp_rank=pp_rank,
            )
            for tp_rank, pcp_rank, dcp_rank, pp_rank in rank_namespaces
        )

    def set_kv_caches_base_addr(self, base_addrs: list[int]) -> None:
        self.kv_caches_base_addr = base_addrs

    def set_block_len(self, block_lens: list[int]) -> None:
        self.block_len = block_lens

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
        num_blocks: int,
    ) -> None:
        seen_region_ptrs: set[int] = set()
        base_addrs: list[int] = []
        block_lens: list[int] = []
        for cache in kv_caches:
            storage = cache.untyped_storage()
            storage_addr = storage.data_ptr()
            storage_len = storage.nbytes()
            if not is_non_overlapping_and_dense(cache[0]):
                for head_idx in range(cache.shape[1]):
                    head_cache = cache[:, head_idx]
                    assert is_non_overlapping_and_dense(head_cache[0])
                    region_addr = head_cache.data_ptr()
                    if region_addr in seen_region_ptrs:
                        continue
                    seen_region_ptrs.add(region_addr)
                    base_addrs.append(region_addr)
                    block_lens.append(head_cache.stride(0) * head_cache.element_size())
            elif cache.stride(0) * cache.element_size() * num_blocks == storage_len:
                if storage_addr in seen_region_ptrs:
                    continue
                seen_region_ptrs.add(storage_addr)
                base_addrs.append(storage_addr)
                block_lens.append(storage_len // num_blocks)
            else:
                region_addr = cache.data_ptr()
                if region_addr in seen_region_ptrs:
                    continue
                seen_region_ptrs.add(region_addr)
                base_addrs.append(region_addr)
                block_lens.append(cache.stride(0) * cache.element_size())
        self.kv_caches_base_addr = base_addrs
        self.block_len = block_lens

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: list[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        if not chunks:
            return [], [], []
        assert len(chunks) == len(shard_ids) and all(
            shard_id == 0 for shard_id in shard_ids
        )
        base = np.asarray(self.kv_caches_base_addr, dtype=np.int64)
        length = len(self.block_len)
        blen = np.asarray(
            [self.block_len[i % length] for i in range(base.shape[0])],
            dtype=np.int64,
        )
        n = len(chunks)
        starts = np.fromiter((c[0] for c in chunks), dtype=np.int64, count=n)
        spans = np.fromiter((c[1] for c in chunks), dtype=np.int64, count=n) - starts
        assert np.all(spans % self.hash_block_size == 0)
        bids = np.fromiter(
            (block_ids[i] for i in (starts // self.block_size).tolist()),
            dtype=np.int64,
            count=n,
        )
        addrs = base[None, :] + bids[:, None] * blen[None, :]
        block_counts = (spans + self.block_size - 1) // self.block_size
        sizes = blen[None, :] * block_counts[:, None]
        return addrs.tolist(), sizes.tolist(), bids.tolist()

    def prepare_value_for_block(self, block_id: int) -> tuple[list[int], list[int]]:
        length = len(self.block_len)
        return (
            [
                base_addr + block_id * self.block_len[index % length]
                for index, base_addr in enumerate(self.kv_caches_base_addr)
            ],
            [
                self.block_len[index % length]
                for index in range(len(self.kv_caches_base_addr))
            ],
        )


class TPShardedStoreLayout(StoreLayout):
    """Store layout shared by divisible TP sizes."""

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
        local_tp_size: int,
        store_tp_size: int,
        tp_rank: int,
        store_chunk_size: int | None = None,
    ) -> None:
        super().__init__(metadata, block_size, hash_block_size)
        self.store_chunk_size = store_chunk_size or block_size
        if block_size % self.store_chunk_size:
            raise ValueError("Store chunk size must divide the local block size")
        self.shards_per_rank = store_tp_size // local_tp_size
        first_shard = tp_rank * self.shards_per_rank
        self.store_shard_ids = tuple(
            range(first_shard, first_shard + self.shards_per_rank)
        )
        self.store_tp_size = store_tp_size
        self._shard_key_prefixes = {
            shard_id: PoolKey.build_prefix(metadata, tp_rank=shard_id)
            for shard_id in self.store_shard_ids
        }
        self._chunk_addr_bases: list[np.ndarray] = []
        self._chunk_block_strides: list[np.ndarray] = []
        self._chunk_sizes: list[np.ndarray] = []

    @property
    def local_shard_ids(self) -> tuple[StoreShardId, ...]:
        return self.store_shard_ids

    def key_for(self, shard_id: StoreShardId, chunk_hash: BlockHash) -> str:
        return PoolKey.build_key_string(
            self._shard_key_prefixes[shard_id], chunk_hash.hex()
        )

    def lookup_key_prefixes(
        self,
        rank_namespaces: Sequence[RankNamespace],
    ) -> tuple[str, ...]:
        pp_ranks = sorted({namespace[3] for namespace in rank_namespaces})
        return tuple(
            PoolKey.build_prefix(self.metadata, tp_rank=shard_id, pp_rank=pp_rank)
            for shard_id in range(self.store_tp_size)
            for pp_rank in pp_ranks
        )

    def _set_segment_templates(
        self,
        templates: Sequence[Sequence[tuple[list[int], list[int], list[int]]]],
    ) -> None:
        self._chunk_addr_bases = [
            np.asarray([template[0] for template in chunk], dtype=np.uint64)
            for chunk in templates
        ]
        self._chunk_block_strides = [
            np.asarray([template[1] for template in chunk], dtype=np.uint64)
            for chunk in templates
        ]
        self._chunk_sizes = [
            np.asarray([template[2] for template in chunk], dtype=np.uint64)
            for chunk in templates
        ]

    def prepare_values(
        self,
        chunks: Sequence[tuple[int, int]],
        block_ids: list[int],
        shard_ids: Sequence[StoreShardId],
    ) -> tuple[list[list[int]], list[list[int]], list[int]]:
        if not chunks:
            return [], [], []
        if len(chunks) != len(shard_ids):
            raise ValueError("Each Store chunk must have one shard ID")

        first_shard = self.store_shard_ids[0]
        addrs: list[list[int]] = []
        sizes: list[list[int]] = []
        selected_block_ids: list[int] = []
        for (start, end), shard_id in zip(chunks, shard_ids, strict=True):
            if (
                start % self.store_chunk_size
                or not 0 < end - start <= self.store_chunk_size
            ):
                raise ValueError("Invalid Store chunk boundary")
            block_index, block_offset = divmod(start, self.block_size)
            chunk_index = block_offset // self.store_chunk_size
            local_shard = shard_id - first_shard
            if not 0 <= local_shard < self.shards_per_rank:
                raise ValueError("Store shard is not owned by this TP rank")
            if block_index >= len(block_ids):
                raise ValueError("Store chunk has no local cache block")
            block_id = block_ids[block_index]
            bases = self._chunk_addr_bases[chunk_index][local_shard]
            strides = self._chunk_block_strides[chunk_index][local_shard]
            addrs.append((bases + block_id * strides).tolist())
            sizes.append(self._chunk_sizes[chunk_index][local_shard].tolist())
            selected_block_ids.append(block_id)
        return addrs, sizes, selected_block_ids


class AttentionStoreLayout(TPShardedStoreLayout):
    store_format: str

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
        local_tp_size: int,
        store_tp_size: int,
        tp_rank: int,
        layer_specs: Sequence[AttentionSpec],
        store_chunk_size: int | None = None,
    ) -> None:
        super().__init__(
            metadata,
            block_size,
            hash_block_size,
            local_tp_size,
            store_tp_size,
            tp_rank,
            store_chunk_size,
        )
        self.layer_specs = tuple(layer_specs)
        self.chunks_per_block = block_size // self.store_chunk_size
        global_head_slots = {
            spec.num_heads * local_tp_size for spec in self.layer_specs
        }
        if len(global_head_slots) != 1:
            raise ValueError("Attention layers in a Store group need equal head slots")
        global_num_head_slots = global_head_slots.pop()
        if global_num_head_slots % store_tp_size:
            raise ValueError(
                "Attention head slots must be divisible by the Store shard count"
            )
        self.heads_per_store_shard = global_num_head_slots // store_tp_size

    @staticmethod
    def resolve_store_shard_count(
        layer_specs: Sequence[AttentionSpec],
        local_tp_size: int,
        requested_store_tp_size: int,
    ) -> int | None:
        global_head_slots = {spec.num_heads * local_tp_size for spec in layer_specs}
        if len(global_head_slots) != 1:
            return None
        global_num_head_slots = global_head_slots.pop()
        smaller = min(requested_store_tp_size, global_num_head_slots)
        larger = max(requested_store_tp_size, global_num_head_slots)
        if larger % smaller or smaller < local_tp_size or smaller % local_tp_size:
            return None
        return smaller

    @staticmethod
    def resolve_store_chunk_size(
        layer_specs: Sequence[AttentionSpec],
        requested_store_chunk_size: int | None = None,
    ) -> int | None:
        if requested_store_chunk_size is not None:
            for spec in layer_specs:
                states = Fraction(requested_store_chunk_size, 1) / spec.tokens_per_state
                if (
                    spec.block_size % requested_store_chunk_size
                    or states.denominator != 1
                ):
                    return None
            return requested_store_chunk_size

        block_sizes = {spec.block_size for spec in layer_specs}
        return block_sizes.pop() if len(block_sizes) == 1 else None

    @staticmethod
    def schema_fingerprint(
        layer_specs: Sequence[AttentionSpec],
        local_tp_size: int,
        store_shard_count: int,
        store_chunk_size: int,
    ) -> str:
        global_num_head_slots = layer_specs[0].num_heads * local_tp_size
        heads_per_store_shard = global_num_head_slots // store_shard_count
        layers = tuple(
            (
                type(spec).__name__,
                store_chunk_size,
                str(Fraction(store_chunk_size, 1) / spec.tokens_per_state),
                spec.state_content_size_bytes,
                heads_per_store_shard,
                str(spec.dtype),
                str(spec.kv_quant_mode),
                str(getattr(spec, "cache_dtype_str", None)),
                str(spec.tokens_per_state),
            )
            for spec in layer_specs
        )
        return _schema_fingerprint(layers)

    @staticmethod
    def _state_range(
        spec: AttentionSpec, store_chunk_size: int, chunk_index: int
    ) -> tuple[int, int]:
        start = Fraction(chunk_index * store_chunk_size, 1) / spec.tokens_per_state
        end = Fraction((chunk_index + 1) * store_chunk_size, 1) / (
            spec.tokens_per_state
        )
        if start.denominator != 1 or end.denominator != 1:
            raise ValueError("Store chunk boundaries must align with attention states")
        return start.numerator, end.numerator

    @staticmethod
    def _cache_strides(
        cache: torch.Tensor, spec: AttentionSpec, num_blocks: int
    ) -> tuple[int, int, int, int, int, int, int]:
        if cache.ndim == 4:
            cache = cache.unsqueeze(1)
        if cache.ndim != 5 or cache.shape[0] != num_blocks:
            raise ValueError(
                "TP-shared attention cache needs logical shape "
                "(blocks, kernel_blocks, heads, states, content)"
            )
        _, kernel_blocks, num_heads, num_states, content = cache.shape
        if num_heads != spec.num_heads:
            raise ValueError("Attention cache head slots do not match its spec")
        expected_states = spec.get_num_kernel_states(spec.block_size)
        if kernel_blocks * num_states != expected_states:
            raise ValueError("Attention cache state count does not match its spec")
        element_size = cache.element_size()
        block_stride, kernel_stride, head_stride, state_stride, content_stride = (
            stride * element_size for stride in cache.stride()
        )
        content_bytes = content * element_size
        if (
            content_stride != element_size
            or content_bytes != spec.state_content_size_bytes
        ):
            raise ValueError("TP-shared Mooncake store requires packed KV content")
        return (
            block_stride,
            kernel_stride,
            head_stride,
            state_stride,
            kernel_blocks,
            num_states,
            content_bytes,
        )


class HeadMajorStoreLayout(AttentionStoreLayout):
    """Store descriptors with layer/head/state value ordering."""

    store_format = "attention_head_major"

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
        num_blocks: int,
    ) -> None:
        templates: list[list[tuple[list[int], list[int], list[int]]]] = [
            [([], [], []) for _ in range(self.shards_per_rank)]
            for _ in range(self.chunks_per_block)
        ]
        for cache, spec in zip(kv_caches, self.layer_specs, strict=True):
            (
                block_stride,
                kernel_stride,
                head_stride,
                state_stride,
                kernel_blocks,
                num_states,
                content_bytes,
            ) = self._cache_strides(cache, spec, num_blocks)
            if state_stride != content_bytes:
                raise ValueError(
                    "TP-shared Mooncake store requires a packed head-major layout"
                )

            for chunk_index, chunk_templates in enumerate(templates):
                state_start, state_end = self._state_range(
                    spec, self.store_chunk_size, chunk_index
                )
                for local_shard, (
                    addr_bases,
                    block_strides,
                    sizes,
                ) in enumerate(chunk_templates):
                    head_start = local_shard * self.heads_per_store_shard
                    for head_index in range(
                        head_start, head_start + self.heads_per_store_shard
                    ):
                        state_index = state_start
                        while state_index < state_end:
                            kernel_index, state_in_kernel = divmod(
                                state_index, num_states
                            )
                            count = min(
                                state_end - state_index,
                                num_states - state_in_kernel,
                            )
                            if kernel_index >= kernel_blocks:
                                raise ValueError(
                                    "Store chunk exceeds the local attention block"
                                )
                            addr_bases.append(
                                cache.data_ptr()
                                + kernel_index * kernel_stride
                                + head_index * head_stride
                                + state_in_kernel * state_stride
                            )
                            block_strides.append(block_stride)
                            sizes.append(count * content_bytes)
                            state_index += count

        self._set_segment_templates(templates)


class LBHNCStoreLayout(HeadMajorStoreLayout):
    pass


class BLHNCStoreLayout(HeadMajorStoreLayout):
    pass


class LHBNCStoreLayout(HeadMajorStoreLayout):
    pass


class BHLNCStoreLayout(HeadMajorStoreLayout):
    pass


class TokenMajorStoreLayout(AttentionStoreLayout):
    """Store descriptors with layer/state/head value ordering."""

    store_format = "attention_token_major"

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
        num_blocks: int,
    ) -> None:
        templates: list[list[tuple[list[int], list[int], list[int]]]] = [
            [([], [], []) for _ in range(self.shards_per_rank)]
            for _ in range(self.chunks_per_block)
        ]
        for cache, spec in zip(kv_caches, self.layer_specs, strict=True):
            (
                block_stride,
                kernel_stride,
                head_stride,
                state_stride,
                kernel_blocks,
                num_states,
                content_bytes,
            ) = self._cache_strides(cache, spec, num_blocks)
            if not (
                head_stride == content_bytes
                and state_stride == spec.num_heads * content_bytes
            ):
                raise ValueError(
                    "TP-shared Mooncake store requires a packed token-major layout"
                )

            for chunk_index, chunk_templates in enumerate(templates):
                state_start, state_end = self._state_range(
                    spec, self.store_chunk_size, chunk_index
                )
                for local_shard, (
                    addr_bases,
                    block_strides,
                    sizes,
                ) in enumerate(chunk_templates):
                    head_start = local_shard * self.heads_per_store_shard
                    for state_index in range(state_start, state_end):
                        kernel_index, state_in_kernel = divmod(state_index, num_states)
                        if kernel_index >= kernel_blocks:
                            raise ValueError(
                                "Store chunk exceeds the local attention block"
                            )
                        addr_bases.append(
                            cache.data_ptr()
                            + kernel_index * kernel_stride
                            + state_in_kernel * state_stride
                            + head_start * head_stride
                        )
                        block_strides.append(block_stride)
                        sizes.append(self.heads_per_store_shard * content_bytes)

        self._set_segment_templates(templates)


class LBNHCStoreLayout(TokenMajorStoreLayout):
    pass


class BLNHCStoreLayout(TokenMajorStoreLayout):
    pass


class MambaStoreLayout(TPShardedStoreLayout):
    """Store descriptors for cache groups represented by ``MambaSpec``.

    This includes Mamba and GDN convolution/recurrent state caches.
    """

    store_format = "mamba_state"

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int,
        local_tp_size: int,
        store_tp_size: int,
        tp_rank: int,
        layer_specs: Sequence[MambaSpec],
    ) -> None:
        super().__init__(
            metadata,
            block_size,
            hash_block_size,
            local_tp_size,
            store_tp_size,
            tp_rank,
        )
        self.layer_specs = tuple(layer_specs)
        self._segments = tuple(
            self._state_segments(spec, local_tp_size, self.shards_per_rank)
            for spec in self.layer_specs
        )

    @staticmethod
    def _state_segments(
        spec: MambaSpec, local_tp_size: int, shards_per_rank: int
    ) -> tuple[tuple[int, int, int], ...]:
        conv = derive_mamba_conv_split(spec, local_tp_size, require_dim_first=False)
        segments = []
        if is_conv_state_dim_first():
            row_bytes = conv.conv_rows * conv.conv_dtype_size
            for proj_index, (offset, size) in enumerate(conv.local_conv_offsets):
                unit_size = row_bytes
                if (
                    spec.mamba_type == MambaAttentionBackendEnum.MAMBA2
                    and proj_index > 0
                ):
                    unit_size *= spec.shapes[1][-1]
                segments.append((offset, size, unit_size))
        else:
            proj_offsets = []
            offset = 0
            for proj_dim in conv.local_proj_dims:
                proj_offsets.append(offset)
                offset += proj_dim
            for row_index in range(conv.conv_rows):
                row_offset = row_index * conv.local_conv_dim
                for proj_index, (proj_offset, proj_dim) in enumerate(
                    zip(proj_offsets, conv.local_proj_dims, strict=True)
                ):
                    unit_size = conv.conv_dtype_size
                    if (
                        spec.mamba_type == MambaAttentionBackendEnum.MAMBA2
                        and proj_index > 0
                    ):
                        unit_size *= spec.shapes[1][-1]
                    segments.append(
                        (
                            (row_offset + proj_offset) * conv.conv_dtype_size,
                            proj_dim * conv.conv_dtype_size,
                            unit_size,
                        )
                    )
        state_offset = conv.ssm_sizes[0]
        for state_index, shape in enumerate(spec.shapes[1:], start=1):
            state_dtype: torch.dtype = spec.dtypes[state_index]
            element_size = torch.empty((), dtype=state_dtype).element_size()
            state_bytes = torch.Size(shape).numel() * element_size
            unit_size = torch.Size(shape[1:]).numel() * element_size
            segments.append((state_offset, state_bytes, unit_size))
            state_offset += state_bytes
        for segment in segments:
            MambaStoreLayout._shard_slice(segment, 0, shards_per_rank)
        return tuple(segments)

    @staticmethod
    def _shard_slice(
        segment: tuple[int, int, int],
        shard_index: int,
        shards_per_rank: int,
    ) -> tuple[int, int]:
        offset, size, unit_size = segment
        if size % unit_size:
            raise ValueError("Mamba state segment contains a partial state unit")
        unit_count = size // unit_size
        if unit_count % shards_per_rank == 0:
            shard_size = size // shards_per_rank
            return offset + shard_index * shard_size, shard_size
        if shards_per_rank % unit_count == 0:
            # A local state unit can serve several finer Store shards.
            copies_per_unit = shards_per_rank // unit_count
            return offset + shard_index // copies_per_unit * unit_size, unit_size
        raise ValueError("Mamba state units cannot be mapped evenly to Store shards")

    @classmethod
    def schema_fingerprint(
        cls,
        layer_specs: Sequence[MambaSpec],
        local_tp_size: int,
        store_tp_size: int,
    ) -> str:
        shards_per_rank = store_tp_size // local_tp_size
        layers = tuple(
            (
                type(spec).__name__,
                tuple(
                    cls._shard_slice(segment, 0, shards_per_rank)[1]
                    for segment in cls._state_segments(
                        spec, local_tp_size, shards_per_rank
                    )
                ),
                tuple(str(dtype) for dtype in spec.dtypes),
                str(spec.mamba_type),
                "DS" if is_conv_state_dim_first() else "SD",
            )
            for spec in layer_specs
        )
        return _schema_fingerprint(layers)

    def register_kv_caches(
        self,
        kv_caches: Sequence[torch.Tensor],
        num_blocks: int,
    ) -> None:
        templates: list[tuple[list[int], list[int], list[int]]] = [
            ([], [], []) for _ in range(self.shards_per_rank)
        ]
        for cache, spec, segments in zip(
            kv_caches, self.layer_specs, self._segments, strict=True
        ):
            if cache.ndim != 4 or tuple(cache.shape[:3]) != (num_blocks, 1, 1):
                raise ValueError(
                    "TP-shared MambaSpec store requires logical cache shape "
                    "(num_blocks, 1, 1, state_bytes)"
                )
            element_size = cache.element_size()
            if cache.stride(3) * element_size != 1:
                raise ValueError(
                    "TP-shared MambaSpec store requires packed state bytes"
                )
            block_stride = cache.stride(0) * element_size
            content_bytes = cache.shape[3] * element_size
            if content_bytes < spec.state_content_size_bytes:
                raise ValueError("MambaSpec cache does not contain the complete state")
            for shard_index, (addr_bases, block_strides, sizes) in enumerate(templates):
                for segment in segments:
                    offset, shard_size = self._shard_slice(
                        segment, shard_index, self.shards_per_rank
                    )
                    addr_bases.append(cache.data_ptr() + offset)
                    block_strides.append(block_stride)
                    sizes.append(shard_size)
        self._set_segment_templates([templates])


class ChunkedTokenDatabase:
    """Enumerates logical token chunks and their hashes."""

    def __init__(
        self,
        metadata: KeyMetadata,
        block_size: int,
        hash_block_size: int | None = None,
        store_layout: StoreLayout | None = None,
    ):
        self.metadata = metadata
        self.block_size = block_size
        self.chunk_size = block_size
        self.hash_block_size = hash_block_size or block_size
        if self.chunk_size % self.hash_block_size != 0:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be a multiple of "
                f"hash_block_size ({self.hash_block_size})"
            )
        self.store_layout = store_layout or RankLocalStoreLayout(
            metadata, block_size, self.hash_block_size
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
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, token_len)
            if chunk_id % put_step != put_step_rank:
                continue
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
        event_block_size: int | None = None,
    ) -> "ReqMeta | None":
        """Create ReqMeta from a RequestTracker."""
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        scheduled_token_len = tracker.num_saved_tokens
        token_ids_start = scheduled_token_len
        if event_block_size is not None:
            token_ids_start = scheduled_token_len // event_block_size * event_block_size
        chunk_boundary = cdiv(scheduled_token_len + 1, block_size) * block_size
        num_tokens_to_save = input_token_len // block_size * block_size

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        # A ReqMeta must never carry both a save AND a load.
        # The save would also be wasted work — the bytes are being looked up
        # in the store right now. Later cached_reqs steps save new tokens
        # normally.
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
