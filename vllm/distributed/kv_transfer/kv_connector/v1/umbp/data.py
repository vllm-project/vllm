# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared data types for the UMBP KV connector.

The types in this module deliberately do not depend on a UMBP client.  They
describe vLLM's logical KV objects and the physical ranges a runtime adapter
will transfer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from math import gcd
from typing import Any, Literal, Sequence

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class RankTopology:
    """Model-parallel ranks that participate in one KV object."""

    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    pcp_rank: int = 0
    pcp_size: int = 1
    dcp_rank: int = 0
    dcp_size: int = 1

    def __post_init__(self) -> None:
        for name in ("tp_size", "pp_size", "pcp_size", "dcp_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for rank, size, name in (
            (self.tp_rank, self.tp_size, "tp_rank"),
            (self.pp_rank, self.pp_size, "pp_rank"),
            (self.pcp_rank, self.pcp_size, "pcp_rank"),
            (self.dcp_rank, self.dcp_size, "dcp_rank"),
        ):
            if rank < 0 or rank >= size:
                raise ValueError(f"{name}={rank} is outside [0, {size})")

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "RankTopology":
        parallel = vllm_config.parallel_config
        tp_size = getattr(parallel, "tensor_parallel_size", 1)
        pp_size = getattr(parallel, "pipeline_parallel_size", 1)
        rank = getattr(parallel, "rank", 0)
        return cls(
            tp_rank=getattr(
                parallel, "tensor_parallel_rank", rank % tp_size
            ),
            tp_size=tp_size,
            pp_rank=getattr(
                parallel,
                "pipeline_parallel_rank",
                (rank // tp_size) % pp_size,
            ),
            pp_size=pp_size,
            pcp_rank=getattr(parallel, "prefill_context_parallel_rank", 0),
            pcp_size=getattr(parallel, "prefill_context_parallel_size", 1),
            dcp_rank=getattr(parallel, "decode_context_parallel_rank", 0),
            dcp_size=getattr(parallel, "decode_context_parallel_size", 1),
        )

    @property
    def local_namespace(self) -> tuple[int, int, int, int]:
        return (self.tp_rank, self.pcp_rank, self.dcp_rank, self.pp_rank)

    @property
    def rank_count(self) -> int:
        return self.tp_size * self.pp_size * self.pcp_size * self.dcp_size

    def all_namespaces(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (tp, pcp, dcp, pp)
            for pp in range(self.pp_size)
            for dcp in range(self.dcp_size)
            for pcp in range(self.pcp_size)
            for tp in range(self.tp_size)
        )


@dataclass(frozen=True)
class RankCompletenessPolicy:
    """Required rank-local objects for one logical KV block."""

    topology: RankTopology

    @property
    def required_namespaces(self) -> tuple[tuple[int, int, int, int], ...]:
        return self.topology.all_namespaces()

    @property
    def required_rank_count(self) -> int:
        return len(self.required_namespaces)


@dataclass(frozen=True)
class UMBPNamespace:
    """Compatibility namespace for objects stored outside the engine."""

    value: str

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: Any,
        kv_cache_config: KVCacheConfig,
    ) -> "UMBPNamespace":
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        configured = extra.get("key_namespace", "auto")
        if configured != "auto":
            if not isinstance(configured, str) or not configured:
                raise ValueError("key_namespace must be a non-empty string or 'auto'")
            return cls(configured)

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        fields = {
            "model": getattr(model_config, "model", ""),
            "revision": getattr(model_config, "revision", None),
            "block_size": vllm_config.cache_config.block_size,
            "cache_layout": getattr(kv_cache_config, "kv_cache_layout", None),
            "groups": [
                {
                    "layers": group.layer_names,
                    "block_size": group.kv_cache_spec.block_size,
                    "kind": type(group.kv_cache_spec).__name__,
                    "dtype": str(getattr(group.kv_cache_spec, "dtype", None)),
                }
                for group in kv_cache_config.kv_cache_groups
            ],
            "tp": parallel_config.tensor_parallel_size,
            "pp": parallel_config.pipeline_parallel_size,
            "pcp": getattr(parallel_config, "prefill_context_parallel_size", 1),
            "dcp": getattr(parallel_config, "decode_context_parallel_size", 1),
            "layout_version": 1,
        }
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        return cls(hashlib.sha256(payload).hexdigest()[:32])


@dataclass(frozen=True)
class BlockIdentityCodec:
    """Build stable rank-local keys from vLLM block hashes."""

    namespace: UMBPNamespace
    tp_rank: int = 0
    pp_rank: int = 0
    pcp_rank: int = 0
    dcp_rank: int = 0

    def key(self, block_hash: bytes, group_id: int = 0) -> str:
        return self.key_for_namespace(
            block_hash,
            group_id,
            (self.tp_rank, self.pcp_rank, self.dcp_rank, self.pp_rank),
        )

    def key_for_namespace(
        self,
        block_hash: bytes,
        group_id: int,
        namespace: tuple[int, int, int, int],
    ) -> str:
        if not isinstance(block_hash, bytes) or not block_hash:
            raise ValueError("block_hash must be non-empty bytes")
        if group_id < 0:
            raise ValueError("group_id must be non-negative")
        tp_rank, pcp_rank, dcp_rank, pp_rank = namespace
        return (
            f"umbp:vllm:v1:{self.namespace.value}:"
            f"tp{tp_rank}:pcp{pcp_rank}:dcp{dcp_rank}:"
            f"pp{pp_rank}:g{group_id}:"
            f"{block_hash.hex()}"
        )

    def keys_for_block(
        self, block_hash: bytes, group_ids: Sequence[int]
    ) -> tuple[str, ...]:
        return tuple(self.key(block_hash, group_id) for group_id in group_ids)

    def keys_for_topology(
        self,
        block_hash: bytes,
        topology: RankTopology,
        group_ids: Sequence[int],
    ) -> tuple[str, ...]:
        return tuple(
            self.key_for_namespace(block_hash, group_id, namespace)
            for namespace in topology.all_namespaces()
            for group_id in group_ids
        )


@dataclass(frozen=True)
class KVRegion:
    """One deterministic region in a rank-local KV object."""

    layer_name: str
    group_id: int
    block_stride: int
    block_bytes: int
    object_offset: int

    def __post_init__(self) -> None:
        if self.block_stride <= 0 or self.block_bytes <= 0:
            raise ValueError("KV region sizes must be positive")
        if self.block_bytes > self.block_stride:
            raise ValueError(
                f"KV region bytes ({self.block_bytes}) exceed its stride "
                f"({self.block_stride}) for {self.layer_name!r}"
            )


@dataclass(frozen=True)
class KVShardSlice:
    """One contiguous KV-head slice copied between TP ranks."""

    producer_rank: int
    source_head: int
    destination_head: int
    num_heads: int


@dataclass(frozen=True)
class TPShardMapping:
    """TP shard mapping for one consumer rank."""

    producer_tp_size: int
    consumer_tp_size: int
    consumer_rank: int
    num_kv_heads: int
    slices: tuple[KVShardSlice, ...]
    replicated: bool = False

    @classmethod
    def build(
        cls,
        producer_tp_size: int,
        consumer_tp_size: int,
        consumer_rank: int,
        num_kv_heads: int,
    ) -> "TPShardMapping":
        if producer_tp_size <= 0 or consumer_tp_size <= 0:
            raise ValueError("TP sizes must be positive")
        if consumer_rank < 0 or consumer_rank >= consumer_tp_size:
            raise ValueError("consumer_rank is outside the consumer TP group")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

        common = producer_tp_size * consumer_tp_size // gcd(
            producer_tp_size, consumer_tp_size
        )
        if num_kv_heads < common:
            # MQA/GQA ranks can replicate a smaller KV-head set.
            return cls(
                producer_tp_size,
                consumer_tp_size,
                consumer_rank,
                num_kv_heads,
                (
                    KVShardSlice(
                        producer_rank=0,
                        source_head=0,
                        destination_head=0,
                        num_heads=num_kv_heads,
                    ),
                ),
                replicated=True,
            )
        if num_kv_heads % common:
            raise ValueError(
                "num_kv_heads must be divisible by lcm(producer_tp_size, "
                "consumer_tp_size)"
            )

        consumer_heads = num_kv_heads // consumer_tp_size
        consumer_start = consumer_rank * consumer_heads
        slices: list[KVShardSlice] = []
        for producer_rank in range(producer_tp_size):
            producer_heads = num_kv_heads // producer_tp_size
            producer_start = producer_rank * producer_heads
            overlap_start = max(consumer_start, producer_start)
            overlap_end = min(
                consumer_start + consumer_heads,
                producer_start + producer_heads,
            )
            if overlap_start < overlap_end:
                slices.append(
                    KVShardSlice(
                        producer_rank=producer_rank,
                        source_head=overlap_start - producer_start,
                        destination_head=overlap_start - consumer_start,
                        num_heads=overlap_end - overlap_start,
                    )
                )
        return cls(
            producer_tp_size,
            consumer_tp_size,
            consumer_rank,
            num_kv_heads,
            tuple(slices),
        )

    def validate(self) -> None:
        if self.replicated:
            return
        expected = self.num_kv_heads // self.consumer_tp_size
        covered = sum(item.num_heads for item in self.slices)
        if covered != expected:
            raise ValueError(
                f"TP mapping covers {covered} heads, expected {expected}"
            )


@dataclass(frozen=True)
class KVLayoutDescriptor:
    """Backend-neutral description of one rank-local KV object layout."""

    regions: tuple[KVRegion, ...]
    topology: RankTopology
    layout_format: str = "rank_local"
    tp_shard_mapping: TPShardMapping | None = None

    def __post_init__(self) -> None:
        if not self.layout_format:
            raise ValueError("layout_format must not be empty")
        names = [region.layer_name for region in self.regions]
        if len(names) != len(set(names)):
            raise ValueError("KV layout contains duplicate layer regions")

    @property
    def object_size(self) -> int:
        return sum(region.block_bytes for region in self.regions)


class KVLayoutPlanner:
    """Create a deterministic all-layer object layout from KVCacheConfig."""

    def __init__(self, regions: Sequence[KVRegion]) -> None:
        self.regions = tuple(regions)
        self._base_addresses: dict[str, int] = {}
        self._physical_strides: dict[str, int] = {}

    @classmethod
    def from_kv_cache_config(cls, config: KVCacheConfig) -> "KVLayoutPlanner":
        group_by_layer = {
            layer: group_id
            for group_id, group in enumerate(config.kv_cache_groups)
            for layer in group.layer_names
            if group.enable_kv_transfer
        }
        regions: list[KVRegion] = []
        offset = 0
        for tensor in config.kv_cache_tensors:
            for layer_name in tensor.layers:
                group_id = group_by_layer.get(layer_name)
                if group_id is None:
                    continue
                spec = config.kv_cache_groups[group_id].kv_cache_spec
                block_bytes = spec.page_size_bytes
                regions.append(
                    KVRegion(
                        layer_name=layer_name,
                        group_id=group_id,
                        block_stride=tensor.block_stride,
                        block_bytes=block_bytes,
                        object_offset=offset,
                    )
                )
                offset += block_bytes
        return cls(regions)

    @property
    def object_size(self) -> int:
        return sum(region.block_bytes for region in self.regions)

    def describe(
        self,
        topology: RankTopology,
        layout_format: str = "rank_local",
        tp_shard_mapping: TPShardMapping | None = None,
    ) -> KVLayoutDescriptor:
        return KVLayoutDescriptor(
            regions=self.regions,
            topology=topology,
            layout_format=layout_format,
            tp_shard_mapping=tp_shard_mapping,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Capture stable base addresses for the planned KV regions."""
        required = {region.layer_name for region in self.regions}
        missing = required - kv_caches.keys()
        if missing:
            raise ValueError(f"missing KV cache layers: {sorted(missing)}")
        addresses: dict[str, int] = {}
        for layer_name in required:
            cache = kv_caches[layer_name]
            if cache.ndim == 0:
                raise ValueError(f"KV cache {layer_name!r} must be a tensor")
            block_stride = cache.stride(0) * cache.element_size()
            addresses[layer_name] = cache.data_ptr()
            for region in self.regions:
                if region.layer_name != layer_name:
                    continue
                if (
                    block_stride > region.block_stride
                    or region.block_stride % block_stride != 0
                    or (
                        region.block_bytes > block_stride
                        and region.block_bytes % block_stride != 0
                    )
                ):
                    raise ValueError(
                        f"KV block stride mismatch for {layer_name!r}: "
                        f"config={region.block_stride}, tensor={block_stride}"
                    )
                self._physical_strides[layer_name] = block_stride
        self._base_addresses = addresses

    def plan_registered_block(
        self,
        key: str,
        block_id: int,
        *,
        request_id: str | None = None,
        generation: int = 0,
    ) -> "BlockTransferPlan":
        if not self._base_addresses:
            raise RuntimeError("KV caches must be registered before planning")
        return self.plan_for_block(
            key,
            block_id,
            self._base_addresses,
            request_id=request_id,
            generation=generation,
        )

    def plan_for_block(
        self,
        key: str,
        block_id: int,
        base_addresses: dict[str, int],
        *,
        request_id: str | None = None,
        generation: int = 0,
    ) -> "BlockTransferPlan":
        """Lower one GPU block into a deterministic scatter/gather plan."""
        if block_id < 0:
            raise ValueError("block_id must be non-negative")
        ranges: list[KVRange] = []
        for region in self.regions:
            try:
                base_address = base_addresses[region.layer_name]
            except KeyError as exc:
                raise ValueError(
                    f"missing base address for layer {region.layer_name!r}"
                ) from exc
            physical_stride = self._physical_strides.get(
                region.layer_name, region.block_stride
            )
            physical_parts = max(1, region.block_bytes // physical_stride)
            for part in range(physical_parts):
                physical_block_id = block_id * physical_parts + part
                part_length = (
                    region.block_bytes
                    if physical_parts == 1
                    else physical_stride
                )
                ranges.append(
                    KVRange(
                        layer_name=region.layer_name,
                        group_id=region.group_id,
                        block_id=block_id,
                        base_address=(
                            base_address
                            + physical_block_id * physical_stride
                        ),
                        stride=physical_stride,
                        length=part_length,
                        object_offset=(
                            region.object_offset + part * physical_stride
                        ),
                    )
                )
        return BlockTransferPlan(
            key=key,
            block_id=block_id,
            ranges=tuple(ranges),
            request_id=request_id,
            generation=generation,
        )


@dataclass(frozen=True)
class KVRange:
    """A source/destination range in a runtime transfer."""

    layer_name: str
    group_id: int
    block_id: int
    base_address: int
    stride: int
    length: int
    object_offset: int


@dataclass(frozen=True)
class BlockTransferPlan:
    """One complete logical block transfer."""

    key: str
    block_id: int
    ranges: tuple[KVRange, ...] = ()
    request_id: str | None = None
    generation: int = 0
    group_id: int | None = None
    block_hash: bytes | None = None
    parent_block_hash: bytes | None = None
    token_ids: tuple[int, ...] = ()
    block_size: int = 0
    medium: str = "CPU"


@dataclass
class LoadSpec:
    """Scheduler decision for one externally loaded prefix."""

    local_tokens: int
    external_tokens: int
    can_load: bool = False

    @property
    def num_tokens_to_load(self) -> int:
        return max(self.external_tokens - self.local_tokens, 0)


class LookupStatus(str, Enum):
    PENDING = "pending"
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class LookupState:
    """Request-scoped lookup state owned by the scheduler."""

    request_id: str
    status: LookupStatus = LookupStatus.PENDING
    matched_tokens: int = 0
    error: str | None = None

    def complete(self, matched_tokens: int) -> None:
        self.status = LookupStatus.HIT if matched_tokens else LookupStatus.MISS
        self.matched_tokens = matched_tokens
        self.error = None

    def fail(self, error: str) -> None:
        self.status = LookupStatus.ERROR
        self.matched_tokens = 0
        self.error = error

    def cancel(self) -> None:
        self.status = LookupStatus.CANCELLED
        self.matched_tokens = 0


@dataclass(frozen=True)
class PartialTailPlan:
    """A non-block-aligned tail that must be pinned and stored separately."""

    request_id: str
    generation: int
    block_id: int
    group_id: int
    start_token: int
    end_token: int
    key: str


@dataclass
class RequestTracker:
    """Core request state shared by every runtime mode."""

    request_id: str
    generation: int
    token_len: int = 0
    saved_tokens: int = 0
    block_ids: tuple[list[int], ...] = ()
    load_spec: LoadSpec | None = None
    save_mode: Literal["eager", "lazy"] = "eager"
    prefill_end_tokens: int = 0
    retry_from_tokens: int | None = None
    pending_tail: PartialTailPlan | None = None

    def reset(self) -> None:
        self.token_len = 0
        self.saved_tokens = 0
        self.block_ids = ()
        self.load_spec = None
        self.prefill_end_tokens = 0
        self.retry_from_tokens = None
        self.pending_tail = None
        self.generation += 1

    def update_blocks(self, block_ids: tuple[list[int], ...]) -> None:
        if not self.block_ids:
            self.block_ids = tuple(list(group) for group in block_ids)
            return
        if len(self.block_ids) != len(block_ids):
            raise ValueError("KV cache group count changed for a request")
        for current, new in zip(self.block_ids, block_ids, strict=True):
            if new[: len(current)] == current:
                current.extend(new[len(current) :])
            elif not new:
                continue
            elif current[-len(new) :] == new:
                continue
            else:
                current.extend(new)

    def mark_saved(self, token_count: int, block_size: int) -> int:
        """Advance the save watermark only to a complete block boundary."""
        complete_tokens = token_count // block_size * block_size
        if complete_tokens < self.saved_tokens:
            return self.saved_tokens
        self.saved_tokens = complete_tokens
        return complete_tokens

    def record_store_failure(self, start_tokens: int) -> None:
        """Retry from the earliest suffix not durably stored."""
        if self.retry_from_tokens is None:
            self.retry_from_tokens = start_tokens
        else:
            self.retry_from_tokens = min(self.retry_from_tokens, start_tokens)

    def clear_store_retry(self) -> None:
        self.retry_from_tokens = None


class TransferJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TransferJobState:
    """Per-key completion state; one failed key does not fail its siblings."""

    plans: tuple[BlockTransferPlan, ...]
    status: TransferJobStatus = TransferJobStatus.PENDING
    completed_keys: set[str] = field(default_factory=set)
    failed_keys: set[str] = field(default_factory=set)
    error: str | None = None

    def start(self) -> None:
        if self.status != TransferJobStatus.PENDING:
            raise RuntimeError(f"cannot start job in state {self.status}")
        self.status = TransferJobStatus.RUNNING

    def complete(self, keys: Sequence[str] = ()) -> None:
        self.completed_keys.update(keys or (plan.key for plan in self.plans))
        self._finish_if_done()

    def fail(self, keys: Sequence[str], error: str | None = None) -> None:
        self.failed_keys.update(keys)
        self.error = error
        self._finish_if_done()

    def cancel(self, error: str = "cancelled") -> None:
        self.error = error
        self.status = TransferJobStatus.CANCELLED

    @property
    def failed_block_ids(self) -> set[int]:
        return {
            plan.block_id for plan in self.plans if plan.key in self.failed_keys
        }

    def _finish_if_done(self) -> None:
        keys = {plan.key for plan in self.plans}
        if self.completed_keys | self.failed_keys >= keys:
            self.status = (
                TransferJobStatus.FAILED
                if self.failed_keys
                else TransferJobStatus.COMPLETED
            )


@dataclass
class UMBPConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for one engine step."""

    load_plans: list[BlockTransferPlan] = field(default_factory=list)
    store_plans: list[BlockTransferPlan] = field(default_factory=list)
    load_requests: dict[str, list[BlockTransferPlan]] = field(default_factory=dict)
    store_requests: dict[str, list[BlockTransferPlan]] = field(default_factory=dict)
    load_plans_by_layer: dict[str, list[BlockTransferPlan]] = field(
        default_factory=dict
    )
    store_plans_by_layer: dict[str, list[BlockTransferPlan]] = field(
        default_factory=dict
    )
    partial_tail_plans: list[PartialTailPlan] = field(default_factory=list)
    lookup_states: dict[str, LookupState] = field(default_factory=dict)
    preempted_block_ids: set[int] = field(default_factory=set)
    preempted_request_ids: set[str] = field(default_factory=set)


@dataclass
class UMBPConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker completion metadata aggregated across ranks."""

    completed_loads: set[str] = field(default_factory=set)
    completed_stores: set[str] = field(default_factory=set)
    kv_events: list[Any] = field(default_factory=list)
    failed_loads: dict[str, str] = field(default_factory=dict)
    failed_store_errors: dict[str, str] = field(default_factory=dict)
    failed_stores: set[str] = field(default_factory=set)
    completed_store_counts: dict[str, int] = field(default_factory=dict)
    failed_store_counts: dict[str, int] = field(default_factory=dict)
    completed_store_tokens: dict[tuple[str, int], int] = field(default_factory=dict)
    failed_store_tokens: dict[tuple[str, int], int] = field(default_factory=dict)
    failed_block_ids: set[int] = field(default_factory=set)

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> "UMBPConnectorWorkerMetadata":
        if not isinstance(other, UMBPConnectorWorkerMetadata):
            raise TypeError("cannot aggregate incompatible UMBP worker metadata")
        self.completed_loads.update(other.completed_loads)
        self.completed_stores.update(other.completed_stores)
        self.kv_events.extend(other.kv_events)
        self.failed_loads.update(other.failed_loads)
        self.failed_store_errors.update(other.failed_store_errors)
        self.failed_stores.update(other.failed_stores)
        for key, count in other.completed_store_counts.items():
            self.completed_store_counts[key] = (
                self.completed_store_counts.get(key, 0) + count
            )
        for key, count in other.failed_store_counts.items():
            self.failed_store_counts[key] = (
                self.failed_store_counts.get(key, 0) + count
            )
        for token, count in other.completed_store_tokens.items():
            self.completed_store_tokens[token] = (
                self.completed_store_tokens.get(token, 0) + count
            )
        for token, count in other.failed_store_tokens.items():
            self.failed_store_tokens[token] = (
                self.failed_store_tokens.get(token, 0) + count
            )
        self.failed_block_ids.update(other.failed_block_ids)
        return self
