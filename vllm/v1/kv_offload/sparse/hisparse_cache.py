# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA/AMD HiSparse resident, host, and hot-cache data plane."""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from math import gcd

import psutil
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

logger = init_logger(__name__)

# fp8_ds_mla KV row: 512 B quantized NoPE + 16 B scales + 128 B RoPE.
FP8_DS_MLA_ROW_BYTES = 656
HiSparseTopKResult = (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


@dataclass(frozen=True)
class PagedCacheView:
    cache: torch.Tensor
    attention_cache: torch.Tensor
    block_size: int
    attention_block_stride: int

    @classmethod
    def bind(
        cls,
        raw_tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        row_width: int,
        byte_offset: int,
        block_stride: int,
        num_blocks: int,
        block_size: int,
    ) -> PagedCacheView:
        itemsize = dtype.itemsize
        assert byte_offset % itemsize == 0 and block_stride % itemsize == 0
        cache = torch.as_strided(
            raw_tensor.view(dtype),
            size=(num_blocks, block_size, row_width),
            stride=(block_stride // itemsize, row_width, 1),
            storage_offset=byte_offset // itemsize,
        )
        row_bytes = row_width * itemsize
        if block_stride % (block_size * row_bytes):
            return cls(cache, cache, block_size, block_size)
        attention_cache = (
            raw_tensor[byte_offset:].view(dtype).view(-1, block_size, row_width)
        )
        return cls(cache, attention_cache, block_size, block_stride // row_bytes)


@triton.jit
def _compress_hisparse_slot_mapping_kernel(
    source_ptr,
    positions_ptr,
    output_ptr,
    num_tokens,
    logical_block_size: tl.constexpr,
    storage_block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_tokens
    source = tl.load(source_ptr + offset, mask=mask, other=-1)
    position = tl.load(positions_ptr + offset, mask=mask, other=-1)
    valid = (source >= 0) & ((position + 1) % compress_ratio == 0)
    compressed = (
        source // logical_block_size * storage_block_size
        + source % logical_block_size // compress_ratio
    )
    tl.store(output_ptr + offset, tl.where(valid, compressed, -1), mask=mask)


def compress_hisparse_slot_mapping(
    source: torch.Tensor,
    positions: torch.Tensor,
    *,
    logical_block_size: int,
    storage_block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map uncompressed HMA slots to a compressed physical cache."""
    num_tokens = min(source.numel(), positions.numel())
    if out is None:
        out = torch.empty_like(source)
    out.fill_(-1)
    if num_tokens:
        block = 256
        _compress_hisparse_slot_mapping_kernel[(triton.cdiv(num_tokens, block),)](
            source,
            positions,
            out,
            num_tokens,
            logical_block_size=logical_block_size,
            storage_block_size=storage_block_size,
            compress_ratio=compress_ratio,
            BLOCK_SIZE=block,
        )
    return out[:num_tokens]


@dataclass(frozen=True)
class ResolvedHiSparseConfig:
    top_k: int
    device_buffer_size: int
    host_pool_gib: float

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        model_top_k: int,
        block_size: int | None = None,
    ) -> ResolvedHiSparseConfig | None:
        config = vllm_config.attention_config.hisparse_config
        if config is None:
            return None

        # Default 2x top_k: at exactly top_k the LRU has zero slack and
        # boundary entries thrash between steps.
        configured_size = config.device_buffer_size
        if configured_size is None:
            device_buffer_size = 2 * model_top_k
        else:
            device_buffer_size = configured_size

        if device_buffer_size < model_top_k:
            raise ValueError(
                "HiSparse device_buffer_size must cover at least the model's "
                "index_topk rows. Got "
                f"device_buffer_size={device_buffer_size}, "
                f"index_topk={model_top_k}; expected at least {model_top_k}."
            )
        max_device_buffer_size = torch.iinfo(torch.int16).max + 1
        if device_buffer_size > max_device_buffer_size:
            raise ValueError(
                "HiSparse device_buffer_size exceeds the int16 slot-index "
                f"limit: got {device_buffer_size}, maximum is "
                f"{max_device_buffer_size}."
            )
        if configured_size is not None and block_size is not None:
            padding = -device_buffer_size % block_size
            if padding:
                logger.warning(
                    "HiSparse device_buffer_size=%d is not aligned to the "
                    "%d-token kernel block size and allocates %d unused "
                    "padding rows per hot group. Use %d to use all allocated "
                    "rows.",
                    device_buffer_size,
                    block_size,
                    padding,
                    device_buffer_size + padding,
                )
        return cls(
            top_k=model_top_k,
            device_buffer_size=device_buffer_size,
            host_pool_gib=config.host_pool_gib,
        )


def check_hisparse_host_memory(rank_bytes: int) -> None:
    """Fail fast when this rank's pinned host pool cannot fit in RAM."""
    mem = psutil.virtual_memory()
    if rank_bytes > mem.available * 0.95:
        raise ValueError(
            f"HiSparse pinned host pool needs ~{rank_bytes / 2**30:.0f} GiB "
            f"but only {mem.available / 2**30:.0f} GiB of RAM is available. "
            "Lower hisparse_config.host_pool_gib or leave headroom for co-tenants."
        )


def allocate_pinned_host_pool(size: int) -> torch.Tensor:
    """Allocate and deterministically register an exact-size host KV region."""
    page = 4096
    padded_size = round_up(size, page)
    backing = torch.empty(padded_size + page, dtype=torch.int8, device="cpu")
    aligned_offset = (-backing.data_ptr()) % page
    registered = backing[aligned_offset : aligned_offset + padded_size]
    pin_tensor(registered)
    _PINNED_HOST_POOLS.append(registered)
    return registered[:size]


def register_indexer_source(
    layer_name: str, cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    _INDEXER_SOURCES[layer_name] = (cache, slot_mapping)


def get_indexer_source(
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    return _INDEXER_SOURCES.get(layer_name)


def _covers_registered_host_range(ptr: int, nbytes: int) -> bool:
    return any(
        pool.data_ptr() <= ptr and ptr + nbytes <= pool.data_ptr() + pool.nbytes
        for pool in _PINNED_HOST_POOLS
    )


def release_pinned_state(runtimes: list[HiSparseOffloadRuntime]) -> None:
    """Synchronize, unregister host KV pools, and drop global state."""
    global _CURRENT_INDEX_GROUP

    _CURRENT_INDEX_GROUP = None
    if _PINNED_HOST_POOLS:
        try:
            torch.accelerator.synchronize()
        except RuntimeError as e:
            logger.warning(
                "HiSparse: CUDA context unusable at teardown (%s); leaving "
                "%d host-pool tensors pinned for kernel exit reclaim.",
                e,
                len(_PINNED_HOST_POOLS),
            )
            return

        cudart = torch.cuda.cudart()
        release_start = time.perf_counter()
        freed_bytes = 0
        while _PINNED_HOST_POOLS:
            tensor = _PINNED_HOST_POOLS[-1]
            err = cudart.cudaHostUnregister(tensor.data_ptr())
            if err.value != 0:
                logger.warning(
                    "HiSparse: cudaHostUnregister failed (code=%d); leaving "
                    "%d host-pool tensors pinned for kernel exit reclaim.",
                    err.value,
                    len(_PINNED_HOST_POOLS),
                )
                cudart.cudaGetLastError()
                break
            freed_bytes += tensor.nbytes
            _PINNED_HOST_POOLS.pop()
        if freed_bytes:
            logger.info(
                "HiSparse: unpinned %.1f GiB of host pool in %.1fs.",
                freed_bytes / 2**30,
                time.perf_counter() - release_start,
            )

    for runtime in runtimes:
        runtime._host_cache = None
    _get_group_plan.cache_clear()
    _get_copy_stream.cache_clear()
    _HOST_WRITE_EVENTS.clear()
    with suppress(RuntimeError):
        torch._C._host_emptyCache()
    _INDEXER_SOURCES.clear()


def wait_for_hisparse_host_writes() -> None:
    """Wait for pending GPU writes before accessing host KV from the CPU."""
    for event in _HOST_WRITE_EVENTS.values():
        event.synchronize()


def register_host_write_event(device: torch.device, event: torch.Event) -> None:
    _HOST_WRITE_EVENTS[str(device)] = event


def hisparse_prefill_staging_remap(
    block_table: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Renumber a block table against its unique referenced blocks."""
    unique_ids, inverse = torch.unique(block_table.clamp(min=0), return_inverse=True)
    new_bt = inverse.to(torch.int32)
    row_ids = (
        unique_ids.to(torch.int32).unsqueeze(1) * block_size
        + torch.arange(block_size, dtype=torch.int32, device=block_table.device)
    ).view(1, -1)
    return new_bt, row_ids


@dataclass(frozen=True)
class HiSparsePrefillStagingPlan:
    block_table: torch.Tensor
    row_ids: torch.Tensor
    dst_rows: torch.Tensor
    miss_mask: torch.Tensor
    block_size: int


def build_hisparse_prefill_staging_plan(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> HiSparsePrefillStagingPlan:
    """Build the layer-independent remap for host-cache prefill staging."""
    device = block_table.device
    used = (seq_lens.to(torch.int64) + block_size - 1) // block_size
    bounded = torch.where(
        torch.arange(block_table.shape[1], device=device)[None, :] < used[:, None],
        block_table,
        0,
    )
    new_bt, row_ids = hisparse_prefill_staging_remap(bounded, block_size)
    dst_rows = torch.arange(row_ids.shape[1], dtype=torch.int32, device=device).view(
        1, -1
    )
    return HiSparsePrefillStagingPlan(
        block_table=new_bt,
        row_ids=row_ids,
        dst_rows=dst_rows,
        miss_mask=torch.ones_like(row_ids),
        block_size=block_size,
    )


def _has_hisparse_ops() -> bool:
    if not hasattr(torch.ops, "_C_cache_ops"):
        return False
    return (
        hasattr(torch.ops._C_cache_ops, "hisparse_swap_in")
        and hasattr(torch.ops._C_cache_ops, "hisparse_gather_plan")
        and hasattr(torch.ops._C_cache_ops, "hisparse_gather_compact")
        and hasattr(torch.ops._C_cache_ops, "hisparse_backup")
        and hasattr(torch.ops._C_cache_ops, "hisparse_backup_layers")
    )


class _GroupPlan:
    """CUDA-graph-safe swap plan replayed by index-sharing runtimes."""

    def __init__(self, device: torch.device, max_rows: int, top_k: int) -> None:
        self.hot_indices = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.attention_indices = torch.empty_like(self.hot_indices)
        self.miss_global_indices = torch.empty(
            (max_rows, top_k), dtype=torch.int32, device=device
        )
        self.miss_hot_indices = torch.empty_like(self.miss_global_indices)
        self.miss_counts = torch.empty(max_rows, dtype=torch.int32, device=device)
        self.valid_counts = torch.empty(max_rows, dtype=torch.int32, device=device)


_PINNED_HOST_POOLS: list[torch.Tensor] = []
_INDEXER_SOURCES: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_HOST_WRITE_EVENTS: dict[str, torch.Event] = {}
_CURRENT_INDEX_GROUP: tuple[object, HiSparseOffloadRuntime | None] | None = None


@cache
def _get_group_plan(device: torch.device, max_rows: int, top_k: int) -> _GroupPlan:
    return _GroupPlan(device, max_rows, top_k)


@cache
def _get_copy_stream(device: torch.device) -> torch.Stream:
    return torch.Stream(device=device)


class HiSparseOffloadRuntime:
    """Per-cache host/hot data plane and GPU replacement state."""

    def __init__(
        self,
        config: ResolvedHiSparseConfig,
        max_num_reqs: int,
        row_width: int,
        kv_dtype: torch.dtype,
        device: torch.device | str,
        storage_block_size: int | None = None,
        row_value_bytes: int | None = None,
    ) -> None:
        if not _has_hisparse_ops():
            raise RuntimeError(
                "HiSparse requires its compiled _C_cache_ops CUDA kernels "
                "(host-resident decode has no Python fallback). Rebuild vLLM "
                "from source so "
                "csrc/libtorch_stable/hisparse_kernels.cu is included."
            )
        self.max_num_reqs = max_num_reqs
        self.row_width = row_width
        self.kv_dtype = kv_dtype
        self.device = torch.device(device)
        self.storage_block_size = storage_block_size
        self.row_value_bytes = row_value_bytes
        # Logical slots per request. Physical rows come from its ephemeral HMA
        # block table and need not be contiguous in the shared slab.
        self.region_stride = config.device_buffer_size

        row_bytes = row_width * kv_dtype.itemsize
        if row_value_bytes is None and row_bytes % 16 != 0:
            raise ValueError(
                f"HiSparse requires 16-byte aligned KV rows, got {row_bytes}B."
            )
        if row_value_bytes is not None and not 0 < row_value_bytes < row_bytes:
            raise ValueError(
                "HiSparse split-page value bytes must be between zero and the "
                f"full row width, got {row_value_bytes} for a {row_bytes}B row."
            )

        self.hot: PagedCacheView | None = None
        self.hot_block_table: torch.Tensor | None = None
        # Per-request LRU state; released in join_group for index-sharing
        # followers, which replay their leader's plan and never resolve
        # the LRU themselves.
        self.device_global_indices: torch.Tensor | None = torch.full(
            (max_num_reqs, self.region_stride),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        lru_init = torch.arange(
            self.region_stride, dtype=torch.int16, device=self.device
        )
        self._lru_init: torch.Tensor | None = lru_init
        self.lru_slots: torch.Tensor | None = lru_init.repeat(
            max_num_reqs, 1
        ).contiguous()
        # In-kernel hit/miss counters (telemetry). stats_row_bytes converts
        # misses to gathered bytes; plan-once wiring adds each follower's
        # row bytes to its leader (the followers re-gather the
        # leader's misses), so the leader's counter covers the whole group.
        self._swap_stats = torch.zeros(2, dtype=torch.uint64, device=self.device)
        self.stats_row_bytes = row_bytes
        self._plan = _get_group_plan(self.device, max_num_reqs, config.top_k)
        self.followers: list[HiSparseOffloadRuntime] = []
        self.leader: HiSparseOffloadRuntime | None = None
        self._prefetch_event: torch.Event | None = None
        self._copy_stream = _get_copy_stream(self.device)

        self._host_cache: torch.Tensor | None = None
        self.eager_host_mirror = False
        self.resident_source_index = -1
        self.request_state_indices: torch.Tensor | None = None

    def join_group(self, leader: HiSparseOffloadRuntime) -> None:
        self.leader = leader
        self._plan = leader._plan
        leader.followers.append(self)
        leader.stats_row_bytes += self.stats_row_bytes
        self.device_global_indices = None
        self.lru_slots = None
        self._lru_init = None

    def join_index_group(self, scope: object, is_leader: bool) -> None:
        """Join the model-order index-sharing group during construction."""
        global _CURRENT_INDEX_GROUP

        leader = None
        if _CURRENT_INDEX_GROUP is not None:
            current_scope, leader = _CURRENT_INDEX_GROUP
            if current_scope is not scope:
                leader = None
        if is_leader:
            leader = self
        elif leader is not None:
            self.join_group(leader)
        _CURRENT_INDEX_GROUP = (scope, leader)

    def bind_hot_cache(
        self,
        raw_tensor: torch.Tensor,
        *,
        byte_offset: int,
        block_stride: int,
        num_blocks: int,
        block_size: int,
        block_table: torch.Tensor,
    ) -> None:
        """Bind this runtime's strided view into the shared GPU HMA slab."""
        storage_block_size = self.storage_block_size or block_size
        self.hot = PagedCacheView.bind(
            raw_tensor,
            dtype=self.kv_dtype,
            row_width=self.row_width,
            byte_offset=byte_offset,
            block_stride=block_stride,
            num_blocks=num_blocks,
            block_size=storage_block_size,
        )
        self.hot_block_table = block_table

    def bind_source_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.dtype != self.kv_dtype or kv_cache.shape[-1] != self.row_width:
            raise ValueError(
                "HiSparse offload runtime bound to a KV cache with mismatched "
                f"layout: expected ({self.row_width}, {self.kv_dtype}), got "
                f"({kv_cache.shape[-1]}, {kv_cache.dtype})."
            )
        # Host-resident pool: the cache itself is the only full-size store;
        # every allocated slot is written before the indexer can select it.
        if kv_cache.device.type != "cpu":
            raise ValueError(
                "HiSparse requires a host-resident KV pool; got a KV cache "
                f"on {kv_cache.device}."
            )
        # The pool is pinned via cudaHostRegister (exact-size, deterministic
        # unpin at shutdown); torch's is_pinned() only recognizes its own
        # caching-host-allocator memory, so also accept ranges the model
        # allocator explicitly registered.
        if not (
            kv_cache.is_pinned()
            or _covers_registered_host_range(kv_cache.data_ptr(), kv_cache.nbytes)
        ):
            raise ValueError("HiSparse host-resident KV pool must be pinned memory.")

        self._host_cache = (
            kv_cache
            if self.row_value_bytes is not None
            else kv_cache.view(-1, kv_cache.shape[-1])
        )

    def stage_prefill_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather referenced host context blocks into a compact GPU cache."""
        block_size = kv_cache.shape[1]
        plan = build_hisparse_prefill_staging_plan(
            block_table,
            seq_lens,
            block_size,
        )
        return self.gather_prefill_cache(kv_cache, plan), plan.block_table

    def gather_prefill_cache(
        self,
        kv_cache: torch.Tensor,
        plan: HiSparsePrefillStagingPlan,
    ) -> torch.Tensor:
        """Gather this runtime's host cache using a shared staging plan."""
        if kv_cache.shape[1] != plan.block_size:
            raise ValueError(
                f"HiSparse staging block size {plan.block_size} does not match "
                f"the KV cache block size {kv_cache.shape[1]}."
            )
        row_width = kv_cache.shape[-1]

        staged = torch.empty(
            (
                plan.row_ids.shape[1] // plan.block_size,
                plan.block_size,
                row_width,
            ),
            dtype=kv_cache.dtype,
            device=plan.block_table.device,
        )
        torch.ops._C_cache_ops.hisparse_gather_plan(
            kv_cache,
            staged,
            plan.row_ids,
            plan.dst_rows,
            plan.miss_mask,
            None,
            None,
            0,
            self.row_value_bytes or 0,
        )
        return staged

    def reset_hot_state(self) -> None:
        """Drop all hot-buffer bookkeeping (hits become misses)."""
        if self.device_global_indices is None:
            return
        assert self.lru_slots is not None and self._lru_init is not None
        self.device_global_indices.fill_(-1)
        self.lru_slots.copy_(self._lru_init.expand_as(self.lru_slots))

    def invalidate_slots(self, slots: torch.Tensor) -> None:
        """Drop hot copies of recycled global slots."""
        if self._host_cache is None:
            return
        assert self.device_global_indices is not None
        stale = torch.isin(
            self.device_global_indices, slots.to(device=self.device, dtype=torch.int32)
        )
        self.device_global_indices[stale] = -1

    def backup_rows(
        self,
        src_cache: torch.Tensor,
        src_indices: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        assert self._host_cache is not None
        torch.ops._C_cache_ops.hisparse_backup(
            src_cache,
            src_indices,
            self._host_cache,
            dst_slots,
            self.row_value_bytes or 0,
        )

    def backup_caches(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the static tensors needed by the all-layer backup plan."""
        assert self.hot is not None and self._host_cache is not None
        return self.hot.cache, self._host_cache

    def swap_in(
        self,
        *,
        resident: HiSparseCacheHandle | None = None,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
        block_size: int,
        return_valid_counts: bool = False,
        produce_plan: bool = False,
    ) -> HiSparseTopKResult:
        """Resolve top-k positions against resident, hot, then host storage."""
        num_tokens = topk_indices.shape[0]
        assert (
            self._host_cache is not None
            and self.hot is not None
            and self.hot.block_size == block_size
            and self.hot_block_table is not None
        )
        request_state_indices = self.request_state_indices
        assert request_state_indices is not None

        relative_indices = topk_indices[:num_tokens].contiguous()

        hot_indices = self._plan.hot_indices[:num_tokens]
        valid_counts = (
            self._plan.valid_counts[:num_tokens] if return_valid_counts else None
        )
        if produce_plan:
            compact_miss_globals = self._plan.miss_global_indices[:num_tokens]
            compact_miss_hots = self._plan.miss_hot_indices[:num_tokens]
            compact_miss_counts = self._plan.miss_counts[:num_tokens]
        else:
            compact_miss_globals = None
            compact_miss_hots = None
            compact_miss_counts = None

        attention_indices = self._plan.attention_indices[:num_tokens]

        # Padded rows are skipped by the kernel (request_state_indices) and must
        # come out as -1 so the attention kernel masks them.
        torch.ops._C_cache_ops.hisparse_swap_in(
            self._host_cache,
            self.hot.cache,
            self.hot_block_table,
            relative_indices,
            hot_indices,
            self.device_global_indices,
            self.lru_slots,
            request_state_indices,
            self.region_stride,
            None,
            self._swap_stats,
            attention_indices,
            self.hot.attention_block_stride,
            req_id_per_token[:num_tokens].contiguous(),
            block_table,
            block_size,
            None,
            valid_counts,
            compact_miss_globals,
            compact_miss_hots,
            compact_miss_counts,
            resident.block_table if resident is not None else None,
            resident.view.block_size
            if resident is not None and resident.view is not None
            else 0,
            0,
            self.row_value_bytes or 0,
        )

        if produce_plan and self.followers:
            self._prefetch_group(num_tokens)

        if not return_valid_counts:
            return self.hot.attention_cache, attention_indices
        assert valid_counts is not None
        return self.hot.attention_cache, attention_indices, valid_counts

    def _gather_plan_into(self, num_tokens: int) -> None:
        assert self.hot is not None
        torch.ops._C_cache_ops.hisparse_gather_compact(
            self._host_cache,
            self.hot.cache,
            self._plan.miss_global_indices[:num_tokens],
            self._plan.miss_hot_indices[:num_tokens],
            self._plan.miss_counts[:num_tokens],
            self.row_value_bytes or 0,
        )

    def _prefetch_group(self, num_tokens: int) -> None:
        compute = torch.accelerator.current_stream(self.device)
        self._copy_stream.wait_stream(compute)
        with self._copy_stream:
            for follower in self.followers:
                if follower._host_cache is None:
                    follower._prefetch_event = None
                    continue
                follower._gather_plan_into(num_tokens)
                if follower._prefetch_event is None:
                    follower._prefetch_event = torch.Event()
                follower._prefetch_event.record(self._copy_stream)

    def apply_plan(
        self,
        *,
        block_size: int,
        num_tokens: int,
        return_valid_counts: bool = False,
    ) -> HiSparseTopKResult:
        """Replay the leader's swap plan for an index-sharing follower."""
        n = num_tokens
        assert self.leader is not None
        assert self.hot is not None and self.hot.block_size == block_size
        assert self.leader.hot is not None
        assert self.hot.attention_block_stride == self.leader.hot.attention_block_stride
        attention_indices = self._plan.attention_indices[:n]
        if self._prefetch_event is not None:
            torch.accelerator.current_stream(self.device).wait_event(
                self._prefetch_event
            )
            self._prefetch_event = None
        else:
            self._gather_plan_into(num_tokens)
        if return_valid_counts:
            return (
                self.hot.attention_cache,
                attention_indices,
                self._plan.valid_counts[:n],
            )
        return self.hot.attention_cache, attention_indices


class HiSparseCacheHandle:
    """Attention-facing handle for resident KV and sparse offload state."""

    def __init__(self, runtime: HiSparseOffloadRuntime) -> None:
        self.view: PagedCacheView | None = None
        self.block_table: torch.Tensor | None = None
        self.slot_mapping: torch.Tensor | None = None
        self.compressed_slot_mapping: torch.Tensor | None = None
        self.logical_block_size = 0
        self.runtime = runtime
        self.fully_resident = False
        self.decode_batch = False

    def bind_cache(
        self,
        raw_tensor: torch.Tensor,
        *,
        byte_offset: int,
        block_stride: int,
        num_blocks: int,
        block_size: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        storage_block_size = self.runtime.storage_block_size or block_size
        self.view = PagedCacheView.bind(
            raw_tensor,
            dtype=self.runtime.kv_dtype,
            row_width=self.runtime.row_width,
            byte_offset=byte_offset,
            block_stride=block_stride,
            num_blocks=num_blocks,
            block_size=storage_block_size,
        )
        self.block_table = block_table
        self.slot_mapping = slot_mapping
        if self.runtime.storage_block_size is not None:
            self.compressed_slot_mapping = torch.empty_like(slot_mapping)
        self.logical_block_size = block_size

    def get_compressed_slot_mapping(
        self, positions: torch.Tensor, compress_ratio: int
    ) -> torch.Tensor:
        assert self.slot_mapping is not None
        assert self.compressed_slot_mapping is not None
        assert self.view is not None
        return compress_hisparse_slot_mapping(
            self.slot_mapping,
            positions,
            logical_block_size=self.logical_block_size,
            storage_block_size=self.view.block_size,
            compress_ratio=compress_ratio,
            out=self.compressed_slot_mapping,
        )

    def write_rows(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        *,
        mirror_to_host: bool,
    ) -> None:
        assert self.view is not None and self.slot_mapping is not None
        host_slots = slot_mapping.flatten()
        num_rows = min(kv_c_normed.shape[0], host_slots.numel())
        if not mirror_to_host:
            num_rows = min(num_rows, self.runtime.max_num_reqs)
        if num_rows == 0:
            return
        resident_slots = self.slot_mapping[:num_rows]
        ops.concat_and_cache_mla(
            kv_c_normed[:num_rows],
            k_pe[:num_rows].squeeze(1),
            self.view.cache,
            resident_slots,
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
        if mirror_to_host or self.runtime.eager_host_mirror:
            self.runtime.backup_rows(
                self.view.cache,
                resident_slots,
                host_slots[:num_rows]
                .to(device=self.runtime.device, dtype=torch.int64)
                .contiguous(),
            )

    def resolve_topk(
        self,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        block_size: int,
        return_valid_counts: bool = False,
    ) -> HiSparseTopKResult:
        num_tokens = topk_indices.shape[0]
        if self.fully_resident:
            assert self.block_table is not None and self.view is not None
            converted = triton_convert_req_index_to_global_index(
                req_id_per_token[:num_tokens],
                self.block_table,
                topk_indices,
                BLOCK_SIZE=self.view.block_size,
                PHYSICAL_BLOCK_STRIDE=self.view.attention_block_stride,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
                BLOCK_N=gcd(topk_indices.shape[1], 128),
                return_valid_counts=return_valid_counts,
            )
            if return_valid_counts:
                indices, valid_counts = converted
                return self.view.attention_cache, indices, valid_counts
            return self.view.attention_cache, converted
        if self.runtime.leader is not None:
            return self.runtime.apply_plan(
                block_size=block_size,
                num_tokens=num_tokens,
                return_valid_counts=return_valid_counts,
            )
        return self.runtime.swap_in(
            resident=self,
            req_id_per_token=req_id_per_token[:num_tokens],
            block_table=block_table,
            topk_indices=topk_indices,
            block_size=block_size,
            return_valid_counts=return_valid_counts,
            produce_plan=bool(self.runtime.followers),
        )


def create_hisparse_cache_handle(
    vllm_config: VllmConfig,
    model_top_k: int,
    *,
    index_group_scope: object,
    is_index_group_leader: bool,
    row_width: int,
    kv_dtype: torch.dtype,
    device: torch.device | str | None = None,
    storage_block_size: int | None = None,
    row_value_bytes: int | None = None,
) -> HiSparseCacheHandle | None:
    config = ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k)
    if config is None:
        return None

    max_num_reqs = vllm_config.scheduler_config.max_num_seqs
    if device is None:
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )

    runtime = HiSparseOffloadRuntime(
        config=config,
        max_num_reqs=max_num_reqs,
        row_width=row_width,
        kv_dtype=kv_dtype,
        device=device,
        storage_block_size=storage_block_size,
        row_value_bytes=row_value_bytes,
    )
    runtime.join_index_group(index_group_scope, is_index_group_leader)
    kv_transfer_config = vllm_config.kv_transfer_config
    runtime.eager_host_mirror = bool(
        kv_transfer_config is not None and kv_transfer_config.is_kv_producer
    )
    logger.info_once(
        "Enabled experimental HiSparse HMA hot cache: top_k=%d, "
        "device_buffer_size=%d (%d LRU rows), host_pool_gib=%s, "
        "max_num_seqs=%d.",
        config.top_k,
        config.device_buffer_size,
        config.device_buffer_size,
        config.host_pool_gib,
        max_num_reqs,
    )
    return HiSparseCacheHandle(runtime)
