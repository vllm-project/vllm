# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental pressure-adaptive HiSparse decode for sparse MLA.

Indexer KV, paged resident MLA KV, and per-request hot buffers share one GPU
HMA pool. New tokens are written to ordinary resident pages. Fully resident
batches use the regular paged attention path and do not allocate hot buffers
or copy KV through host memory.

When the shared pool comes under pressure, sealed resident pages are copied to
the pinned host source pool and recycled after the copy is enqueued. Only
requests with CPU-only history acquire an ephemeral hot buffer. Their sparse
lookup hierarchy is resident GPU page, hot-buffer hit, then host-source miss;
misses are gathered without host synchronization. The host source remains the
authoritative backing store for spilled pages and prefix-cache entries.
"""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass, field
from math import gcd

import numpy as np
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
HISPARSE_KERNEL_BLOCK_SIZE = 64


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
    ) -> ResolvedHiSparseConfig | None:
        config = vllm_config.attention_config.hisparse_config
        if config is None:
            return None

        # Default 2x top_k: at exactly top_k the LRU has zero slack and
        # boundary entries thrash between steps.
        configured_size = config.device_buffer_size
        if configured_size is None:
            device_buffer_size = round_up(2 * model_top_k, HISPARSE_KERNEL_BLOCK_SIZE)
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
        if configured_size is not None:
            padding = -device_buffer_size % HISPARSE_KERNEL_BLOCK_SIZE
            if padding:
                logger.warning(
                    "HiSparse device_buffer_size=%d is not aligned to the "
                    "%d-token kernel block size and allocates %d unused "
                    "padding rows per hot group. Use %d to use all allocated "
                    "rows.",
                    device_buffer_size,
                    HISPARSE_KERNEL_BLOCK_SIZE,
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
    _STATE.pinned_host_pools.append(registered)
    return registered[:size]


def register_indexer_source(
    layer_name: str, cache: torch.Tensor, slot_mapping: torch.Tensor
) -> None:
    _STATE.indexer_sources[layer_name] = (cache, slot_mapping)


def get_indexer_source(
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    return _STATE.indexer_sources.get(layer_name)


def _covers_registered_host_range(ptr: int, nbytes: int) -> bool:
    return any(
        pool.data_ptr() <= ptr and ptr + nbytes <= pool.data_ptr() + pool.nbytes
        for pool in _STATE.pinned_host_pools
    )


def release_pinned_state() -> None:
    """Synchronize, unregister host KV pools, and drop global state."""
    if _STATE.pinned_host_pools:
        try:
            torch.accelerator.synchronize()
        except RuntimeError as e:
            logger.warning(
                "HiSparse: CUDA context unusable at teardown (%s); leaving "
                "%d host-pool tensors pinned for kernel exit reclaim.",
                e,
                len(_STATE.pinned_host_pools),
            )
            return

        cudart = torch.cuda.cudart()
        release_start = time.perf_counter()
        freed_bytes = 0
        while _STATE.pinned_host_pools:
            tensor = _STATE.pinned_host_pools[-1]
            err = cudart.cudaHostUnregister(tensor.data_ptr())
            if err.value != 0:
                logger.warning(
                    "HiSparse: cudaHostUnregister failed (code=%d); leaving "
                    "%d host-pool tensors pinned for kernel exit reclaim.",
                    err.value,
                    len(_STATE.pinned_host_pools),
                )
                cudart.cudaGetLastError()
                break
            freed_bytes += tensor.nbytes
            _STATE.pinned_host_pools.pop()
        if freed_bytes:
            logger.info(
                "HiSparse: unpinned %.1f GiB of host pool in %.1fs.",
                freed_bytes / 2**30,
                time.perf_counter() - release_start,
            )

    _STATE.pinned_staging = None
    _STATE.pinned_staging_event = None
    for leader in _STATE.coordinators:
        for coordinator in (leader, *leader.group_shared):
            if coordinator._host_cache is not None:
                coordinator._host_cache = None
    _STATE.coordinators.clear()
    _STATE.group_plans.clear()
    _STATE.copy_streams.clear()
    _STATE.request_state_indices.clear()
    _STATE.request_state_source_ptrs.clear()
    _STATE.current_group_leader = None
    with suppress(RuntimeError):
        torch._C._host_emptyCache()
    _STATE.indexer_sources.clear()


def _pinned_to_device(values: list[int], device: torch.device) -> torch.Tensor:
    """Copy a small int list to ``device`` via pinned staging (grow-on-demand,
    power-of-2) instead of a per-step pageable tensor."""
    n = len(values)
    if _STATE.pinned_staging is None or _STATE.pinned_staging.shape[0] < n:
        size = 1 << max(10, (n - 1).bit_length())
        _STATE.pinned_staging = torch.empty(size, dtype=torch.long, pin_memory=True)
        _STATE.pinned_staging_event = None
    if _STATE.pinned_staging_event is not None:
        _STATE.pinned_staging_event.synchronize()
    staging = _STATE.pinned_staging[:n]
    staging.copy_(torch.from_numpy(np.asarray(values, dtype=np.int64)))
    out = staging.to(device, non_blocking=True)
    if _STATE.pinned_staging_event is None:
        _STATE.pinned_staging_event = torch.Event()
    _STATE.pinned_staging_event.record(torch.accelerator.current_stream(device))
    return out


def invalidate_blocks(block_ids: list[int], block_size: int) -> None:
    """Drop cached HiSparse state for the given blocks in every layer.

    Called from the KV connector lifecycle when blocks are (re)assigned to newly
    scheduled or preemption-resumed requests, before a forward can select them.
    This makes block recycling safe for any writer (local prefill, connector
    RDMA into host memory) without per-connector reporting hooks.
    """
    if not block_ids:
        return
    slots: torch.Tensor | None = None
    for coordinator in _STATE.coordinators:
        if slots is None:
            # Built once on device and shared by every leader.
            blocks = _pinned_to_device(block_ids, coordinator.device)
            offsets = torch.arange(
                block_size, dtype=torch.long, device=coordinator.device
            )
            slots = (blocks[:, None] * block_size + offsets[None, :]).flatten()
        coordinator.invalidate_slots(slots)


def wait_for_hisparse_host_writes() -> None:
    """Wait for pending GPU writes before accessing host KV from the CPU."""
    waited_devices: set[torch.device] = set()
    for coordinator in _STATE.coordinators:
        device = coordinator.device
        event = coordinator._host_write_event
        if event is not None and device not in waited_devices:
            event.synchronize()
            waited_devices.add(device)


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
    """Group-shared swap-in plan for GLM-5.2 index sharing.

    A "full" layer's swap_in(produce_plan=True) writes the resolved plan here;
    its "shared" layers replay it via apply_plan without re-resolving LRU. One
    set per (device, max_rows, top_k), shared across all coordinators -- mirrors
    the model-global topk_indices_buffer (layers run sequentially: full writes,
    its shared layers read, the next full overwrites). Static shapes so the
    replay is CUDA-graph-capture safe.
    """

    __slots__ = (
        "hot_indices",
        "miss_global_indices",
        "miss_hot_indices",
        "miss_counts",
        "valid_counts",
    )

    def __init__(self, device: torch.device, max_rows: int, top_k: int) -> None:
        self.hot_indices = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.miss_global_indices = torch.empty(
            (max_rows, top_k), dtype=torch.int32, device=device
        )
        self.miss_hot_indices = torch.empty_like(self.miss_global_indices)
        self.miss_counts = torch.empty(max_rows, dtype=torch.int32, device=device)
        self.valid_counts = torch.empty(max_rows, dtype=torch.int32, device=device)


@dataclass
class _HiSparseProcessState:
    coordinators: list[HiSparseCoordinator] = field(default_factory=list)
    current_group_leader: HiSparseCoordinator | None = None
    pinned_staging: torch.Tensor | None = None
    pinned_staging_event: torch.Event | None = None
    pinned_host_pools: list[torch.Tensor] = field(default_factory=list)
    indexer_sources: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    group_plans: dict[tuple[str, int, int], _GroupPlan] = field(default_factory=dict)
    copy_streams: dict[str, torch.Stream] = field(default_factory=dict)
    request_state_indices: dict[tuple[str, int], torch.Tensor] = field(
        default_factory=dict
    )
    request_state_source_ptrs: dict[tuple[str, int], int] = field(default_factory=dict)


_STATE = _HiSparseProcessState()


def _get_group_plan(device: torch.device, max_rows: int, top_k: int) -> _GroupPlan:
    key = (str(device), max_rows, top_k)
    plan = _STATE.group_plans.get(key)
    if plan is None:
        plan = _GroupPlan(device, max_rows, top_k)
        _STATE.group_plans[key] = plan
    return plan


def _get_copy_stream(device: torch.device) -> torch.Stream:
    key = str(device)
    stream = _STATE.copy_streams.get(key)
    if stream is None:
        stream = torch.Stream(device=device)
        _STATE.copy_streams[key] = stream
    return stream


def _get_request_state_indices(device: torch.device, max_num_reqs: int) -> torch.Tensor:
    key = (str(device), max_num_reqs)
    indices = _STATE.request_state_indices.get(key)
    if indices is None:
        indices = torch.arange(max_num_reqs, dtype=torch.int32, device=device)
        _STATE.request_state_indices[key] = indices
    return indices


class HiSparseCoordinator:
    """Per-layer resident/hot/host resolver for sparse MLA KV rows.

    Hot-buffer hits are keyed by global host slot ID. Correctness therefore
    requires stale copies to be invalidated before a recycled host block is
    reused; the KV connector lifecycle performs that invalidation for every
    block assigned to an incoming request.
    """

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
        self.config = config
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

        self.hot_cache: torch.Tensor | None = None
        self.attention_hot_cache: torch.Tensor | None = None
        self.resident_block_table: torch.Tensor | None = None
        self.resident_slot_mapping: torch.Tensor | None = None
        self.compressed_resident_slot_mapping: torch.Tensor | None = None
        self.resident_group_id = -1
        self.resident_block_size = 0
        self.resident_logical_block_size = 0
        self.fully_resident_batch = False
        self.decode_batch = False
        self.hot_block_table: torch.Tensor | None = None
        self.attention_block_stride = 0
        self._hot_indices = torch.empty(
            (max_num_reqs, config.top_k), dtype=torch.int32, device=self.device
        )
        self._attention_indices = torch.empty_like(self._hot_indices)
        self._valid_counts = torch.empty(
            max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Per-request LRU state; released in join_group for index-sharing
        # "shared" layers, which replay their leader's plan and never resolve
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
        self.request_state_indices = _get_request_state_indices(
            self.device, max_num_reqs
        )

        # In-kernel hit/miss counters (telemetry). stats_row_bytes converts
        # misses to gathered bytes; plan-once wiring adds each shared
        # layer's row bytes to its leader (the shared layers re-gather the
        # leader's misses), so the leader's counter covers the whole group.
        self._swap_stats = torch.zeros(2, dtype=torch.uint64, device=self.device)
        self.stats_row_bytes = row_bytes
        _STATE.coordinators.append(self)

        self._plan = _get_group_plan(self.device, max_num_reqs, config.top_k)
        self.group_shared: list[HiSparseCoordinator] = []
        self.leader: HiSparseCoordinator | None = None
        self._prefetch_event: torch.Event | None = None
        self._copy_stream = _get_copy_stream(self.device)

        self._host_cache: torch.Tensor | None = None
        self._host_write_event: torch.Event | None = None
        self.eager_host_mirror = False

    def set_request_state_indices(
        self, indices: torch.Tensor, *, force: bool = False
    ) -> None:
        if indices.numel() > self.max_num_reqs:
            raise ValueError(
                "HiSparse request-state mapping exceeds max_num_seqs: "
                f"{indices.numel()} > {self.max_num_reqs}."
            )
        if torch.cuda.is_current_stream_capturing():
            return
        key = (str(self.device), self.max_num_reqs)
        source_ptr = indices.data_ptr()
        if not force and _STATE.request_state_source_ptrs.get(key) == source_ptr:
            return
        self.request_state_indices[: indices.numel()].copy_(indices)
        _STATE.request_state_source_ptrs[key] = source_ptr

    def join_indexer_group(self, has_indexer: bool) -> None:
        if has_indexer:
            _STATE.current_group_leader = self
        elif _STATE.current_group_leader is not None:
            self.join_group(_STATE.current_group_leader)

    def join_group(self, leader: HiSparseCoordinator) -> None:
        self.leader = leader
        leader.group_shared.append(self)
        leader.stats_row_bytes += self.stats_row_bytes
        _STATE.coordinators.remove(self)
        self.device_global_indices = None
        self.lru_slots = None
        self._lru_init = None

    def hot_cache_paged(self, block_size: int) -> torch.Tensor:
        """Hot buffer shaped like a regular paged MLA cache."""
        assert self.attention_hot_cache is not None
        assert self.attention_hot_cache.shape[1] == block_size
        return self.attention_hot_cache

    def bind_hot_cache(
        self,
        raw_tensor: torch.Tensor,
        *,
        byte_offset: int,
        block_stride: int,
        num_blocks: int,
        block_size: int,
    ) -> None:
        """Bind this layer's strided view into the shared GPU HMA slab."""
        itemsize = self.kv_dtype.itemsize
        assert byte_offset % itemsize == 0 and block_stride % itemsize == 0
        row_bytes = self.row_width * itemsize
        storage_block_size = self.storage_block_size or block_size
        self.hot_cache = torch.as_strided(
            raw_tensor.view(self.kv_dtype),
            size=(num_blocks, storage_block_size, self.row_width),
            stride=(block_stride // itemsize, self.row_width, 1),
            storage_offset=byte_offset // itemsize,
        )
        if block_stride % (storage_block_size * row_bytes) == 0:
            attention_storage = raw_tensor[byte_offset:].view(self.kv_dtype)
            self.attention_hot_cache = attention_storage.view(
                -1, storage_block_size, self.row_width
            )
            self.attention_block_stride = block_stride // row_bytes
        else:
            self.attention_hot_cache = self.hot_cache
            self.attention_block_stride = storage_block_size

    def bind_hot_block_table(self, block_table: torch.Tensor) -> None:
        self.hot_block_table = block_table

    def bind_resident_cache(
        self,
        raw_tensor: torch.Tensor,
        *,
        byte_offset: int,
        block_stride: int,
        num_blocks: int,
        block_size: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        group_id: int,
    ) -> None:
        """Bind the paged resident view sharing the hot HMA layout."""
        itemsize = self.kv_dtype.itemsize
        storage_block_size = self.storage_block_size or block_size
        resident = torch.as_strided(
            raw_tensor.view(self.kv_dtype),
            size=(num_blocks, storage_block_size, self.row_width),
            stride=(block_stride // itemsize, self.row_width, 1),
            storage_offset=byte_offset // itemsize,
        )
        if self.hot_cache is not None:
            assert resident.untyped_storage().data_ptr() == (
                self.hot_cache.untyped_storage().data_ptr()
            )
            assert resident.stride() == self.hot_cache.stride()
        self.resident_block_table = block_table
        self.resident_slot_mapping = slot_mapping
        if self.storage_block_size is not None:
            self.compressed_resident_slot_mapping = torch.empty_like(slot_mapping)
        self.resident_group_id = group_id
        self.resident_block_size = storage_block_size
        self.resident_logical_block_size = block_size

    def get_compressed_resident_slot_mapping(
        self, positions: torch.Tensor, compress_ratio: int
    ) -> torch.Tensor:
        """Return the current batch's resident slots in compressed units."""
        assert self.resident_slot_mapping is not None
        assert self.compressed_resident_slot_mapping is not None
        return compress_hisparse_slot_mapping(
            self.resident_slot_mapping,
            positions,
            logical_block_size=self.resident_logical_block_size,
            storage_block_size=self.resident_block_size,
            compress_ratio=compress_ratio,
            out=self.compressed_resident_slot_mapping,
        )

    def bind_source_cache(self, kv_cache: torch.Tensor) -> None:
        if kv_cache.dtype != self.kv_dtype or kv_cache.shape[-1] != self.row_width:
            raise ValueError(
                "HiSparse coordinator bound to a KV cache with mismatched "
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

    def resolve_resident(
        self,
        req_id_per_token: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        return_valid_counts: bool,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Resolve a fully resident batch without consulting hot-cache state."""
        assert self.resident_block_table is not None
        assert self.resident_block_size > 0
        converted = triton_convert_req_index_to_global_index(
            req_id_per_token[: topk_indices.shape[0]],
            self.resident_block_table,
            topk_indices,
            BLOCK_SIZE=self.resident_block_size,
            PHYSICAL_BLOCK_STRIDE=self.attention_block_stride,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            BLOCK_N=gcd(topk_indices.shape[1], 128),
            return_valid_counts=return_valid_counts,
        )
        cache = self.hot_cache_paged(self.resident_block_size)
        if return_valid_counts:
            indices, valid_counts = converted
            return cache, indices, valid_counts
        return cache, converted

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
        """Gather one layer's host cache using a shared staging plan."""
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

    def _invalidate_hot_copies(self, slots: torch.Tensor) -> None:
        assert self.device_global_indices is not None
        # Block recycling runs once per batch. If it becomes frequent enough
        # for this full-state scan to matter, maintain a reverse slot index.
        stale = torch.isin(
            self.device_global_indices, slots.to(device=self.device, dtype=torch.int32)
        )
        self.device_global_indices[stale] = -1

    def invalidate_slots(self, slots: torch.Tensor) -> None:
        """Drop all cached state for the given global slots.

        Called when blocks are (re)assigned to a request, regardless of who
        writes them (local prefill, KV connector RDMA, ...). Hot-buffer
        copies of recycled slots must never be served as hits.
        """
        if self._host_cache is None:
            return
        self._invalidate_hot_copies(slots)

    def _backup_rows(
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

    def backup_compressed_rows(
        self, src_cache: torch.Tensor, src_slots: torch.Tensor, dst_slots: torch.Tensor
    ) -> None:
        """Mirror already-encoded compressed rows into the host source."""
        self._backup_rows(src_cache, src_slots, dst_slots)

    def backup_caches(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the static tensors needed by the all-layer backup plan."""
        assert self.hot_cache is not None and self._host_cache is not None
        return self.hot_cache, self._host_cache

    # ------------------------------------------------------- newest-token path

    def write_newest_rows(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Decode-step KV update.

        Writes the newest token into its ordinary resident GPU page. The runtime
        materializes sealed pages in the host pool only when required.
        """
        assert self.hot_cache is not None and self.resident_slot_mapping is not None
        # Pad clamp: the forward can run more rows than the scheduler
        # produced (DP alignment pads to a peer's batch, eager/PIECEWISE pads
        # to a capture size) while slot_mapping stays unpadded. Real rows are
        # always a prefix of both, so clamp instead of asserting — a length
        # mismatch trips the backup kernel's shape check and kills the rank
        # (and with it the whole DP fleet).
        num_tokens = min(kv_c_normed.shape[0], slot_mapping.numel(), self.max_num_reqs)
        if num_tokens == 0:
            return
        global_slots = slot_mapping[:num_tokens].to(torch.int64)
        resident_slots = self.resident_slot_mapping[:num_tokens]

        # The cache-update kernel skips -1 slots introduced by graph padding.
        ops.concat_and_cache_mla(
            kv_c_normed[:num_tokens],
            k_pe[:num_tokens].squeeze(1),
            self.hot_cache,
            resident_slots,
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
        if self.eager_host_mirror:
            self._backup_rows(
                self.hot_cache,
                resident_slots,
                global_slots,
            )
        # Recycled-slot hygiene is handled at block-assignment time. The KV
        # connector invalidates every block (re)assigned to any request
        # (new, resumed, or growing) before the step that first writes it,
        # so no per-step in-graph invalidation is needed here.

    def write_rows_to_host(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Write prefill/mixed-batch rows to resident and host storage.

        Local prefill on a decode instance (router shortcut, preemption
        resume, recompute after a failed KV load) stages attention context
        from the host pool. Quantize new rows into resident pages when they
        exist, then mirror them to their authoritative global host slots.
        Recycled-slot hygiene is handled at block-assignment time by the
        KV connector lifecycle, so no hot-copy invalidation is needed here.
        """
        # CUDA graph padding can make kv_c_normed/k_pe longer than slot_mapping.
        # Only rows represented by slot_mapping correspond to real KV writes.
        flat_slots = slot_mapping.flatten()[: kv_c_normed.shape[0]]
        num_rows = flat_slots.numel()
        if num_rows == 0:
            return
        dst = flat_slots.to(device=self.device, dtype=torch.int64).contiguous()
        real_kv_rows = kv_c_normed[:num_rows]
        real_pe_rows = k_pe[:num_rows]
        assert self.hot_cache is not None and self.resident_slot_mapping is not None
        src = self.resident_slot_mapping[:num_rows]
        ops.concat_and_cache_mla(
            real_kv_rows,
            real_pe_rows.squeeze(1),
            self.hot_cache,
            src,
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
        self._backup_rows(self.hot_cache, src, dst)

    # ---------------------------------------------------------------- swap-in

    def swap_in(
        self,
        *,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
        block_size: int,
        return_valid_counts: bool = False,
        produce_plan: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Resolve top-k positions against the hot buffer.

        Returns ``(hot_cache_paged, hot_indices)`` (plus ``valid_counts`` when
        requested). ``hot_cache_paged`` has the same paged layout as a regular
        MLA KV cache; ``hot_indices`` are global token ids within it.
        """
        num_tokens = topk_indices.shape[0]
        assert (
            self._host_cache is not None
            and self.hot_cache is not None
            and self.hot_block_table is not None
        )

        relative_indices = topk_indices[:num_tokens].contiguous()

        if produce_plan:
            hot_indices = self._plan.hot_indices[:num_tokens]
            valid_counts = self._plan.valid_counts[:num_tokens]
            compact_miss_globals = self._plan.miss_global_indices[:num_tokens]
            compact_miss_hots = self._plan.miss_hot_indices[:num_tokens]
            compact_miss_counts = self._plan.miss_counts[:num_tokens]
        else:
            hot_indices = self._hot_indices[:num_tokens]
            valid_counts = (
                self._valid_counts[:num_tokens] if return_valid_counts else None
            )
            compact_miss_globals = None
            compact_miss_hots = None
            compact_miss_counts = None

        attention_indices = self._attention_indices[:num_tokens]

        # Padded rows are skipped by the kernel (request_state_indices) and must
        # come out as -1 so the attention kernel masks them.
        torch.ops._C_cache_ops.hisparse_swap_in(
            self._host_cache,
            self.hot_cache,
            self.hot_block_table,
            relative_indices,
            hot_indices,
            self.device_global_indices,
            self.lru_slots,
            self.request_state_indices,
            self.region_stride,
            None,
            self._swap_stats,
            attention_indices,
            self.attention_block_stride,
            req_id_per_token[:num_tokens].contiguous(),
            block_table,
            block_size,
            None,
            valid_counts,
            compact_miss_globals,
            compact_miss_hots,
            compact_miss_counts,
            self.resident_block_table,
            self.resident_block_size,
            0,
            self.row_value_bytes or 0,
        )

        if produce_plan and self.group_shared:
            self._prefetch_group(num_tokens)

        if not return_valid_counts:
            return self.hot_cache_paged(block_size), attention_indices
        assert valid_counts is not None
        return self.hot_cache_paged(block_size), attention_indices, valid_counts

    def _gather_plan_into(self, num_tokens: int) -> None:
        torch.ops._C_cache_ops.hisparse_gather_compact(
            self._host_cache,
            self.hot_cache,
            self._plan.miss_global_indices[:num_tokens],
            self._plan.miss_hot_indices[:num_tokens],
            self._plan.miss_counts[:num_tokens],
            self.row_value_bytes or 0,
        )

    def _prefetch_group(self, num_tokens: int) -> None:
        compute = torch.accelerator.current_stream(self.device)
        self._copy_stream.wait_stream(compute)
        with self._copy_stream:
            for shared in self.group_shared:
                if shared._host_cache is None:
                    shared._prefetch_event = None
                    continue
                shared._gather_plan_into(num_tokens)
                if shared._prefetch_event is None:
                    shared._prefetch_event = torch.Event()
                shared._prefetch_event.record(self._copy_stream)

    def apply_plan(
        self,
        *,
        block_size: int,
        num_tokens: int,
        return_valid_counts: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Replay the group's plan for an index-sharing "shared" layer.

        A "full" layer resolved the plan via ``swap_in(produce_plan=True)``;
        this gathers only THIS layer's own missed rows into the identical
        planned hot slots, with no LRU resolution. Fixed shape -> capture-safe.
        """
        n = num_tokens
        assert self.leader is not None
        assert self.attention_block_stride == self.leader.attention_block_stride
        attention_indices = self.leader._attention_indices[:n]
        if self._prefetch_event is not None:
            torch.accelerator.current_stream(self.device).wait_event(
                self._prefetch_event
            )
            self._prefetch_event = None
        else:
            self._gather_plan_into(num_tokens)
        if return_valid_counts:
            return (
                self.hot_cache_paged(block_size),
                attention_indices,
                self._plan.valid_counts[:n],
            )
        return self.hot_cache_paged(block_size), attention_indices


def create_hisparse_coordinator(
    vllm_config: VllmConfig,
    model_top_k: int,
    *,
    row_width: int,
    kv_dtype: torch.dtype,
    device: torch.device | str | None = None,
    storage_block_size: int | None = None,
    row_value_bytes: int | None = None,
) -> HiSparseCoordinator | None:
    config = ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k)
    if config is None:
        return None

    max_num_reqs = vllm_config.scheduler_config.max_num_seqs
    if device is None:
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )

    coordinator = HiSparseCoordinator(
        config=config,
        max_num_reqs=max_num_reqs,
        row_width=row_width,
        kv_dtype=kv_dtype,
        device=device,
        storage_block_size=storage_block_size,
        row_value_bytes=row_value_bytes,
    )
    kv_transfer_config = vllm_config.kv_transfer_config
    coordinator.eager_host_mirror = bool(
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
    return coordinator
