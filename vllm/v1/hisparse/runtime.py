# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA HiSparse resident, host, and hot-cache data plane."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import psutil
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import current_stream
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

# fp8_ds_mla KV row: 512 B quantized NoPE + 16 B scales + 128 B RoPE.
FP8_DS_MLA_ROW_BYTES = 656
HiSparseTopKResult: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _get_max_decode_query_len(vllm_config: VllmConfig) -> int:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if (
        speculative_config is not None
        and speculative_config.num_speculative_tokens is not None
    ):
        return 1 + speculative_config.num_speculative_tokens * (
            2 if speculative_config.parallel_drafting else 1
        )
    return 1


def _get_max_swap_rows(vllm_config: VllmConfig) -> int:
    max_query_len = _get_max_decode_query_len(vllm_config)
    scheduler_config = vllm_config.scheduler_config
    return min(
        scheduler_config.max_num_batched_tokens,
        scheduler_config.max_num_seqs * max_query_len,
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
    eager_host_mirror: bool = True

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

        max_query_len = _get_max_decode_query_len(vllm_config)
        min_device_buffer_size = max_query_len * model_top_k
        # Retain the worst-case union of every speculative step's top-k and
        # leave one additional top-k of LRU slack.
        configured_size = config.device_buffer_size
        if configured_size is None:
            device_buffer_size = (max_query_len + 1) * model_top_k
        else:
            device_buffer_size = configured_size

        if device_buffer_size < min_device_buffer_size:
            raise ValueError(
                "HiSparse device_buffer_size must cover every decode query's "
                "index_topk rows. Got "
                f"device_buffer_size={device_buffer_size}, "
                f"max_decode_query_len={max_query_len}, "
                f"index_topk={model_top_k}; expected at least "
                f"{min_device_buffer_size}."
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
            eager_host_mirror=config.eager_host_mirror,
        )


def check_hisparse_host_memory(pool_bytes: int) -> None:
    """Reject a physical host-pool allocation that cannot fit in RAM."""
    mem = psutil.virtual_memory()
    if pool_bytes > mem.available * 0.95:
        raise ValueError(
            f"HiSparse pinned host pool needs ~{pool_bytes / 2**30:.0f} GiB "
            f"but only {mem.available / 2**30:.0f} GiB of RAM is available. "
            "Lower the HiSparseConnector host_pool_gib or leave headroom "
            "for co-tenants."
        )


def allocate_pinned_host_pool(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate and deterministically register an exact-size host KV region."""
    page = 4096
    padded_size = round_up(size, page)
    backing = torch.empty(padded_size + page, dtype=torch.int8, device="cpu")
    aligned_offset = (-backing.data_ptr()) % page
    registered = backing[aligned_offset : aligned_offset + padded_size]
    pin_tensor(registered)
    return registered[:size], registered


def release_pinned_state(
    runtimes: list[HiSparseRuntime], pinned_host_pools: list[torch.Tensor]
) -> None:
    """Synchronize and release registered host KV pools."""
    hisparse_owned_pools = {
        runtime.registered_host_pool.data_ptr()
        for runtime in runtimes
        if runtime.host_pool_registration_owned_by_hisparse
    }
    externally_owned_pools = {
        runtime.registered_host_pool.data_ptr()
        for runtime in runtimes
        if not runtime.host_pool_registration_owned_by_hisparse
    }
    hisparse_owned_pools -= externally_owned_pools
    pinned_host_pools[:] = [
        pool for pool in pinned_host_pools if pool.data_ptr() in hisparse_owned_pools
    ]
    if pinned_host_pools:
        try:
            torch.accelerator.synchronize()
        except RuntimeError as e:
            logger.warning(
                "HiSparse: CUDA context unusable at teardown (%s); leaving "
                "%d host-pool tensors pinned for kernel exit reclaim.",
                e,
                len(pinned_host_pools),
            )
            return

        cudart = torch.cuda.cudart()
        release_start = time.perf_counter()
        freed_bytes = 0
        while pinned_host_pools:
            tensor = pinned_host_pools[-1]
            err = cudart.cudaHostUnregister(tensor.data_ptr())
            if err.value != 0:
                logger.warning(
                    "HiSparse: cudaHostUnregister failed (code=%d); leaving "
                    "%d host-pool tensors pinned for kernel exit reclaim.",
                    err.value,
                    len(pinned_host_pools),
                )
                cudart.cudaGetLastError()
                break
            freed_bytes += tensor.nbytes
            pinned_host_pools.pop()
        if freed_bytes:
            logger.info(
                "HiSparse: unpinned %.1f GiB of host pool in %.1fs.",
                freed_bytes / 2**30,
                time.perf_counter() - release_start,
            )

    for runtime in runtimes:
        del runtime._host_cache
        del runtime.registered_host_pool
        del runtime.hot_backing


def hisparse_prefill_staging_remap(
    block_table: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Renumber blocks without a data-dependent CUDA output allocation."""
    block_ids = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=block_table.device),
            block_table.flatten().clamp(min=0).to(torch.int32),
        )
    )
    sorted_ids, permutation = torch.sort(block_ids)
    is_new = torch.cat(
        (
            torch.ones(1, dtype=torch.bool, device=block_table.device),
            sorted_ids[1:] != sorted_ids[:-1],
        )
    )
    compact_sorted = torch.cumsum(is_new, dim=0, dtype=torch.int32) - 1
    inverse = torch.empty_like(compact_sorted)
    inverse.scatter_(0, permutation, compact_sorted)

    unique_ids = torch.full_like(sorted_ids, -1)
    unique_ids.scatter_(0, compact_sorted, sorted_ids)
    valid = (
        torch.arange(sorted_ids.numel(), device=block_table.device)
        <= compact_sorted[-1]
    )
    unique_ids = torch.where(valid, unique_ids, -1)
    new_bt = inverse[1:].view_as(block_table)
    offsets = torch.arange(block_size, dtype=torch.int32, device=block_table.device)
    row_ids = (
        torch.where(
            valid[:, None],
            unique_ids[:, None] * block_size + offsets,
            -1,
        )
    ).view(1, -1)
    return new_bt, row_ids


@dataclass
class HiSparsePrefillStagingPlan:
    block_table: torch.Tensor
    row_ids: torch.Tensor
    dst_rows: torch.Tensor
    miss_mask: torch.Tensor
    block_size: int
    # Host rows with a GPU-resident copy (adopted shadow pages): the flat
    # resident-cache row to read instead of DMAing from host, -1 for misses.
    gpu_row_ids: torch.Tensor | None = None
    gpu_source_key: tuple[int, int] | None = None

    def ensure_gpu_sources(
        self,
        resident_block_table: torch.Tensor,
        resident_block_size: int,
    ) -> None:
        """Resolve which staged rows can be served from the resident cache.

        Computed once per plan (the resident block table is shared by every
        layer in the group); non-null resident pages become miss_mask=0 rows
        gathered device-to-device by ``gather_prefill_cache``.
        """
        source_key = (resident_block_table.data_ptr(), resident_block_size)
        if self.gpu_source_key == source_key:
            return
        block_size = self.block_size
        if resident_block_size <= 0 or block_size % resident_block_size != 0:
            return
        device = self.row_ids.device
        num_unique = self.row_ids.shape[1] // block_size
        host_ids = self.row_ids[0].view(num_unique, block_size)[:, 0] // block_size
        new_bt = self.block_table.to(torch.int64)
        num_rows, num_cols = new_bt.shape
        if num_rows == 0 or resident_block_table.shape[0] < num_rows:
            return
        # One representative (row, col) per unique host block: any request
        # referencing the block holds an equivalent (refcounted) resident view.
        flat_pos = torch.arange(num_rows * num_cols, device=device)
        rep = torch.full(
            (num_unique,), num_rows * num_cols, device=device, dtype=torch.int64
        )
        rep.scatter_reduce_(
            0, new_bt.reshape(-1), flat_pos, reduce="amin", include_self=True
        )
        rep_row = (rep // num_cols).clamp(max=resident_block_table.shape[0] - 1)
        rep_col = rep % num_cols
        pages_per_block = block_size // resident_block_size
        res_pos = rep_col[:, None] * pages_per_block + torch.arange(
            pages_per_block, device=device
        )
        res_cols = resident_block_table.shape[1]
        res_blocks = torch.where(
            res_pos < res_cols,
            torch.gather(
                resident_block_table.to(torch.int64)[rep_row],
                1,
                res_pos.clamp(max=max(res_cols - 1, 0)),
            ),
            torch.zeros_like(res_pos),
        )
        offsets = torch.arange(block_size, device=device)
        per_off_block = res_blocks[:, offsets // resident_block_size]
        gpu_rows = (
            per_off_block * resident_block_size
            + (offsets % resident_block_size)[None, :]
        )
        # Block id 0 is the null block in both id spaces.
        valid_rows = self.row_ids.view(num_unique, block_size) >= 0
        hit = (per_off_block > 0) & (host_ids[:, None] > 0) & valid_rows
        self.gpu_row_ids = torch.where(hit, gpu_rows, -1).reshape(1, -1).to(torch.int32)
        self.miss_mask = ((self.gpu_row_ids < 0) & valid_rows.view(1, -1)).int()
        self.gpu_source_key = source_key


def build_hisparse_prefill_staging_plan(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    staging_block_capacity: int,
) -> HiSparsePrefillStagingPlan:
    """Build an asynchronous layer-independent host-cache staging remap."""
    device = block_table.device
    used = (seq_lens.to(torch.int64) + block_size - 1) // block_size
    packed_offsets = torch.arange(staging_block_capacity, device=device)
    row_ends = torch.cumsum(used, dim=0)
    rows = torch.searchsorted(row_ends, packed_offsets, right=True)
    valid = rows < block_table.shape[0]
    rows = rows.clamp(max=block_table.shape[0] - 1)
    row_starts = torch.cat((torch.zeros_like(row_ends[:1]), row_ends[:-1]))
    columns = packed_offsets - row_starts[rows]
    valid &= columns < block_table.shape[1]
    flat_positions = rows * block_table.shape[1] + columns.clamp(
        max=block_table.shape[1] - 1
    )
    referenced = torch.where(valid, block_table.flatten()[flat_positions], 0)
    referenced_bt, row_ids = hisparse_prefill_staging_remap(referenced, block_size)
    scratch_index = block_table.numel()
    scatter_positions = torch.where(valid, flat_positions, scratch_index)
    remapped = torch.where(valid, referenced_bt.flatten(), 0)
    new_bt_storage = torch.zeros(scratch_index + 1, dtype=torch.int32, device=device)
    new_bt_storage.scatter_(0, scatter_positions, remapped)
    new_bt = new_bt_storage[:-1].view_as(block_table)
    valid_rows = row_ids >= 0
    dst_rows = torch.arange(row_ids.shape[1], dtype=torch.int32, device=device).view(
        1, -1
    )
    return HiSparsePrefillStagingPlan(
        block_table=new_bt,
        row_ids=row_ids,
        dst_rows=dst_rows,
        miss_mask=valid_rows.to(torch.int32),
        block_size=block_size,
    )


def _has_hisparse_ops() -> bool:
    if not hasattr(torch.ops, "_C_cache_ops"):
        return False
    return (
        hasattr(torch.ops._C_cache_ops, "hisparse_resolve_residency")
        and hasattr(torch.ops._C_cache_ops, "hisparse_invalidate_written_slots")
        and hasattr(torch.ops._C_cache_ops, "hisparse_gather_plan")
        and hasattr(torch.ops._C_cache_ops, "hisparse_gather_compact")
    )


class _SharedTopKState:
    """CUDA-graph-safe residency result shared by index-sharing layers."""

    def __init__(self, device: torch.device, max_rows: int, top_k: int) -> None:
        self.device_topk_rows = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.physical_topk_indices = torch.empty_like(self.device_topk_rows)
        self.swap_host_physical_rows = torch.empty(
            (max_rows, top_k), dtype=torch.int32, device=device
        )
        self.swap_device_physical_rows = torch.empty_like(self.swap_host_physical_rows)
        self.swap_counts = torch.empty(max_rows, dtype=torch.int32, device=device)
        self.valid_topk_counts = torch.empty(max_rows, dtype=torch.int32, device=device)


def _create_shared_topk_state(
    device: torch.device, max_rows: int, top_k: int
) -> _SharedTopKState:
    return _SharedTopKState(device, max_rows, top_k)


def _create_copy_stream(device: torch.device) -> torch.Stream:
    return torch.Stream(device=device)


class HiSparseIndexGroup:
    """GPU index-resolution state shared by one leader and its followers."""

    leader: HiSparseRuntime

    def __init__(
        self,
        device: torch.device,
        max_num_reqs: int,
        region_stride: int,
        max_swap_rows: int,
        top_k: int,
        copy_stream: torch.Stream | None = None,
        logical_topk_ready: torch.Event | None = None,
    ) -> None:
        self.device_global_indices = torch.full(
            (max_num_reqs, region_stride),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.lru_init = torch.arange(region_stride, dtype=torch.int16, device=device)
        self.lru_slots = self.lru_init.repeat(max_num_reqs, 1).contiguous()
        self.shared_topk = _create_shared_topk_state(device, max_swap_rows, top_k)
        self.copy_stream = (
            copy_stream if copy_stream is not None else _create_copy_stream(device)
        )
        self.logical_topk_ready = logical_topk_ready
        self.followers: list[HiSparseRuntime] = []
        self.swap_stats = torch.zeros(2, dtype=torch.uint64, device=device)
        self.swap_stats_host = torch.empty(
            2, dtype=torch.uint64, device="cpu", pin_memory=True
        )
        self.stats_row_bytes = 0


class HiSparseRuntime:
    """Per-cache host/hot data plane and GPU replacement state."""

    hot: PagedCacheView
    hot_backing: torch.Tensor
    hot_block_table: torch.Tensor
    _host_cache: torch.Tensor
    registered_host_pool: torch.Tensor

    def __init__(
        self,
        config: ResolvedHiSparseConfig,
        max_num_reqs: int,
        row_width: int,
        kv_dtype: torch.dtype,
        device: torch.device | str,
        storage_block_size: int | None = None,
        row_value_bytes: int | None = None,
        max_swap_rows: int | None = None,
        index_group: HiSparseIndexGroup | None = None,
        copy_stream: torch.Stream | None = None,
        logical_topk_ready: torch.Event | None = None,
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

        self._layer_ready_event: torch.Event | None = None
        self._swap_staged = False
        self._swap_step = 0
        self.is_group_leader = index_group is None
        if index_group is None:
            index_group = HiSparseIndexGroup(
                self.device,
                max_num_reqs,
                self.region_stride,
                max_swap_rows or max_num_reqs,
                config.top_k,
                copy_stream,
                logical_topk_ready,
            )
            index_group.leader = self
        else:
            index_group.followers.append(self)
        self.index_group = index_group
        index_group.stats_row_bytes += row_bytes

        self.eager_host_mirror = config.eager_host_mirror
        self.resident_source_index = -1
        self.request_state_indices: torch.Tensor | None = None

    @property
    def host_cache(self) -> torch.Tensor:
        return self._host_cache

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
        self.hot_backing = raw_tensor
        self.hot_block_table = block_table

    def bind_source_cache(
        self,
        kv_cache: torch.Tensor,
        *,
        registered_host_pool: torch.Tensor | None = None,
    ) -> None:
        if kv_cache.dtype != self.kv_dtype or kv_cache.shape[-1] != self.row_width:
            raise ValueError(
                "HiSparse runtime bound to a KV cache with mismatched "
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
        if not (kv_cache.is_pinned() or registered_host_pool is not None):
            raise ValueError("HiSparse host-resident KV pool must be pinned memory.")

        self._host_cache = (
            kv_cache
            if self.row_value_bytes is not None
            else kv_cache.view(-1, kv_cache.shape[-1])
        )
        self.registered_host_pool = (
            registered_host_pool if registered_host_pool is not None else kv_cache
        )
        self.host_pool_registration_owned_by_hisparse = True

    def gather_prefill_cache(
        self,
        kv_cache: torch.Tensor,
        plan: HiSparsePrefillStagingPlan,
        resident_cache: torch.Tensor | None = None,
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
        if plan.gpu_row_ids is not None and resident_cache is not None:
            gpu_rows = plan.gpu_row_ids[0].to(torch.long)
            src = gpu_rows.clamp_min(0)
            resident_block_size = resident_cache.shape[1]
            rows = resident_cache[src // resident_block_size, src % resident_block_size]
            if rows.dtype != staged.dtype:
                rows = rows.contiguous().view(staged.dtype)
            staged.view(-1, row_width).copy_(rows)
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
        group = self.index_group
        compute_stream = current_stream()
        compute_stream.wait_stream(group.copy_stream)
        group.device_global_indices.fill_(-1)
        group.lru_slots.copy_(group.lru_init.expand_as(group.lru_slots))
        group.swap_stats.zero_()
        group.copy_stream.wait_stream(compute_stream)

    def invalidate_slots(
        self,
        slots: torch.Tensor,
        request_state_indices: torch.Tensor,
    ) -> None:
        """Drop scheduled requests' hot copies of recycled global slots."""
        slots = slots.to(device=self.device, dtype=torch.int32)
        state_indices = request_state_indices.to(device=self.device, dtype=torch.long)
        sorted_slots = torch.sort(slots).values
        self.invalidate_sorted_slots(sorted_slots, state_indices)

    def invalidate_sorted_slots(
        self,
        sorted_slots: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> None:
        """Drop hot copies using shared, preprocessed invalidation inputs."""
        device_global_indices = self.index_group.device_global_indices
        active_indices = device_global_indices.index_select(0, state_indices)
        positions = torch.searchsorted(sorted_slots, active_indices)
        positions.clamp_(max=sorted_slots.numel() - 1)
        active_indices.masked_fill_(
            sorted_slots[positions] == active_indices,
            -1,
        )
        device_global_indices.index_copy_(0, state_indices, active_indices)

    def invalidate_written_slots(
        self,
        written_slots: torch.Tensor,
        req_id_per_token: torch.Tensor,
    ) -> None:
        num_tokens = min(written_slots.numel(), req_id_per_token.numel())
        if num_tokens == 0:
            return
        assert self.request_state_indices is not None
        torch.ops._C_cache_ops.hisparse_invalidate_written_slots(
            self.index_group.device_global_indices,
            self.request_state_indices,
            req_id_per_token[:num_tokens],
            written_slots[:num_tokens],
        )

    def begin_forward(self) -> None:
        self._swap_step = 0

    def backup_rows(
        self,
        src_cache: torch.Tensor,
        src_indices: torch.Tensor,
        dst_slots: torch.Tensor,
    ) -> None:
        torch.ops._C_cache_ops.hisparse_backup(
            src_cache,
            src_indices,
            self.host_cache,
            dst_slots,
            self.row_value_bytes or 0,
        )

    def _step_rows(self, num_tokens: int) -> slice:
        start = self._swap_step * num_tokens
        stop = start + num_tokens
        if stop > self.index_group.shared_topk.physical_topk_indices.shape[0]:
            raise ValueError(
                "HiSparse swap rows exceed the configured speculative decode "
                f"capacity: stop={stop}, capacity="
                f"{self.index_group.shared_topk.physical_topk_indices.shape[0]}."
            )
        self._swap_step += 1
        return slice(start, stop)

    def _resolve_residency(
        self,
        *,
        resident: HiSparseCacheHandle | None = None,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
        block_size: int,
        return_valid_counts: bool = False,
        shared_rows: slice,
        attention_indices_out: torch.Tensor | None = None,
        valid_counts_out: torch.Tensor | None = None,
    ) -> None:
        """Resolve logical top-k positions and compact rows requiring swaps."""
        num_tokens = topk_indices.shape[0]
        hot = self.hot
        host_cache = self.host_cache
        assert hot.block_size == block_size
        request_state_indices = self.request_state_indices
        assert request_state_indices is not None
        group = self.index_group
        shared = group.shared_topk

        device_topk_rows = shared.device_topk_rows[shared_rows]
        valid_topk_counts = None
        if return_valid_counts:
            valid_topk_counts = (
                shared.valid_topk_counts[shared_rows]
                if valid_counts_out is None
                else valid_counts_out
            )
        physical_topk_indices = (
            shared.physical_topk_indices[shared_rows]
            if attention_indices_out is None
            else attention_indices_out
        )
        swap_host_physical_rows = shared.swap_host_physical_rows[shared_rows]
        swap_device_physical_rows = shared.swap_device_physical_rows[shared_rows]
        swap_counts = shared.swap_counts[shared_rows]

        # Padded rows are skipped by the kernel (request_state_indices) and must
        # come out as -1 so the attention kernel masks them.
        torch.ops._C_cache_ops.hisparse_resolve_residency(
            host_cache,
            hot.cache,
            self.hot_block_table,
            topk_indices,
            device_topk_rows,
            group.device_global_indices,
            group.lru_slots,
            request_state_indices,
            self.region_stride,
            None,
            group.swap_stats,
            physical_topk_indices,
            hot.attention_block_stride,
            req_id_per_token[:num_tokens].contiguous(),
            block_table,
            block_size,
            None,
            valid_topk_counts,
            swap_host_physical_rows,
            swap_device_physical_rows,
            swap_counts,
            resident.block_table if resident is not None else None,
            resident.view.block_size
            if resident is not None and resident.view is not None
            else 0,
            0,
            self.row_value_bytes or 0,
        )

    def _swap_rows(self, shared_rows: slice) -> None:
        hot = self.hot
        shared = self.index_group.shared_topk
        torch.ops._C_cache_ops.hisparse_gather_compact(
            self.host_cache,
            hot.cache,
            shared.swap_host_physical_rows[shared_rows],
            shared.swap_device_physical_rows[shared_rows],
            shared.swap_counts[shared_rows],
            self.row_value_bytes or 0,
        )

    def _resolve_and_stage_group(
        self,
        *,
        resident: HiSparseCacheHandle,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        logical_topk_indices: torch.Tensor,
        block_size: int,
        return_valid_counts: bool,
        shared_rows: slice,
        attention_indices_out: torch.Tensor | None,
        valid_counts_out: torch.Tensor | None,
    ) -> None:
        group = self.index_group
        compute_stream = current_stream()
        if group.logical_topk_ready is not None:
            group.copy_stream.wait_event(group.logical_topk_ready)
        else:
            group.copy_stream.wait_stream(compute_stream)
        with group.copy_stream:
            self._resolve_residency(
                resident=resident,
                req_id_per_token=req_id_per_token,
                block_table=block_table,
                topk_indices=logical_topk_indices,
                block_size=block_size,
                return_valid_counts=return_valid_counts,
                shared_rows=shared_rows,
                attention_indices_out=attention_indices_out,
                valid_counts_out=valid_counts_out,
            )
            runtimes = [self]
            if resident.decode_batch:
                runtimes.extend(group.followers)
            for runtime in runtimes:
                runtime._swap_rows(shared_rows)
                if runtime._layer_ready_event is None:
                    runtime._layer_ready_event = torch.Event()
                runtime._layer_ready_event.record(group.copy_stream)
                runtime._swap_staged = True

    def _stage_this_layer(self, shared_rows: slice) -> None:
        group = self.index_group
        compute_stream = current_stream()
        group.copy_stream.wait_stream(compute_stream)
        with group.copy_stream:
            self._swap_rows(shared_rows)
            if self._layer_ready_event is None:
                self._layer_ready_event = torch.Event()
            self._layer_ready_event.record(group.copy_stream)
            self._swap_staged = True

    def wait_for_staged_swap(self) -> None:
        if not self._swap_staged or self._layer_ready_event is None:
            raise RuntimeError("HiSparse layer swap was not prefetched.")
        current_stream().wait_event(self._layer_ready_event)
        self._swap_staged = False

    def swap_in(
        self,
        *,
        resident: HiSparseCacheHandle,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        logical_topk_indices: torch.Tensor,
        block_size: int,
        return_valid_counts: bool = False,
        attention_indices_out: torch.Tensor | None = None,
        valid_counts_out: torch.Tensor | None = None,
    ) -> HiSparseTopKResult:
        """Resolve once per group and ensure this layer's rows are available."""
        num_tokens = logical_topk_indices.shape[0]
        shared_rows = self._step_rows(num_tokens)
        group = self.index_group
        hot = self.hot
        assert hot.block_size == block_size
        assert hot.attention_block_stride == group.leader.hot.attention_block_stride

        if self.is_group_leader:
            self._resolve_and_stage_group(
                resident=resident,
                req_id_per_token=req_id_per_token,
                block_table=block_table,
                logical_topk_indices=logical_topk_indices,
                block_size=block_size,
                return_valid_counts=return_valid_counts,
                shared_rows=shared_rows,
                attention_indices_out=attention_indices_out,
                valid_counts_out=valid_counts_out,
            )

        if not self._swap_staged:
            self._stage_this_layer(shared_rows)
        self.wait_for_staged_swap()

        physical_topk_indices = group.shared_topk.physical_topk_indices[shared_rows]
        if return_valid_counts:
            return (
                physical_topk_indices,
                group.shared_topk.valid_topk_counts[shared_rows],
            )
        return physical_topk_indices


class HiSparseCacheHandle:
    """Attention-facing handle for resident KV and sparse offload state."""

    def __init__(self, runtime: HiSparseRuntime) -> None:
        self.view: PagedCacheView | None = None
        self.block_table: torch.Tensor | None = None
        self.source_block_table: torch.Tensor | None = None
        self.slot_mapping: torch.Tensor | None = None
        self.compressed_slot_mapping: torch.Tensor | None = None
        self.logical_block_size = 0
        self.runtime = runtime
        self.dummy_batch = False
        self.decode_batch = False
        self.all_context_pages_resident = True
        self.num_actual_tokens = 0
        self.num_decode_tokens = 0
        self.req_id_per_token: torch.Tensor | None = None
        self.host_mirror_required = False
        self.mirror_from_resident = False
        self.mirror_slot_mapping: torch.Tensor | None = None
        self.mirror_staging_cache: torch.Tensor | None = None
        self.mirror_staging_slots: torch.Tensor | None = None
        self.submit_layer_mirror: Callable[[], None] | None = None
        self.index_group_caches: list[HiSparseCacheHandle] = [self]

    def prepare_group_for_batch(self, attn_metadata: Any | None) -> None:
        assert self.runtime.is_group_leader
        for cache in self.index_group_caches:
            cache._prepare_for_batch(attn_metadata)

    def _prepare_for_batch(self, attn_metadata: Any | None) -> None:
        self.dummy_batch = attn_metadata is None
        self.runtime.begin_forward()
        self.num_actual_tokens = (
            attn_metadata.num_actual_tokens if attn_metadata is not None else 0
        )
        self.num_decode_tokens = (
            attn_metadata.num_decode_tokens if attn_metadata is not None else 0
        )
        self.req_id_per_token = (
            attn_metadata.req_id_per_token if attn_metadata is not None else None
        )
        self.decode_batch = (
            attn_metadata is not None
            and attn_metadata.num_decode_tokens == attn_metadata.num_actual_tokens
            and attn_metadata.max_query_len == 1
            and attn_metadata.num_reqs == attn_metadata.num_actual_tokens
        )
        self.host_mirror_required = attn_metadata is not None and (
            not self.decode_batch or self.runtime.eager_host_mirror
        )

    def write_target(
        self, num_input_rows: int, num_slot_rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        assert self.view is not None and self.slot_mapping is not None
        num_rows = min(num_input_rows, num_slot_rows, self.num_actual_tokens)
        if self.decode_batch:
            num_rows = min(num_rows, self.runtime.max_num_reqs)
        return self.view.cache, self.slot_mapping[:num_rows], num_rows

    def mirror_write_target(
        self, num_rows: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not self.host_mirror_required
            or self.decode_batch
            or self.mirror_from_resident
        ):
            return None
        cache = self.mirror_staging_cache
        slots = self.mirror_staging_slots
        if cache is None or slots is None:
            raise RuntimeError("HiSparse prefill mirror staging is not bound.")
        if num_rows > slots.numel():
            raise RuntimeError(
                "HiSparse prefill mirror exceeds staging capacity: "
                f"{num_rows} > {slots.numel()}."
            )
        return cache, slots[:num_rows]

    def finish_kv_update(self) -> None:
        if self.dummy_batch:
            return
        if (
            self.submit_layer_mirror is not None
            and not self.decode_batch
            and get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL
        ):
            self.submit_layer_mirror()

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

    def swap_in(
        self,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        logical_topk_indices: torch.Tensor,
        *,
        block_size: int,
        return_valid_counts: bool = False,
        attention_indices_out: torch.Tensor | None = None,
        valid_counts_out: torch.Tensor | None = None,
    ) -> HiSparseTopKResult:
        return self.runtime.swap_in(
            resident=self,
            req_id_per_token=req_id_per_token[: logical_topk_indices.shape[0]],
            block_table=block_table,
            logical_topk_indices=logical_topk_indices,
            block_size=block_size,
            return_valid_counts=return_valid_counts,
            attention_indices_out=attention_indices_out,
            valid_counts_out=valid_counts_out,
        )


def create_hisparse_cache_handle(
    vllm_config: VllmConfig,
    model_top_k: int,
    *,
    is_index_group_leader: bool,
    row_width: int,
    kv_dtype: torch.dtype,
    index_group: HiSparseMLAIndexGroup | None = None,
    device: torch.device | str | None = None,
    storage_block_size: int | None = None,
    row_value_bytes: int | None = None,
) -> HiSparseCacheHandle | None:
    config = ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k)
    if config is None:
        return None

    max_num_reqs = vllm_config.scheduler_config.max_num_seqs
    max_swap_rows = _get_max_swap_rows(vllm_config)
    if device is None:
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )

    hisparse_group = (
        None
        if is_index_group_leader or index_group is None
        else index_group.hisparse_group
    )
    runtime = HiSparseRuntime(
        config=config,
        max_num_reqs=max_num_reqs,
        max_swap_rows=max_swap_rows,
        row_width=row_width,
        kv_dtype=kv_dtype,
        device=device,
        storage_block_size=storage_block_size,
        row_value_bytes=row_value_bytes,
        index_group=hisparse_group,
        copy_stream=index_group.side_stream if index_group is not None else None,
        logical_topk_ready=(
            index_group.logical_topk_ready
            if index_group is not None and index_group.has_indexer
            else None
        ),
    )
    if is_index_group_leader and index_group is not None:
        index_group.hisparse_group = runtime.index_group
    logger.info_once(
        "Enabled experimental HiSparse HMA hot cache: top_k=%d, "
        "device_buffer_size=%d (%d LRU rows), max_num_seqs=%d.",
        config.top_k,
        config.device_buffer_size,
        config.device_buffer_size,
        max_num_reqs,
    )
    handle = HiSparseCacheHandle(runtime)
    speculative_config = vllm_config.speculative_config
    handle.mirror_from_resident = bool(
        vllm_config.scheduler_config.async_scheduling
        and speculative_config is not None
        and speculative_config.uses_draft_kv_cache()
    )
    return handle
