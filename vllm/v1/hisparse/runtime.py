# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA HiSparse resident, host, and hot-cache data plane."""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
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


def check_hisparse_host_memory(pool_bytes: int) -> None:
    """Reject a physical host-pool allocation that cannot fit in RAM."""
    mem = psutil.virtual_memory()
    if pool_bytes > mem.available * 0.95:
        raise ValueError(
            f"HiSparse pinned host pool needs ~{pool_bytes / 2**30:.0f} GiB "
            f"but only {mem.available / 2**30:.0f} GiB of RAM is available. "
            "Lower hisparse_config.host_pool_gib or leave headroom for co-tenants."
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


def _create_group_plan(device: torch.device, max_rows: int, top_k: int) -> _GroupPlan:
    return _GroupPlan(device, max_rows, top_k)


def _create_copy_stream(device: torch.device) -> torch.Stream:
    return torch.Stream(device=device)


@dataclass
class HiSparseIndexGroupBuilder:
    """Carry the latest index-producing runtime through model construction."""

    current_group: HiSparseIndexGroup | None = None


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
    ) -> None:
        self.device_global_indices = torch.full(
            (max_num_reqs, region_stride),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.lru_init = torch.arange(region_stride, dtype=torch.int16, device=device)
        self.lru_slots = self.lru_init.repeat(max_num_reqs, 1).contiguous()
        self.plan = _create_group_plan(device, max_swap_rows, top_k)
        self.copy_stream = _create_copy_stream(device)
        self.followers: list[HiSparseRuntime] = []


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
        max_swap_rows: int | None = None,
        index_group: HiSparseIndexGroup | None = None,
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
        # Logical slots per request. Physical rows come from its ephemeral HMA
        # block table and need not be contiguous in the shared slab.
        self.region_stride = config.device_buffer_size

        row_bytes = row_width * kv_dtype.itemsize
        if row_bytes % 16 != 0:
            raise ValueError(
                f"HiSparse requires 16-byte aligned KV rows, got {row_bytes}B."
            )

        self._prefetch_event: torch.Event | None = None
        self.is_group_leader = index_group is None
        if index_group is None:
            index_group = HiSparseIndexGroup(
                self.device,
                max_num_reqs,
                self.region_stride,
                max_swap_rows or max_num_reqs,
                config.top_k,
            )
            index_group.leader = self
        else:
            index_group.followers.append(self)
        self.index_group = index_group

        self.eager_host_mirror = False
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
        self.hot = PagedCacheView.bind(
            raw_tensor,
            dtype=self.kv_dtype,
            row_width=self.row_width,
            byte_offset=byte_offset,
            block_stride=block_stride,
            num_blocks=num_blocks,
            block_size=block_size,
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

        self._host_cache = kv_cache.view(-1, kv_cache.shape[-1])
        self.registered_host_pool = (
            registered_host_pool if registered_host_pool is not None else kv_cache
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
            kv_cache.view(-1, row_width),
            staged,
            plan.row_ids,
            plan.dst_rows,
            plan.miss_mask,
            None,
            None,
            0,
        )
        return staged

    def reset_hot_state(self) -> None:
        """Drop all hot-buffer bookkeeping (hits become misses)."""
        group = self.index_group
        group.device_global_indices.fill_(-1)
        group.lru_slots.copy_(group.lru_init.expand_as(group.lru_slots))

    def invalidate_slots(
        self,
        slots: torch.Tensor,
        request_state_indices: torch.Tensor,
    ) -> None:
        """Drop scheduled requests' hot copies of recycled global slots."""
        device_global_indices = self.index_group.device_global_indices
        slots = slots.to(device=self.device, dtype=torch.int32)
        state_indices = request_state_indices.to(device=self.device, dtype=torch.long)
        active_indices = device_global_indices.index_select(0, state_indices)
        active_indices.masked_fill_(torch.isin(active_indices, slots), -1)
        device_global_indices.index_copy_(0, state_indices, active_indices)

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
        )

    def backup_caches(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the static tensors needed by the all-layer backup plan."""
        return self.hot.cache, self.host_cache

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
        plan_row_offset: int = 0,
        prefetch_followers: bool = True,
    ) -> HiSparseTopKResult:
        """Resolve top-k positions against resident, hot, then host storage."""
        num_tokens = topk_indices.shape[0]
        hot = self.hot
        host_cache = self.host_cache
        assert hot.block_size == block_size
        request_state_indices = self.request_state_indices
        assert request_state_indices is not None
        group = self.index_group
        plan = group.plan

        relative_indices = topk_indices[:num_tokens].contiguous()

        plan_rows = slice(plan_row_offset, plan_row_offset + num_tokens)
        hot_indices = plan.hot_indices[plan_rows]
        valid_counts = plan.valid_counts[plan_rows] if return_valid_counts else None
        if produce_plan:
            compact_miss_globals = plan.miss_global_indices[plan_rows]
            compact_miss_hots = plan.miss_hot_indices[plan_rows]
            compact_miss_counts = plan.miss_counts[plan_rows]
        else:
            compact_miss_globals = None
            compact_miss_hots = None
            compact_miss_counts = None

        attention_indices = plan.attention_indices[plan_rows]

        # Padded rows are skipped by the kernel (request_state_indices) and must
        # come out as -1 so the attention kernel masks them.
        torch.ops._C_cache_ops.hisparse_swap_in(
            host_cache,
            hot.cache,
            self.hot_block_table,
            relative_indices,
            hot_indices,
            group.device_global_indices,
            group.lru_slots,
            request_state_indices,
            self.region_stride,
            None,
            attention_indices,
            hot.attention_block_stride,
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
        )

        if produce_plan and group.followers and prefetch_followers:
            self._prefetch_group(num_tokens, plan_row_offset)

        if not return_valid_counts:
            return hot.attention_cache, attention_indices
        assert valid_counts is not None
        return hot.attention_cache, attention_indices, valid_counts

    def _gather_plan_into(self, num_tokens: int, plan_row_offset: int = 0) -> None:
        hot = self.hot
        plan = self.index_group.plan
        plan_rows = slice(plan_row_offset, plan_row_offset + num_tokens)
        torch.ops._C_cache_ops.hisparse_gather_compact(
            self.host_cache,
            hot.cache,
            plan.miss_global_indices[plan_rows],
            plan.miss_hot_indices[plan_rows],
            plan.miss_counts[plan_rows],
        )

    def _prefetch_group(self, num_tokens: int, plan_row_offset: int = 0) -> None:
        group = self.index_group
        compute = torch.accelerator.current_stream(self.device)
        group.copy_stream.wait_stream(compute)
        with group.copy_stream:
            for follower in group.followers:
                follower._gather_plan_into(num_tokens, plan_row_offset)
                if follower._prefetch_event is None:
                    follower._prefetch_event = torch.Event()
                follower._prefetch_event.record(group.copy_stream)

    def apply_plan(
        self,
        *,
        block_size: int,
        num_tokens: int,
        return_valid_counts: bool = False,
        plan_row_offset: int = 0,
    ) -> HiSparseTopKResult:
        """Replay the leader's swap plan for an index-sharing follower."""
        n = num_tokens
        group = self.index_group
        leader = group.leader
        plan = group.plan
        hot = self.hot
        assert hot.block_size == block_size
        assert hot.attention_block_stride == leader.hot.attention_block_stride
        plan_rows = slice(plan_row_offset, plan_row_offset + n)
        attention_indices = plan.attention_indices[plan_rows]
        if self._prefetch_event is not None:
            torch.accelerator.current_stream(self.device).wait_event(
                self._prefetch_event
            )
            self._prefetch_event = None
        else:
            self._gather_plan_into(num_tokens, plan_row_offset)
        if return_valid_counts:
            return (
                hot.attention_cache,
                attention_indices,
                plan.valid_counts[plan_rows],
            )
        return hot.attention_cache, attention_indices


class HiSparseCacheHandle:
    """Attention-facing handle for resident KV and sparse offload state."""

    def __init__(self, runtime: HiSparseRuntime) -> None:
        self.view: PagedCacheView | None = None
        self.block_table: torch.Tensor | None = None
        self.slot_mapping: torch.Tensor | None = None
        self.runtime = runtime
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
        self.view = PagedCacheView.bind(
            raw_tensor,
            dtype=self.runtime.kv_dtype,
            row_width=self.runtime.row_width,
            byte_offset=byte_offset,
            block_stride=block_stride,
            num_blocks=num_blocks,
            block_size=block_size,
        )
        self.block_table = block_table
        self.slot_mapping = slot_mapping

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
        plan_row_offset: int = 0,
        prefetch_followers: bool = True,
    ) -> HiSparseTopKResult:
        num_tokens = topk_indices.shape[0]
        if not self.runtime.is_group_leader:
            return self.runtime.apply_plan(
                block_size=block_size,
                num_tokens=num_tokens,
                return_valid_counts=return_valid_counts,
                plan_row_offset=plan_row_offset,
            )
        return self.runtime.swap_in(
            resident=self,
            req_id_per_token=req_id_per_token[:num_tokens],
            block_table=block_table,
            topk_indices=topk_indices,
            block_size=block_size,
            return_valid_counts=return_valid_counts,
            produce_plan=bool(self.runtime.index_group.followers),
            plan_row_offset=plan_row_offset,
            prefetch_followers=prefetch_followers,
        )


def create_hisparse_cache_handle(
    vllm_config: VllmConfig,
    model_top_k: int,
    *,
    is_index_group_leader: bool,
    row_width: int,
    kv_dtype: torch.dtype,
    index_group_builder: HiSparseIndexGroupBuilder | None = None,
    device: torch.device | str | None = None,
) -> HiSparseCacheHandle | None:
    config = ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k)
    if config is None:
        return None

    max_num_reqs = vllm_config.scheduler_config.max_num_seqs
    # Each speculative step needs its own replayable plan rows even though the
    # per-request hot state is reused between steps.
    max_decode_query_len = 1
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is not None
        and speculative_config.num_speculative_tokens is not None
    ):
        max_decode_query_len += speculative_config.num_speculative_tokens * (
            2 if speculative_config.parallel_drafting else 1
        )
    max_swap_rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        max_num_reqs * max_decode_query_len,
    )
    if device is None:
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )

    index_group = (
        None
        if is_index_group_leader or index_group_builder is None
        else index_group_builder.current_group
    )
    runtime = HiSparseRuntime(
        config=config,
        max_num_reqs=max_num_reqs,
        max_swap_rows=max_swap_rows,
        row_width=row_width,
        kv_dtype=kv_dtype,
        device=device,
        index_group=index_group,
    )
    if is_index_group_leader and index_group_builder is not None:
        index_group_builder.current_group = runtime.index_group
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
