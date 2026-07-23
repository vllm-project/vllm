# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental HiSparse hot-buffer decode for sparse MLA (GLM-5 / DSA).

Port of SGLang's HiSparse decode design to vLLM's attention-backend boundary.
HiSparse keeps the full-size MLA KV off the GPU and serves decode from a small
per-request device hot buffer; that is what frees GPU memory for higher decode
concurrency.

- Full-size KV residency. The V2 KV allocator places MLA layers in pinned host
  memory; only the GPU indexer pool and the
  per-request hot buffers stay on device. Decode-step writes land in the
  reserved hot slot and are scattered back to the host pool by a CUDA kernel
  writing straight into pinned memory, so the backup is stream-ordered with
  the decode kernels and needs no host synchronization.
- Each batch row owns a fixed per-layer hot region of
  ``region_stride = round_up(device_buffer_size + 1, 128)`` KV rows (padded to
  128 so the flat buffer can be viewed with any kernel page size up to 128).
  Slots ``[0, device_buffer_size)`` are LRU-managed; slot ``device_buffer_size``
  is reserved for the newest token, which the KV-cache update writes there
  directly.
- At decode time, the indexer's top-k positions are converted to global slot
  ids and a swap-in kernel resolves them against the hot region: hits are
  reused, misses are copied from the pinned host pool, and the LRU order is
  updated.
"""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import psutil
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.metrics.stats import HiSparseStats
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor

logger = init_logger(__name__)

# fp8_ds_mla KV row: 512 B quantized NoPE + 16 B scales + 128 B RoPE.
FP8_DS_MLA_ROW_BYTES = 656

# Hot regions are padded to a multiple of this so the flat hot buffer can be
# viewed with any kernel page size up to 128.
HOT_REGION_ALIGN = 128


def is_hisparse_decode_batch(
    *,
    max_query_len: int,
    num_reqs: int,
    num_actual_tokens: int,
) -> bool:
    return max_query_len == 1 and num_reqs == num_actual_tokens


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
        device_buffer_size = config.device_buffer_size
        if device_buffer_size is None:
            device_buffer_size = 2 * model_top_k

        if device_buffer_size < model_top_k:
            raise ValueError(
                "HiSparse device_buffer_size must be at least the model's "
                f"index_topk. Got device_buffer_size={device_buffer_size}, "
                f"index_topk={model_top_k}."
            )
        return cls(
            top_k=model_top_k,
            device_buffer_size=device_buffer_size,
            host_pool_gib=config.host_pool_gib,
        )


_COORDINATORS: list[HiSparseCoordinator] = []
_METRICS_INTERVAL = 2000
_metrics_calls = 0
_metrics_last = HiSparseStats()
_CURRENT_GROUP_LEADER: HiSparseCoordinator | None = None
_PREFILL_REMAP: tuple | None = None


def take_hisparse_stats() -> HiSparseStats | None:
    """Return counter deltas periodically, avoiding per-step synchronization."""
    global _metrics_calls, _metrics_last
    _metrics_calls += 1
    if not _COORDINATORS or _metrics_calls % _METRICS_INTERVAL != 0:
        return None

    current = HiSparseStats()
    for coordinator in _COORDINATORS:
        hits, misses = coordinator._swap_stats.cpu().tolist()
        current.cache_hits += hits
        current.cache_misses += misses
        current.host_to_device_bytes += misses * coordinator.stats_row_bytes

    delta = HiSparseStats(
        cache_hits=current.cache_hits - _metrics_last.cache_hits,
        cache_misses=current.cache_misses - _metrics_last.cache_misses,
        host_to_device_bytes=(
            current.host_to_device_bytes - _metrics_last.host_to_device_bytes
        ),
    )
    _metrics_last = current
    if delta.cache_hits == 0 and delta.cache_misses == 0:
        return None
    return delta


# Persistent pinned staging for per-step invalidated-slot uploads. One shared buffer is
# safe: these uploads run eagerly at batch preparation, never under
# CUDA-graph capture, and the event guards against overwriting bytes a
# previous upload's async copy is still reading.
_PINNED_STAGING: torch.Tensor | None = None
_PINNED_STAGING_EVENT: torch.Event | None = None

# Host ranges the V2 allocator pinned via cudaHostRegister for the
# host-resident pool. torch's Tensor.is_pinned() only recognizes memory from
# its own caching host allocator, so bind_source_cache consults this registry
# for the fail-loud "pool must be pinned" check.
_REGISTERED_HOST_RANGES: list[tuple[int, int]] = []
_PINNED_HOST_POOLS: list[torch.Tensor] = []


def check_hisparse_host_memory(vllm_config: VllmConfig, rank_bytes: int) -> None:
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
    note_registered_host_range(registered.data_ptr(), registered.nbytes)
    _PINNED_HOST_POOLS.append(registered)
    return registered[:size]


def note_registered_host_range(ptr: int, nbytes: int) -> None:
    _REGISTERED_HOST_RANGES.append((ptr, nbytes))


def discard_registered_host_range(ptr: int) -> None:
    global _REGISTERED_HOST_RANGES
    _REGISTERED_HOST_RANGES = [(p, n) for p, n in _REGISTERED_HOST_RANGES if p != ptr]


def _covers_registered_host_range(ptr: int, nbytes: int) -> bool:
    return any(p <= ptr and ptr + nbytes <= p + n for p, n in _REGISTERED_HOST_RANGES)


def release_pinned_state() -> bool:
    """Synchronize, unregister host KV pools, and drop global state."""
    global _CURRENT_GROUP_LEADER, _PINNED_STAGING, _PINNED_STAGING_EVENT
    global _PREFILL_REMAP, _metrics_calls, _metrics_last
    released = bool(_PINNED_HOST_POOLS or _PINNED_STAGING is not None)
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
            return released

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
            discard_registered_host_range(tensor.data_ptr())
            _PINNED_HOST_POOLS.pop()
        if freed_bytes:
            logger.info(
                "HiSparse: unpinned %.1f GiB of host pool in %.1fs.",
                freed_bytes / 2**30,
                time.perf_counter() - release_start,
            )

    _PINNED_STAGING = None
    _PINNED_STAGING_EVENT = None
    for leader in _COORDINATORS:
        for coordinator in (leader, *leader.group_shared):
            if coordinator._host_cache is not None:
                coordinator._host_cache = None
                released = True
    _COORDINATORS.clear()
    _metrics_calls = 0
    _metrics_last = HiSparseStats()
    _GROUP_PLANS.clear()
    _COPY_STREAMS.clear()
    _CURRENT_GROUP_LEADER = None
    _PREFILL_REMAP = None
    with suppress(RuntimeError):
        torch._C._host_emptyCache()
    return released


def _pinned_to_device(values: list[int], device: torch.device) -> torch.Tensor:
    """Copy a small int list to ``device`` via pinned staging (grow-on-demand,
    power-of-2) instead of a per-step pageable tensor."""
    global _PINNED_STAGING, _PINNED_STAGING_EVENT
    n = len(values)
    if _PINNED_STAGING is None or _PINNED_STAGING.shape[0] < n:
        size = 1 << max(10, (n - 1).bit_length())
        _PINNED_STAGING = torch.empty(size, dtype=torch.long, pin_memory=True)
        _PINNED_STAGING_EVENT = None
    if _PINNED_STAGING_EVENT is not None:
        _PINNED_STAGING_EVENT.synchronize()
    staging = _PINNED_STAGING[:n]
    staging.copy_(torch.from_numpy(np.asarray(values, dtype=np.int64)))
    out = staging.to(device, non_blocking=True)
    if _PINNED_STAGING_EVENT is None:
        _PINNED_STAGING_EVENT = torch.Event()
    _PINNED_STAGING_EVENT.record(torch.accelerator.current_stream(device))
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
    for coordinator in _COORDINATORS:
        if slots is None:
            # Built once on device and shared by every leader.
            blocks = _pinned_to_device(block_ids, coordinator.device)
            offsets = torch.arange(
                block_size, dtype=torch.long, device=coordinator.device
            )
            slots = (blocks[:, None] * block_size + offsets[None, :]).flatten()
        coordinator.invalidate_slots(slots)


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


def _has_hisparse_ops() -> bool:
    if not hasattr(torch.ops, "_C_cache_ops"):
        return False
    return (
        hasattr(torch.ops._C_cache_ops, "hisparse_swap_in")
        and hasattr(torch.ops._C_cache_ops, "hisparse_gather_plan")
        and hasattr(torch.ops._C_cache_ops, "hisparse_backup")
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

    __slots__ = ("global_indices", "hot_indices", "miss_mask", "valid_counts")

    def __init__(self, device: torch.device, max_rows: int, top_k: int) -> None:
        self.global_indices = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.hot_indices = torch.full(
            (max_rows, top_k), -1, dtype=torch.int32, device=device
        )
        self.miss_mask = torch.zeros(
            (max_rows, top_k), dtype=torch.int32, device=device
        )
        self.valid_counts: torch.Tensor | None = None


_GROUP_PLANS: dict[tuple[str, int, int], _GroupPlan] = {}


def _get_group_plan(device: torch.device, max_rows: int, top_k: int) -> _GroupPlan:
    key = (str(device), max_rows, top_k)
    plan = _GROUP_PLANS.get(key)
    if plan is None:
        plan = _GroupPlan(device, max_rows, top_k)
        _GROUP_PLANS[key] = plan
    return plan


_COPY_STREAMS: dict[str, torch.Stream] = {}


def _get_copy_stream(device: torch.device) -> torch.Stream:
    key = str(device)
    stream = _COPY_STREAMS.get(key)
    if stream is None:
        stream = torch.Stream(device=device)
        _COPY_STREAMS[key] = stream
    return stream


class HiSparseCoordinator:
    """Per-layer decode-time hot buffer for sparse MLA KV rows.

    The pinned host-resident KV pool is the only full-size store; misses are
    always served from it. Hot-buffer hits are keyed by global KV slot id, so
    correctness relies on one invariant: a recycled slot's stale state is
    dropped before reuse — the KV connector invalidates all blocks
    (re)assigned to incoming requests (covering connector RDMA loads of any
    kind) before any step can select them.
    """

    def __init__(
        self,
        config: ResolvedHiSparseConfig,
        max_num_reqs: int,
        row_width: int,
        kv_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if not _has_hisparse_ops():
            raise RuntimeError(
                "HiSparse requires the compiled _C_cache_ops.hisparse_swap_in/"
                "gather_plan/backup CUDA kernels (host-resident decode has no "
                "Python fallback). Rebuild vLLM from source so "
                "csrc/libtorch_stable/hisparse_kernels.cu is included."
            )
        self.config = config
        self.max_num_reqs = max_num_reqs
        self.row_width = row_width
        self.kv_dtype = kv_dtype
        self.device = torch.device(device)
        # One reserved slot for the newest token. Sparse MLA kernels consume a
        # paged layout, and the page size is only known at swap-in time.
        self.region_stride = round_up(config.device_buffer_size + 1, HOT_REGION_ALIGN)

        row_bytes = row_width * kv_dtype.itemsize
        if row_bytes % 16 != 0:
            raise ValueError(
                f"HiSparse requires 16-byte aligned KV rows, got {row_bytes}B."
            )

        hot_rows = max_num_reqs * self.region_stride
        # Allocated eagerly so vLLM's memory profiling accounts for it when
        # sizing the main KV cache.
        self.hot_cache = torch.zeros(
            (hot_rows, row_width), dtype=kv_dtype, device=self.device
        )
        self.device_global_indices: torch.Tensor | None = torch.full(
            (max_num_reqs, config.device_buffer_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        lru_init = torch.arange(
            config.device_buffer_size, dtype=torch.int16, device=self.device
        )
        self._lru_init: torch.Tensor | None = lru_init
        self.lru_slots: torch.Tensor | None = lru_init.repeat(
            max_num_reqs, 1
        ).contiguous()
        self.request_state_indices = torch.arange(
            max_num_reqs, dtype=torch.int32, device=self.device
        )

        self._swap_stats = torch.zeros(2, dtype=torch.uint64, device=self.device)
        self.stats_row_bytes = row_bytes
        _COORDINATORS.append(self)

        self._plan = _get_group_plan(self.device, max_num_reqs, config.top_k)
        self.group_shared: list[HiSparseCoordinator] = []
        self.leader: HiSparseCoordinator | None = None
        self._prefetch_event: torch.Event | None = None
        self._copy_stream = _get_copy_stream(self.device)

        self._host_cache: torch.Tensor | None = None
        # The host backup runs inline on the current stream: the decode
        # KV-update executes inside the FULL_DECODE_ONLY CUDA graph, where a
        # cross-stream wait raises "dependency created on uncaptured work in
        # another stream" and aborts capture. The per-step backup is only the
        # batch's newest rows, so there is no overlap worth reclaiming.

    def set_request_state_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() > self.max_num_reqs:
            raise ValueError(
                "HiSparse request-state mapping exceeds max_num_seqs: "
                f"{indices.numel()} > {self.max_num_reqs}."
            )
        self.request_state_indices = indices

    def join_indexer_group(self, has_indexer: bool) -> None:
        global _CURRENT_GROUP_LEADER
        if has_indexer:
            _CURRENT_GROUP_LEADER = self
        elif _CURRENT_GROUP_LEADER is not None:
            self.join_group(_CURRENT_GROUP_LEADER)

    def join_group(self, leader: HiSparseCoordinator) -> None:
        self.leader = leader
        leader.group_shared.append(self)
        leader.stats_row_bytes += self.stats_row_bytes
        _COORDINATORS.remove(self)
        self.device_global_indices = None
        self.lru_slots = None
        self._lru_init = None

    def hot_cache_paged(self, block_size: int) -> torch.Tensor:
        """Hot buffer shaped like a regular paged MLA cache."""
        return self.hot_cache.view(-1, block_size, self.row_width)

    def bind_source_cache(self, kv_cache: torch.Tensor) -> None:
        flat = kv_cache.view(-1, kv_cache.shape[-1])
        if self._host_cache is not None and (
            self._host_cache.data_ptr() == flat.data_ptr()
            and self._host_cache.shape == flat.shape
        ):
            return

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

        self._host_cache = flat

    def stage_prefill_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather referenced host context blocks into a compact GPU cache."""
        global _PREFILL_REMAP
        device = block_table.device
        block_size = kv_cache.shape[1]
        row_width = kv_cache.shape[-1]
        key = (block_table.data_ptr(), block_table.shape, block_table._version)
        cached = _PREFILL_REMAP
        if cached is not None and cached[0] == key:
            _, new_bt, row_ids, dst_rows, miss_mask = cached
        else:
            used = (seq_lens.to(torch.int64) + block_size - 1) // block_size
            bounded = torch.where(
                torch.arange(block_table.shape[1], device=device)[None, :]
                < used[:, None],
                block_table,
                0,
            )
            new_bt, row_ids = hisparse_prefill_staging_remap(bounded, block_size)
            dst_rows = torch.arange(
                row_ids.shape[1], dtype=torch.int32, device=device
            ).view(1, -1)
            miss_mask = torch.ones_like(row_ids)
            _PREFILL_REMAP = (key, new_bt, row_ids, dst_rows, miss_mask)

        staged = torch.empty(
            (row_ids.shape[1] // block_size, block_size, row_width),
            dtype=kv_cache.dtype,
            device=device,
        )
        torch.ops._C_cache_ops.hisparse_gather_plan(
            kv_cache.view(-1, row_width),
            staged.view(-1, row_width),
            row_ids,
            dst_rows,
            miss_mask,
            None,
        )
        return staged, new_bt

    def reset_hot_state(self) -> None:
        """Drop all hot-buffer bookkeeping (hits become misses)."""
        if self.device_global_indices is None:
            return
        assert self.lru_slots is not None and self._lru_init is not None
        self.device_global_indices.fill_(-1)
        self.lru_slots.copy_(self._lru_init.expand_as(self.lru_slots))

    def _invalidate_hot_copies(self, slots: torch.Tensor) -> None:
        assert self.device_global_indices is not None
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
        )

    # ------------------------------------------------------- newest-token path

    def write_newest_rows(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Decode-step KV update.

        Writes the newest token of each batch row to its reserved hot slot
        (so the swap-in kernel can serve the newest token without a host
        roundtrip), then scatters it into the host pool at its global KV slot
        (so later steps can miss-load it).
        """
        if kv_cache.numel() == 0:
            return
        self.bind_source_cache(kv_cache)
        # Pad clamp: the forward can run more rows than the scheduler
        # produced (DP alignment pads to a peer's batch, eager/PIECEWISE pads
        # to a capture size) while slot_mapping stays unpadded. Real rows are
        # always a prefix of both, so clamp instead of asserting — a length
        # mismatch trips the backup kernel's shape check and kills the rank
        # (and with it the whole DP fleet).
        num_tokens = min(kv_c_normed.shape[0], slot_mapping.numel(), self.max_num_reqs)
        state_rows = self.request_state_indices[:num_tokens]
        newest_slots = torch.where(
            state_rows >= 0,
            state_rows.to(torch.int64) * self.region_stride
            + self.config.device_buffer_size,
            -1,
        )
        global_slots = slot_mapping[:num_tokens].to(torch.int64)

        # -1-padded slots (CUDA-graph padding) are skipped by the backup
        # kernel.
        ops.concat_and_cache_mla(
            kv_c_normed[:num_tokens],
            k_pe[:num_tokens].squeeze(1),
            self.hot_cache.view(-1, 1, self.row_width),
            newest_slots,
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
        self._backup_rows(
            self.hot_cache,
            newest_slots,
            global_slots,
        )
        # Recycled-slot hygiene is handled at block-assignment time: the
        # The KV connector invalidates every block (re)assigned to any request
        # (new, resumed, or growing) before the step that first writes it,
        # so no per-step in-graph invalidation is needed here.

    def write_rows_to_host(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Write prefill/mixed-batch rows directly to the host pool.

        Local prefill on a decode instance (router shortcut, preemption
        resume, recompute after a failed KV load): quantize the rows on GPU,
        then scatter them to their global host slots via the backup kernel.
        Recycled-slot hygiene is handled at block-assignment time by the
        KV connector lifecycle, so no hot-copy invalidation is needed here.
        """
        if kv_cache.numel() == 0:
            return
        self.bind_source_cache(kv_cache)
        # CUDA graph padding can make kv_c_normed/k_pe longer than slot_mapping.
        # Only rows represented by slot_mapping correspond to real KV writes.
        flat_slots = slot_mapping.flatten()[: kv_c_normed.shape[0]]
        num_rows = flat_slots.numel()
        if num_rows == 0:
            return
        # Fixed shapes only: -1 (padding) slots are carried through and
        # skipped by the backup kernel. Masking them out here
        # (flat_slots[flat_slots >= 0]) would force a device->host sync per
        # layer per mixed step; a few wasted copies of padded rows into the
        # staging tensor are cheaper.
        dst = flat_slots.to(device=self.device, dtype=torch.int64).contiguous()
        real_kv_rows = kv_c_normed[:num_rows]
        real_pe_rows = k_pe[:num_rows]

        if kv_cache_dtype == "fp8_ds_mla":
            rows = torch.empty(
                (num_rows, self.row_width),
                dtype=self.kv_dtype,
                device=self.device,
            )
            ops.concat_and_cache_mla(
                real_kv_rows,
                real_pe_rows.squeeze(1),
                rows.view(-1, 1, self.row_width),
                torch.arange(num_rows, dtype=torch.int64, device=self.device),
                kv_cache_dtype=kv_cache_dtype,
                scale=k_scale,
            )
        else:
            rows = torch.cat(
                [real_kv_rows, real_pe_rows.squeeze(1)], dim=-1
            ).contiguous()
        src = torch.arange(num_rows, dtype=torch.int64, device=self.device)
        self._backup_rows(rows, src, dst)

    # ---------------------------------------------------------------- swap-in

    def swap_in(
        self,
        *,
        kv_cache: torch.Tensor,
        req_id_per_token: torch.Tensor,
        block_table: torch.Tensor,
        topk_indices: torch.Tensor,
        block_size: int,
        slot_mapping: torch.Tensor | None,
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
        top_k = topk_indices.shape[1]
        if self.region_stride % block_size != 0:
            raise ValueError(
                f"HiSparse region_stride {self.region_stride} is not "
                f"divisible by the kernel block_size {block_size}."
            )
        self.bind_source_cache(kv_cache)

        converted = triton_convert_req_index_to_global_index(
            req_id_per_token[:num_tokens],
            block_table,
            topk_indices,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=top_k,
            BLOCK_N=128 if top_k % 128 == 0 else top_k,
            return_valid_counts=return_valid_counts,
        )
        if return_valid_counts:
            global_indices, valid_counts = converted
        else:
            global_indices = converted
            valid_counts = None

        # Newest tokens resolve to the reserved hot slot the KV update wrote
        # this step, keyed by their global slot id from the slot mapping.
        # When slot_mapping is None (mixed batches: the newest rows were
        # written to the host pool, not the reserved hot slots), newest
        # tokens are resolved like any other entry (miss -> host load). That
        # is only visible for THIS layer's compute-stream gather (its own
        # backup ran earlier in the same forward); the overlapped prefetch of
        # the group's shared layers is gated off below, since it would read
        # host rows their backups have not written yet.
        newest_global = (
            slot_mapping[:num_tokens].to(torch.int32).contiguous()
            if slot_mapping is not None
            else None
        )

        if produce_plan:
            self._plan.global_indices[:num_tokens].copy_(global_indices)
            hot_indices = self._plan.hot_indices[:num_tokens]
            miss_mask = self._plan.miss_mask[:num_tokens]
            hot_indices.fill_(-1)
        else:
            hot_indices = torch.full_like(global_indices, -1)
            miss_mask = None

        # Padded rows are skipped by the kernel (request_state_indices) and must
        # come out as -1 so the attention kernel masks them.
        torch.ops._C_cache_ops.hisparse_swap_in(
            self._host_cache,
            self.hot_cache,
            global_indices,
            newest_global,
            hot_indices,
            self.device_global_indices,
            self.lru_slots,
            self.request_state_indices,
            self.region_stride,
            miss_mask,
            self._swap_stats,
        )

        if produce_plan:
            self._plan.valid_counts = valid_counts
            if self.group_shared:
                if slot_mapping is not None:
                    self._prefetch_group(num_tokens)
                else:
                    for shared in self.group_shared:
                        shared._prefetch_event = None

        if valid_counts is None:
            return self.hot_cache_paged(block_size), hot_indices
        return self.hot_cache_paged(block_size), hot_indices, valid_counts

    def _gather_plan_into(self, num_tokens: int) -> None:
        torch.ops._C_cache_ops.hisparse_gather_plan(
            self._host_cache,
            self.hot_cache,
            self._plan.global_indices[:num_tokens],
            self._plan.hot_indices[:num_tokens],
            self._plan.miss_mask[:num_tokens],
            self.request_state_indices,
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
        kv_cache: torch.Tensor,
        block_size: int,
        num_tokens: int,
        return_valid_counts: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        hot_indices = self._plan.hot_indices[:num_tokens]
        if self._prefetch_event is not None:
            torch.accelerator.current_stream(self.device).wait_event(
                self._prefetch_event
            )
            self._prefetch_event = None
        else:
            self.bind_source_cache(kv_cache)
            self._gather_plan_into(num_tokens)
        if return_valid_counts:
            assert self._plan.valid_counts is not None
            return (
                self.hot_cache_paged(block_size),
                hot_indices,
                self._plan.valid_counts[:num_tokens],
            )
        return self.hot_cache_paged(block_size), hot_indices


def create_hisparse_coordinator(
    vllm_config: VllmConfig,
    model_top_k: int,
    *,
    row_width: int,
    kv_dtype: torch.dtype,
    device: torch.device | str | None = None,
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
    )
    hot_bytes = coordinator.hot_cache.numel() * kv_dtype.itemsize
    logger.info_once(
        "Enabled experimental HiSparse sparse MLA hot buffer: top_k=%d, "
        "device_buffer_size=%d (region_stride=%d), host_pool_gib=%s, "
        "%.1f MiB GPU hot buffer per layer (max_num_seqs=%d).",
        config.top_k,
        config.device_buffer_size,
        coordinator.region_stride,
        config.host_pool_gib,
        hot_bytes / 2**20,
        max_num_reqs,
    )
    return coordinator
