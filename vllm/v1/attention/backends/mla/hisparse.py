# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental HiSparse hot-buffer decode for sparse MLA (GLM-5 / DSA).

Port of SGLang's HiSparse decode design to vLLM's attention-backend boundary.
HiSparse keeps the full-size MLA KV off the GPU and serves decode from a small
per-request device hot buffer; that is what frees GPU memory for higher decode
concurrency.

- Full-size KV residency. The MLA KV layers are allocated in pinned host
  memory (see ``_is_hisparse_host_layer`` and ``_allocate_kv_cache_tensors``
  in the cache utils / model runner); only the GPU indexer pool and the
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

from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

logger = init_logger(__name__)

# fp8_ds_mla KV row: 512 B quantized NoPE + 16 B scales + 128 B RoPE.
FP8_DS_MLA_ROW_BYTES = 656

# Hot regions are padded to a multiple of this so the flat hot buffer can be
# viewed with any kernel page size up to 128.
HOT_REGION_ALIGN = 128

# Warn when the estimated hot-buffer footprint exceeds this fraction of total
# GPU memory (model load will likely OOM).
HOT_BUFFER_GPU_WARN_FRACTION = 0.5


def is_hisparse_decode_batch(
    *,
    max_query_len: int,
    num_reqs: int,
    num_actual_tokens: int,
) -> bool:
    return max_query_len == 1 and num_reqs == num_actual_tokens


@dataclass(frozen=True)
class HiSparseConfig:
    top_k: int
    device_buffer_size: int
    host_pool_gib: float

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        model_top_k: int,
    ) -> HiSparseConfig | None:
        raw_config = vllm_config.attention_config.hisparse_config
        if raw_config is None:
            return None

        known_keys = {"device_buffer_size", "host_pool_gib"}
        unknown_keys = set(raw_config) - known_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown hisparse_config keys: {sorted(unknown_keys)}. "
                f"Known keys: {sorted(known_keys)}."
            )

        # Default 2x top_k: at exactly top_k the LRU has zero slack and
        # boundary entries thrash between steps.
        device_buffer_size = int(raw_config.get("device_buffer_size", 2 * model_top_k))
        if raw_config.get("host_pool_gib") is None:
            raise ValueError(
                "HiSparse requires hisparse_config.host_pool_gib: size it as "
                "(usable node RAM - co-tenants) / ranks-per-node."
            )
        host_pool_gib = float(raw_config["host_pool_gib"])

        if device_buffer_size < model_top_k:
            raise ValueError(
                "HiSparse device_buffer_size must be at least the model's "
                f"index_topk. Got device_buffer_size={device_buffer_size}, "
                f"index_topk={model_top_k}."
            )
        if host_pool_gib <= 0:
            raise ValueError("HiSparse host_pool_gib must be positive.")

        return cls(
            top_k=model_top_k,
            device_buffer_size=device_buffer_size,
            host_pool_gib=host_pool_gib,
        )


# One per device, shared by all per-layer coordinators: number of real
# (non-CUDA-graph-padding) requests in the current batch. Updated by the
# model runner outside captured regions; read by the swap-in kernel from
# device memory so graph replays observe the per-step value.
_NUM_REAL_REQS: dict[torch.device, torch.Tensor] = {}


def get_num_real_reqs_tensor(device: torch.device) -> torch.Tensor:
    tensor = _NUM_REAL_REQS.get(device)
    if tensor is None:
        # Default to "all rows are real" so standalone use (tests, eager
        # deployments without the runner hook) processes every row; the model
        # runner overwrites this each step when driving CUDA graphs.
        tensor = torch.full(
            (1,), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device
        )
        _NUM_REAL_REQS[device] = tensor
    return tensor


# Plan-producing coordinators register here for telemetry; index-sharing
# "shared" layers deregister via join_group (their leader's counter, with
# stats_row_bytes summed over the group, covers them).
_STATS: list[HiSparseCoordinator] = []
_STATS_INTERVAL = 2000
_stats_calls = 0
_stats_last = (0, 0, 0)


def _maybe_log_hisparse_stats() -> None:
    """Log aggregate hot-buffer hit rate + PCIe gather volume periodically.

    Counters accumulate in-kernel (graph-safe); this host-side read costs a
    few tiny D2H copies every _STATS_INTERVAL steps.
    """
    global _stats_calls, _stats_last
    _stats_calls += 1
    if not _STATS or _stats_calls % _STATS_INTERVAL != 0:
        return
    hits = misses = gather_bytes = 0
    for c in _STATS:
        h, m = c._swap_stats.cpu().tolist()
        hits += h
        misses += m
        gather_bytes += m * c.stats_row_bytes
    d_hits = hits - _stats_last[0]
    d_misses = misses - _stats_last[1]
    d_bytes = gather_bytes - _stats_last[2]
    _stats_last = (hits, misses, gather_bytes)
    total = d_hits + d_misses
    if total == 0:
        return
    logger.info(
        "HiSparse (last %d steps): hit rate %.1f%%, %.2f GiB gathered "
        "host->device (%d misses).",
        _STATS_INTERVAL,
        100.0 * d_hits / total,
        d_bytes / 2**30,
        d_misses,
    )


def set_num_real_reqs(num_reqs: int) -> None:
    """Called by the model runner before each (real or dummy) forward."""
    for tensor in _NUM_REAL_REQS.values():
        tensor.fill_(num_reqs)
    _maybe_log_hisparse_stats()


def _leader_coordinators(
    static_forward_context: dict,
) -> list[HiSparseCoordinator]:
    """Coordinators owning live hot-buffer state (dgi/lru).

    Index-sharing "shared" layers replay their leader's plan and never write
    dgi/lru, so per-block/per-row state maintenance skips them.
    """
    return [
        coordinator
        for layer in static_forward_context.values()
        if (
            coordinator := getattr(
                getattr(layer, "impl", None), "hisparse_coordinator", None
            )
        )
        is not None
        and coordinator.leader is None
    ]


# Persistent pinned staging for the per-step H2D index uploads below
# (invalidated slots, reset rows). One shared buffer is
# safe: these uploads run eagerly at batch preparation, never under
# CUDA-graph capture, and the event guards against overwriting bytes a
# previous upload's async copy is still reading.
_PINNED_STAGING: torch.Tensor | None = None
_PINNED_STAGING_EVENT: torch.Event | None = None

# Host ranges the model runner pinned via cudaHostRegister for the
# host-resident pool. torch's Tensor.is_pinned() only recognizes memory from
# its own caching host allocator, so bind_source_cache consults this registry
# for the fail-loud "pool must be pinned" check.
_REGISTERED_HOST_RANGES: list[tuple[int, int]] = []


def note_registered_host_range(ptr: int, nbytes: int) -> None:
    _REGISTERED_HOST_RANGES.append((ptr, nbytes))


def discard_registered_host_range(ptr: int) -> None:
    global _REGISTERED_HOST_RANGES
    _REGISTERED_HOST_RANGES = [(p, n) for p, n in _REGISTERED_HOST_RANGES if p != ptr]


def _covers_registered_host_range(ptr: int, nbytes: int) -> bool:
    return any(p <= ptr and ptr + nbytes <= p + n for p, n in _REGISTERED_HOST_RANGES)


def release_pinned_state() -> bool:
    """Drop every module/coordinator reference to pinned host memory.

    Graceful teardown (drain-then-free): the runner unpins the host pool
    itself (cudaHostUnregister); this drops the module-level pinned staging
    buffer and every coordinator's pool reference so the caller can return
    the allocator-owned staging blocks to the OS. Shared (index-sharing)
    coordinators are removed from _STATS by join_group, so they are reached
    through their leader's group_shared list. Returns whether any pinned
    reference was held.
    """
    global _PINNED_STAGING, _PINNED_STAGING_EVENT
    released = _PINNED_STAGING is not None
    _PINNED_STAGING = None
    _PINNED_STAGING_EVENT = None
    for leader in _STATS:
        for coordinator in (leader, *leader.group_shared):
            if coordinator._host_cache is not None:
                coordinator._host_cache = None
                released = True
    _STATS.clear()
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


def invalidate_blocks(
    static_forward_context: dict, block_ids: list[int], block_size: int
) -> None:
    """Drop cached HiSparse state for the given blocks in every layer.

    Called by the model runner when blocks are (re)assigned to newly scheduled
    or preemption-resumed requests, before any forward step can select them.
    This makes block recycling safe for any writer (local prefill, connector
    RDMA into host memory) without per-connector reporting hooks.
    """
    if not block_ids:
        return
    slots: torch.Tensor | None = None
    for coordinator in _leader_coordinators(static_forward_context):
        if slots is None:
            # Built once on device and shared by every leader.
            blocks = _pinned_to_device(block_ids, coordinator.device)
            offsets = torch.arange(
                block_size, dtype=torch.long, device=coordinator.device
            )
            slots = (blocks[:, None] * block_size + offsets[None, :]).flatten()
        coordinator.invalidate_slots(slots)


def reset_rows(static_forward_context: dict, row_ids: list[int]) -> None:
    """Drop per-row HiSparse hot-buffer state in every layer."""
    if not row_ids:
        return
    rows: torch.Tensor | None = None
    for coordinator in _leader_coordinators(static_forward_context):
        if rows is None:
            rows = _pinned_to_device(row_ids, coordinator.device)
        coordinator.reset_hot_rows(rows)


def _has_hisparse_ops() -> bool:
    try:
        try:
            import vllm._C_stable_libtorch  # noqa: F401
        except ImportError:
            import vllm._C  # noqa: F401
    except ImportError:
        return False
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
        # Populated only when the producer resolves with return_valid_counts
        # (the bf16 path); shared layers reuse it (identical top-k -> identical
        # counts). None until first produced.
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
    """One dedicated copy stream per device, shared across a group's coordinators
    for overlapped prefetch. Capturing a fork/join around work on it into a
    FULL_DECODE_ONLY graph is supported."""
    key = str(device)
    s = _COPY_STREAMS.get(key)
    if s is None:
        s = torch.Stream(device=device)
        _COPY_STREAMS[key] = s
    return s


class HiSparseCoordinator:
    """Per-layer decode-time hot buffer for sparse MLA KV rows.

    The pinned host-resident KV pool is the only full-size store; misses are
    always served from it. Hot-buffer hits are keyed by global KV slot id, so
    correctness relies on one invariant: a recycled slot's stale state is
    dropped before reuse — the model runner invalidates all blocks
    (re)assigned to incoming requests (covering connector RDMA loads of any
    kind) before any step can select them.
    """

    def __init__(
        self,
        config: HiSparseConfig,
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
        # One reserved slot for the newest token; the FlashMLA FP8 sparse
        # kernel needs a paged layout and the actual block size is only known
        # at swap-in time, hence the alignment padding.
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
        # Per-request LRU state; released in join_group for index-sharing
        # "shared" layers, which replay their leader's plan and never resolve
        # the LRU themselves.
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
        self._newest_hot_slots = (
            torch.arange(max_num_reqs, dtype=torch.int64, device=self.device)
            * self.region_stride
            + config.device_buffer_size
        )

        self.num_real_reqs = get_num_real_reqs_tensor(self.device)

        # In-kernel hit/miss counters (telemetry). stats_row_bytes converts
        # misses to gathered bytes; plan-once wiring adds each shared
        # layer's row bytes to its leader (the shared layers re-gather the
        # leader's misses), so the leader's counter covers the whole group.
        self._swap_stats = torch.zeros(2, dtype=torch.uint64, device=self.device)
        self.stats_row_bytes = row_bytes
        _STATS.append(self)

        # Group-shared plan buffers for index-sharing plan-once (see _GroupPlan).
        self._plan = _get_group_plan(self.device, max_num_reqs, config.top_k)

        # Overlapped prefetch: a "full" layer issues its group's "shared"
        # gathers early on a copy stream, overlapping the PCIe transfer with
        # intervening compute (measured +24% / +41% end-to-end decode
        # throughput at c=32 / c=96 on GLM-5.2-FP8 vs inline gathers).
        # group_shared/leader are wired via join_group; _prefetch_event is
        # recorded on the copy stream by the leader and awaited in the shared
        # layer's apply_plan.
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

    def join_group(self, leader: HiSparseCoordinator) -> None:
        """Attach to an index-sharing group as a "shared" layer.

        Shared layers replay the leader's plan: they never resolve the LRU
        (their per-request state is released) and never produce stats (their
        re-gathered bytes are covered by the leader's miss counter).
        """
        self.leader = leader
        leader.group_shared.append(self)
        leader.stats_row_bytes += self.stats_row_bytes
        _STATS.remove(self)
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
        # runner explicitly registered.
        if not (
            kv_cache.is_pinned()
            or _covers_registered_host_range(kv_cache.data_ptr(), kv_cache.nbytes)
        ):
            raise ValueError("HiSparse host-resident KV pool must be pinned memory.")

        self._host_cache = flat
        self.reset_hot_state()

    def reset_hot_state(self) -> None:
        """Drop all hot-buffer bookkeeping (hits become misses)."""
        if self.device_global_indices is None:
            # Shared layer: the leader owns the group's LRU state.
            return
        assert self.lru_slots is not None and self._lru_init is not None
        self.device_global_indices.fill_(-1)
        self.lru_slots.copy_(self._lru_init.expand_as(self.lru_slots))

    def reset_hot_rows(self, rows: torch.Tensor) -> None:
        """Drop hot-buffer bookkeeping for newly assigned batch rows."""
        assert self.device_global_indices is not None
        assert self.lru_slots is not None and self._lru_init is not None
        self.device_global_indices[rows] = -1
        self.lru_slots[rows] = self._lru_init

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
        from vllm import _custom_ops as ops

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
        newest_slots = self._newest_hot_slots[:num_tokens]
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
        # model runner invalidates every block (re)assigned to any request
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
        model runner, so no hot-copy invalidation is needed here.
        """
        from vllm import _custom_ops as ops

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

        When ``produce_plan`` (an index-sharing "full" layer), the resolved plan
        (global ids, hot slots, miss mask) is also written to the group-shared
        buffers so the group's "shared" layers can replay it via ``apply_plan``
        without re-resolving the LRU.
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

        # For a plan producer, resolve into the group-shared buffers so the
        # group's shared layers can replay the identical plan.
        if produce_plan:
            self._plan.global_indices[:num_tokens].copy_(global_indices)
            hot_indices = self._plan.hot_indices[:num_tokens]
            miss_mask = self._plan.miss_mask[:num_tokens]
            # The kernel skips CUDA-graph padding rows, so their entries in
            # the persistent plan buffer would otherwise keep stale values;
            # refill so padded rows come out -1 (masked by attention).
            hot_indices.fill_(-1)
        else:
            hot_indices = torch.full_like(global_indices, -1)
            miss_mask = None

        # Padded rows are skipped by the kernel (num_real_reqs) and must
        # come out as -1 so the attention kernel masks them.
        torch.ops._C_cache_ops.hisparse_swap_in(
            self._host_cache,
            self.hot_cache,
            global_indices,
            newest_global,
            hot_indices,
            self.device_global_indices,
            self.lru_slots,
            self.num_real_reqs,
            self.region_stride,
            miss_mask,
            self._swap_stats,
        )

        if produce_plan:
            # Shared layers reuse the group's counts (identical top-k).
            self._plan.valid_counts = valid_counts
            # Overlapped prefetch: issue the group's shared gathers early on
            # the copy stream so their PCIe transfer overlaps intervening
            # compute. Pure decode only (slot_mapping present): with
            # slot_mapping=None the newest tokens are in the miss set, served
            # from host rows the shared layers' own backups have not written
            # yet when the copy stream forks. apply_plan then gathers inline
            # on the compute stream, which IS ordered after each backup.
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
        """Gather THIS coord's misses per the group-shared plan into its own
        hot cache, from the host pool, on the current stream."""
        torch.ops._C_cache_ops.hisparse_gather_plan(
            self._host_cache,
            self.hot_cache,
            self._plan.global_indices[:num_tokens],
            self._plan.hot_indices[:num_tokens],
            self._plan.miss_mask[:num_tokens],
            self.num_real_reqs,
        )

    def _prefetch_group(self, num_tokens: int) -> None:
        """Leader-side: fork the copy stream from the compute stream (plan is
        ready) and issue each shared layer's gather on it, recording a per-shared
        event the shared layer's apply_plan awaits. Fork here + wait_event there
        is captured into the decode graph."""
        compute = torch.accelerator.current_stream(self.device)
        self._copy_stream.wait_stream(compute)  # fork
        with self._copy_stream:
            for shared in self.group_shared:
                # Need the host pool bound to serve misses. Until then, the
                # shared layer gathers inline in apply_plan.
                if shared._host_cache is None:
                    shared._prefetch_event = None
                    continue
                shared._gather_plan_into(num_tokens)
                if shared._prefetch_event is None:
                    shared._prefetch_event = torch.Event()
                shared._prefetch_event.record(self._copy_stream)

    # ---------------------------------------------------------- plan replay

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
        """Replay the group's plan for an index-sharing "shared" layer.

        A "full" layer resolved the plan via ``swap_in(produce_plan=True)``;
        this gathers only THIS layer's own missed rows into the identical
        planned hot slots, with no LRU resolution. Fixed shape -> capture-safe.
        """
        n = num_tokens
        hot_indices = self._plan.hot_indices[:n]
        if self._prefetch_event is not None:
            # Leader already gathered this layer's misses on the copy stream;
            # just join so attention sees them, then consume the event so the
            # next step re-prefetches (or falls back if it can't).
            torch.accelerator.current_stream(self.device).wait_event(
                self._prefetch_event
            )
            self._prefetch_event = None
        else:
            # Not prefetched (mixed batch, or host pool not yet bound): gather
            # this layer's misses inline, same path the leader's prefetch uses.
            self.bind_source_cache(kv_cache)
            self._gather_plan_into(n)
        if return_valid_counts:
            assert self._plan.valid_counts is not None, (
                "apply_plan(return_valid_counts=True) requires the group's full "
                "layer to have produced the plan with counts (bf16 path)."
            )
            return (
                self.hot_cache_paged(block_size),
                hot_indices,
                self._plan.valid_counts[:n],
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
    config = HiSparseConfig.from_vllm_config(vllm_config, model_top_k)
    if config is None:
        return None

    # HiSparse targets PD decode instances, where KV arrives via a consumer
    # connector (NIXL) into the host pool. Local prefill (preemption resume,
    # recompute after a failed KV load) is supported but stages context
    # host->GPU, so it is slower than a normal GPU prefill.
    hf_config = vllm_config.model_config.hf_config
    if not hasattr(hf_config, "index_topk"):
        raise ValueError("HiSparse is only supported for DSA models with index_topk.")
    if hasattr(hf_config, "compress_ratios") and len(hf_config.compress_ratios) > 0:
        raise ValueError("This HiSparse path targets GLM-5/DSA, not DeepSeek V4.")

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
    # Hot buffers are allocated eagerly per layer and scale with
    # max_num_seqs; a too-large product otherwise surfaces as an unrelated
    # CUDA OOM at model load (DeepEP buffers, weights, ...).
    num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    total_gpu = current_platform.get_device_total_memory(coordinator.device.index or 0)
    if num_layers * hot_bytes > HOT_BUFFER_GPU_WARN_FRACTION * total_gpu:
        logger.warning_once(
            "HiSparse hot buffers need %.1f GiB of GPU memory (%d layers x "
            "max_num_seqs=%d x device_buffer_size=%d). This usually OOMs at "
            "model load; lower max_num_seqs on decode instances.",
            num_layers * hot_bytes / 2**30,
            num_layers,
            max_num_reqs,
            config.device_buffer_size,
        )
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
