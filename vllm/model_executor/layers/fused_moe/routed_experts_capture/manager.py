# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side routed-experts slot/offload buffer and block mapping.

Holds the per-token routing IDs the worker captured, indexed by physical KV
block slot, and mirrors them to the offloaded-block buffer along the KV
transfer jobs. Torch-free (numpy only): runs in the scheduler process.
"""

from __future__ import annotations

import contextlib
import logging
import mmap
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    get_num_experts,
    get_num_experts_per_tok,
    require_full_attention_gid,
    routing_slot_shape_dtype,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    SharedRoutingRegion,
    shared_routing_mmap_path,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = logging.getLogger(__name__)


class FullAttnBlockMap(NamedTuple):
    """GPU-block -> offloaded-block sub-block mapping for one transfer job.

    Restricted to the full-attention anchor group (the only group whose
    routing is mirrored). All three arrays have the same length ``G`` =
    ``group_sizes[attn_gid]`` (number of GPU blocks the job moves for that
    group), so ``cpu_blocks[cpu_block_ids[i], sub_offsets[i]]`` is the
    offloaded sub-block slot holding GPU block ``gpu_block_ids[i]``.
    """

    gpu_block_ids: np.ndarray  # GPU block id per moved block
    cpu_block_ids: np.ndarray  # offloaded block id holding that block
    sub_offsets: np.ndarray  # sub-block index within the offloaded block

    @classmethod
    def concatenate(cls, maps: list[FullAttnBlockMap]) -> FullAttnBlockMap:
        """Merge per-job maps so a step's offload transfers apply in ONE
        vectorized fancy-index instead of one numpy call per job. The moved
        data is identical; this only removes per-job call overhead on the
        scheduler thread. ``maps`` must be non-empty (callers drop empty maps).
        """
        return cls(
            gpu_block_ids=np.concatenate([m.gpu_block_ids for m in maps]),
            cpu_block_ids=np.concatenate([m.cpu_block_ids for m in maps]),
            sub_offsets=np.concatenate([m.sub_offsets for m in maps]),
        )


def _cdiv(a: int, b: int) -> int:
    """Ceiling division of non-negative integers."""
    return -(-a // b)


def compute_full_attn_block_map(
    gpu_block_ids: np.ndarray,
    cpu_block_ids: np.ndarray,
    group_sizes: Sequence[int],
    block_indices: Sequence[int],
    attn_gid: int,
    block_size_factor: int,
    expected_num_groups: int | None = None,
) -> FullAttnBlockMap:
    """Slice a transfer job to the full-attention group, mapping each GPU
    block to its offloaded sub-block slot, and validate the layout contract.

    Routing-only analogue of the worker KV copy
    (``kv_offload/cpu/gpu_worker.py``: ``transfer_async`` /
    ``compute_sub_block_ptrs``); it MUST use the same group-major layout and
    sub-block arithmetic so routing tracks the KV bytes block-for-block:

      - GPU block ids are group-major: ``group_sizes[g]`` blocks per group
        (``sum(group_sizes) == len(gpu_block_ids)``).
      - With ``factor`` GPU blocks per offloaded block, group ``g``'s first
        block may be unaligned by ``block_indices[g] % factor`` sub-blocks;
        that skip folds into the sub-block index.
      - CPU block ids are group-major too, per-group length
        ``cdiv(group_sizes[g] + block_indices[g] % factor, factor)`` — the
        worker's ``src/dst_blocks_count``.

    ``factor == 1`` is the legacy 1:1 mapping (skip = 0, sub = 0).

    Args:
        gpu_block_ids: group-major GPU block ids for the whole job.
        cpu_block_ids: group-major offloaded block ids for the whole job.
        group_sizes: GPU block count per KV cache group.
        block_indices: logical block index (in GPU blocks) of each group's
            first block, used to recover the sub-block skip.
        attn_gid: the full-attention anchor group index.
        block_size_factor: GPU blocks per offloaded block (``factor >= 1``).
        expected_num_groups: if set, the KV-group count the job must span;
            mismatch signals a contract break.

    Returns:
        A ``FullAttnBlockMap`` covering only the anchor group.

    Raises:
        RuntimeError: if the group-major flat-order contract is violated.
    """
    factor = block_size_factor
    # Per-group offloaded-block counts: cdiv(group_size + skip, factor),
    # matching the worker's src/dst_blocks_count. Computed once and reused
    # for both contract validation and the anchor group's CPU offset.
    cpu_counts = [
        _cdiv(int(gs) + int(block_indices[g]) % factor, factor)
        for g, gs in enumerate(group_sizes)
    ]
    if (
        (expected_num_groups is not None and len(group_sizes) != expected_num_groups)
        or sum(group_sizes) != len(gpu_block_ids)
        or sum(cpu_counts) != len(cpu_block_ids)
    ):
        raise RuntimeError(
            "routed-experts offload transfer violates the group-major "
            f"flat-order contract: group_sizes={list(group_sizes)}, "
            f"block_indices={list(block_indices)}, attn_gid={attn_gid}, "
            f"factor={factor}, len(gpu)={len(gpu_block_ids)}, "
            f"len(cpu)={len(cpu_block_ids)}, expected_cpu={sum(cpu_counts)}, "
            f"expected_num_groups={expected_num_groups}"
        )

    # GPU offset: anchor group's GPU blocks start after prior groups'.
    gpu_off = int(sum(group_sizes[:attn_gid]))
    n = int(group_sizes[attn_gid])
    gpu_local = np.asarray(gpu_block_ids[gpu_off : gpu_off + n])

    if n == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return FullAttnBlockMap(empty_i, empty_i, empty_i.copy())

    # CPU offset: prior groups consume their cdiv counts.
    cpu_off = sum(cpu_counts[:attn_gid])
    skip = int(block_indices[attn_gid]) % factor
    p = np.arange(n, dtype=np.int64)
    sub_offsets = (skip + p) % factor
    cpu_local_idx = (skip + p) // factor
    cpu_local = np.asarray(cpu_block_ids)[cpu_off + cpu_local_idx]
    return FullAttnBlockMap(gpu_local, cpu_local, sub_offsets)


def _mmap_zeroed(shape: tuple[int, ...], dtype: npt.DTypeLike) -> np.ndarray:
    """Allocate a zero-initialized ndarray backed by an anonymous mmap.

    Unlike ``np.zeros`` (eager heap allocation that commits every page up
    front), an anonymous ``MAP_PRIVATE | MAP_ANONYMOUS`` mapping is demand-paged
    by the kernel: pages are zero-filled and faulted in only on first touch,
    and can be reclaimed under memory pressure. For routing buffers sized to
    the whole block pool (multiple GB, and tens of GB at 1M-token contexts)
    this avoids committing physical RAM for slots that are never written.

    The returned ndarray keeps the mmap alive via its ``base`` chain, so the
    backing memory is released when the array is GC'd. Scheduler-private (no
    ``MAP_SHARED``): only the scheduler process touches these buffers.
    """
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if nbytes == 0:
        return np.zeros(shape, dtype=dtype)
    buf = mmap.mmap(-1, nbytes)  # anonymous, demand-paged, zero-filled
    return np.frombuffer(buf, dtype=dtype).reshape(shape)


class RoutedExpertsManager:
    """Scheduler-side slot-indexed buffer for routed experts.

    Lives on CPU in the scheduler process. Each slot corresponds to
    ``block_id * block_size + offset_in_block`` where ``block_id`` is
    drawn from the physical KV-cache block pool, so routing data is
    tied to physical blocks and naturally survives preemption for
    prefix-cached blocks (prefix hits re-expose the same slots).

    Data flow per step:
      1. Worker (output_rank) scatters this step's captured routing
         directly into the /dev/shm slot buffer (``scatter``), and
         returns only the per-token ``slot_mapping`` via
         ``ModelRunnerOutput.routed_experts_slots`` — no routing payload
         is sent. The returned output is the happens-before barrier.
      2. On each step the scheduler reads decode tokens' routing back
         from the shared buffer by those returned slots (``get_by_slots``);
         on request completion / abort / preemption it calls ``get`` with
         the request's block IDs to recover the full per-token routing.

    Memory: ``routed_experts_by_slot`` is sized for the whole block pool
    (``num_blocks * block_size`` slots), which can reach multiple GB; see the
    init log for the exact size.

    KV offload (``num_offload_blocks`` set): ``routed_experts_by_cpu_block``
    holds offloaded routing indexed by offloaded (CPU) block id, with an extra
    sub-block axis ``(num_offload_blocks, factor, block_size, layers, top_k)``
    since one offloaded block packs ``block_size_factor`` GPU blocks. The
    store/load helpers replay each transfer job's full-attention group with
    the SAME sub-block arithmetic as the worker KV copy
    (``kv_offload/cpu/gpu_worker.py``), so routing shares the KV blocks'
    lifecycle for any ``factor``. Stale rows need no zeroing: only block ids
    the connector loads are read, and reuse overwrites on the next store.

    Disk / multi-tier offload: with a ``TieringOffloadingManager``, a
    ``RoutedExpertsBlockLifecycleObserver`` follows the same cascade
    (CPU->secondary) / promotion (secondary->CPU) events as the KV blocks,
    persisting / restoring rows through a ``RoutedExpertsSecondaryStore`` so
    routing survives CPU eviction exactly as the KV bytes do.

    Stored values are LOGICAL expert IDs (pre-EPLB mapping). Each DP rank owns
    an independent buffer for its own requests; cross-DP migration is
    unsupported.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        num_offload_blocks: int | None = None,
        block_size_factor: int = 1,
    ) -> None:
        # Same gate the worker model runner uses, so both anchor on (and agree
        # on) the same group; raises with no full-attention group, warns on a
        # multi-group hybrid.
        self.attn_gid = require_full_attention_gid(kv_cache_config)
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size
        if block_size_factor < 1:
            raise ValueError(f"block_size_factor must be >= 1, got {block_size_factor}")
        self.block_size_factor = block_size_factor

        # All kv_cache_groups share the same physical block pool, so
        # block IDs span [0, num_blocks) regardless of how many groups
        # exist. Sizing to the full pool avoids index-out-of-range
        # when different groups happen to land on the same block.
        hf_config = vllm_config.model_config.hf_text_config
        num_experts = get_num_experts(hf_config)
        num_experts_per_tok = get_num_experts_per_tok(hf_config)
        self.num_layers = hf_config.num_hidden_layers
        self.num_experts_per_tok = num_experts_per_tok
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        # Expert IDs are 0..num_experts-1; uint8 fits 256 distinct
        # values so the boundary is ``<= 256`` (NOT ``< 256``). Keeping
        # this narrow matters because the slot buffer is sized for the
        # whole block pool and can reach multiple GB.
        expert_id_dtype = np.uint8 if num_experts <= 256 else np.uint16
        self.expert_id_dtype = expert_id_dtype
        # Slot buffer is shared scheduler<->worker via /dev/shm: the worker
        # (output_rank) writes this step's routing directly, the scheduler
        # reads it back per request. Scheduler is the creator. Backing changes
        # from anonymous MAP_PRIVATE to named MAP_SHARED; the flat
        # (max_num_slots, layers, top_k) layout and all store_batch/get/
        # _blocks_view semantics below are unchanged. Demand-paged either way:
        # untouched slots never fault in physical RAM.
        # slot_shape/dtype from the shared helper so the worker writer derives
        # an IDENTICAL layout (they must match or the shared mmap mismatches).
        slot_shape, slot_dtype = routing_slot_shape_dtype(vllm_config, kv_cache_config)
        expected_shape = (max_num_slots, self.num_layers, num_experts_per_tok)
        if slot_shape != expected_shape:
            raise RuntimeError(
                "routed-experts slot buffer layout mismatch: "
                f"routing_slot_shape_dtype gave {slot_shape}, manager sized "
                f"{expected_shape}. The worker writer derives its mmap from the "
                "same helper, so a divergence here means the shared /dev/shm "
                "buffer would be misinterpreted across processes."
            )
        self._slot_region = SharedRoutingRegion(
            path=shared_routing_mmap_path(
                vllm_config.instance_id,
                vllm_config.parallel_config.data_parallel_rank,
            ),
            shape=slot_shape,
            dtype=slot_dtype,
        )
        self.routed_experts_by_slot = self._slot_region.array
        # Block-major view over the slot buffer for whole-block copies
        # (C-contiguous reshape, zero-copy).
        self._blocks_view = self.routed_experts_by_slot.reshape(
            kv_cache_config.num_blocks,
            self.block_size,
            self.num_layers,
            num_experts_per_tok,
        )
        # Indexed by offloaded (CPU) block id. The sub-block axis
        # (``factor``) holds the ``block_size_factor`` GPU blocks packed
        # into one offloaded block; rows are overwritten when the
        # OffloadingManager reuses an offloaded block id.
        self.routed_experts_by_cpu_block: np.ndarray | None = None
        if num_offload_blocks is not None:
            self.routed_experts_by_cpu_block = _mmap_zeroed(
                (
                    num_offload_blocks,
                    self.block_size_factor,
                    self.block_size,
                    self.num_layers,
                    num_experts_per_tok,
                ),
                dtype=expert_id_dtype,
            )
        logger.info(
            "RoutedExpertsManager CPU buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s), "
            "offloaded routed experts: %.2f GB "
            "(cpu_blocks=%s, block_size_factor=%d)",
            self.routed_experts_by_slot.nbytes / 1e9,
            max_num_slots,
            self.num_layers,
            num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
            self.routed_experts_by_cpu_block.nbytes / 1e9
            if self.routed_experts_by_cpu_block is not None
            else 0.0,
            num_offload_blocks,
            self.block_size_factor,
        )

    def store_batch(self, data: np.ndarray, slot_mapping: np.ndarray) -> None:
        """Persist one step's routed experts into the slot buffer.

        Equivalent to ``slot_buffer[slot_mapping] = data``; numpy fancy
        indexing handles repeated / out-of-order indices. The worker writes
        the live slot buffer directly via ``scatter`` (shared mmap), so this
        scheduler-side writer is now only exercised by unit tests.
        """
        self.routed_experts_by_slot[slot_mapping] = data

    def shutdown(self) -> None:
        """Release the shared slot mmap (creator unlinks). Idempotent."""
        region = getattr(self, "_slot_region", None)
        if region is not None:
            # Drop the ndarray view before closing the mmap it is backed by.
            self.routed_experts_by_slot = None  # type: ignore[assignment]
            self._blocks_view = None  # type: ignore[assignment]
            region.close()
            self._slot_region = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()

    def _cpu_blocks(self) -> np.ndarray:
        """Return the offloaded-block buffer, or raise if never allocated.

        The three ``FullAttnBlockMap`` arrays are equal-length by
        construction (see ``compute_full_attn_block_map``), so the only thing
        to guard here is that an offload transfer is not observed when the
        offload buffer is absent (single-tier-without-offload misconfig).
        """
        if self.routed_experts_by_cpu_block is None:
            raise RuntimeError(
                "routed-experts offload buffer is not initialized "
                "but a KV offload transfer was observed"
            )
        return self.routed_experts_by_cpu_block

    def store_to_offload_blocks(self, block_map: FullAttnBlockMap) -> None:
        """Store routed experts GPU slots -> offloaded sub-block rows.

        Must run after ``store_batch`` of the step whose
        scheduler_output carried the store job, so the GPU slots hold
        the routing of every token covered by the job. ``block_map`` is
        the full-attention anchor group's GPU->offloaded-sub-block
        mapping (see ``compute_full_attn_block_map``).
        """
        cpu_blocks = self._cpu_blocks()
        if len(block_map.gpu_block_ids) == 0:
            return
        # cpu_blocks[cpu_id, sub] = blocks_view[gpu_id], vectorized over the
        # job's blocks. factor==1 -> sub is all-zero (legacy 1:1 copy).
        cpu_blocks[block_map.cpu_block_ids, block_map.sub_offsets] = self._blocks_view[
            block_map.gpu_block_ids
        ]

    def load_from_offload_blocks(self, block_map: FullAttnBlockMap) -> None:
        """Load offloaded sub-block rows -> GPU slots for one load job.

        Loaded tokens never re-run a forward pass, so this is the only
        writer of their slots until the blocks are freed. ``block_map``
        is the anchor group's GPU<-offloaded-sub-block mapping.
        """
        cpu_blocks = self._cpu_blocks()
        if len(block_map.gpu_block_ids) == 0:
            return
        self._blocks_view[block_map.gpu_block_ids] = cpu_blocks[
            block_map.cpu_block_ids, block_map.sub_offsets
        ]

    def read_cpu_blocks(self, cpu_block_ids: np.ndarray) -> np.ndarray:
        """Copy whole offloaded-block rows (CPU primary -> secondary).

        Returns a contiguous ``(len(cpu_block_ids), factor, block_size,
        layers, top_k)`` copy, safe to hand to a secondary store. Used by
        the disk/object tiering observer when KV blocks cascade out of the
        CPU primary tier.
        """
        return self._cpu_blocks()[cpu_block_ids]

    def write_cpu_blocks(self, cpu_block_ids: np.ndarray, rows: np.ndarray) -> None:
        """Write whole offloaded-block rows (secondary -> CPU primary).

        Inverse of ``read_cpu_blocks``; used when KV blocks are
        promoted from a secondary tier back into the CPU primary tier.
        """
        self._cpu_blocks()[cpu_block_ids] = rows

    def get(
        self,
        block_ids: list[int],
        num_tokens: int,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read routed experts data for a token range of a request.

        Reconstructs a per-token slot_mapping from the request's block
        IDs and returns the routing slice. Used for the full prompt routing
        at prefill completion and on request completion / abort / preemption.
        Because numpy fancy indexing returns a **copy** (not a view), the
        returned ndarray is safe to hold across subsequent worker ``scatter``
        writes — do not replace the fancy index with a slice without
        re-verifying.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of tokens that have gone through a forward
                pass and therefore have routing data written to their
                slots (typically ``request.num_tokens - 1``; the last
                sampled token has not been forwarded yet). Slots beyond
                ``request.num_computed_tokens`` are zero-initialized.
            token_start: Skip the first ``token_start`` tokens. The
                slot_mapping is sliced before the fancy-index read, so no
                large intermediate is allocated.

        Returns:
            Array of shape (num_tokens - token_start, num_layers,
            num_experts_per_tok).
        """
        bs = self.block_size
        block_ids_array = np.asarray(block_ids, dtype=np.int64)
        # slot = block_id * block_size + offset_in_block, computed only for the
        # requested ``[token_start, num_tokens)`` window (not the whole-sequence
        # grid) so decode reads stay O(returned tokens) and prefill avoids
        # materializing the full (num_blocks, block_size) grid.
        pos = np.arange(token_start, num_tokens)
        slot_mapping = block_ids_array[pos // bs] * bs + (pos % bs)
        return self.routed_experts_by_slot[slot_mapping]

    def get_by_slots(self, slots: np.ndarray) -> np.ndarray:
        """Read routing for explicit slot indices (decode path).

        ``slots`` are the physical KV-cache slots the worker scattered this
        step's routing into, returned per step (step-paired, so correct under
        async scheduling). Fancy indexing returns a copy, safe to hold.
        """
        return self.routed_experts_by_slot[slots]
