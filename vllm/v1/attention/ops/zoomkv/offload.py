# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned CPU KV pool + async D2H/H2D for ZoomKV K+V offload.

Block lifecycle (three states per physical block, per layer):

1. **GPU-only** — no CPU copy. All reads hit the paged cache.
2. **warm** — a CPU copy exists (D2H issued when the block completed during
   the KV-cache update) but the GPU page is still intact, because a dense
   reader (the same step's prefill attention, later prefill chunks, or a
   mixed dense-decode batch) may still need it.
3. **cold** — the GPU page has been zeroed; the CPU copy is the only
   full-precision source. Only entered from the sparse decode path, which
   never reads cold pages from GPU (retrieval uses block summaries and the
   hybrid gather reads cold tokens straight from pinned memory).

Transitions:
  offload_blocks_bulk : GPU-only -> warm  (D2H copy, no zeroing)
  mark_cold           : warm -> cold      (GPU zero only, no PCIe traffic)
  restore_blocks      : cold -> warm      (H2D copy back; CPU copy retained,
                                           so the next mark_cold is free)
  free_gpu_blocks     : any -> GPU-only   (block reused by the allocator)

Because block content is immutable once a block completes, a warm block's
CPU copy stays valid forever (until the block is freed), which is what makes
the cold/restore cycle cheap: only the first offload pays D2H and only an
actual dense read pays H2D.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ZoomKVOffloadMetrics:
    d2h_bytes: int = 0
    h2d_bytes: int = 0
    d2h_events: int = 0
    h2d_events: int = 0
    cpu_slots_in_use: int = 0
    cpu_slots_capacity: int = 0


class ZoomKVCpuKeyPool:
    """Per-rank pinned CPU KV pool indexed by physical GPU block id.

    Layout (per layer):
      key:   [num_slots, block_size, num_kv_heads, head_dim] pinned host
      value: [num_slots, block_size, num_kv_heads, head_dim] pinned host
    Block summaries for offloaded chunks stay on GPU, indexed by CPU slot.
    """

    def __init__(
        self,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_names: list[str],
        strict: bool = False,
    ) -> None:
        if block_size != 16:
            raise ValueError(f"ZoomKV offload requires block_size=16, got {block_size}")
        self.num_slots = int(num_slots)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.dtype = dtype
        self.device = device
        self.strict = strict
        self.layer_names = list(layer_names)
        self.metrics = ZoomKVOffloadMetrics(cpu_slots_capacity=self.num_slots)
        self._lock = threading.Lock()
        self._free_slots: list[int] = list(range(self.num_slots))
        # (layer_name, gpu_block_id) -> cpu_slot
        self._map: dict[tuple[str, int], int] = {}
        self._slot_to_block: dict[tuple[str, int], int] = {}
        # Host mirrors of block state so hot paths never sync on the GPU
        # offloaded_mask. warm: CPU copy exists, GPU page intact.
        # cold: GPU page zeroed (mirrors offloaded_mask).
        self._warm: dict[str, set[int]] = {n: set() for n in layer_names}
        self._cold: dict[str, set[int]] = {n: set() for n in layer_names}

        n_pack = head_dim // 8
        self.key: dict[str, torch.Tensor] = {}
        # Value is offloaded alongside Key (symmetric K+V offload): each CPU
        # slot holds the full-precision Value page for its physical block.
        self.value: dict[str, torch.Tensor] = {}
        self.slot_min: dict[str, torch.Tensor] = {}
        self.slot_max: dict[str, torch.Tensor] = {}
        self.slot_centroid: dict[str, torch.Tensor] = {}
        self.slot_packed: dict[str, torch.Tensor] = {}
        self.slot_valid: dict[str, torch.Tensor] = {}
        # GPU bool mask [num_gpu_blocks] — True when Key was offloaded.
        self.offloaded_mask: dict[str, torch.Tensor] = {}

        for name in self.layer_names:
            self.key[name] = torch.zeros(
                self.num_slots,
                block_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                pin_memory=True,
            )
            self.value[name] = torch.zeros(
                self.num_slots,
                block_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                pin_memory=True,
            )
            self.slot_min[name] = torch.zeros(
                self.num_slots, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            self.slot_max[name] = torch.zeros(
                self.num_slots, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            self.slot_centroid[name] = torch.zeros(
                self.num_slots, num_kv_heads, head_dim, device=device, dtype=dtype
            )
            self.slot_packed[name] = torch.zeros(
                self.num_slots,
                num_kv_heads,
                n_pack,
                block_size,
                device=device,
                dtype=torch.int32,
            )
            self.slot_valid[name] = torch.zeros(
                self.num_slots, device=device, dtype=torch.bool
            )

        self.d2h_stream = torch.cuda.Stream(device=device)
        self.h2d_stream = torch.cuda.Stream(device=device)

        # Each slot stores both a Key and a Value page (K+V offload).
        bytes_per_slot = (
            2
            * block_size
            * num_kv_heads
            * head_dim
            * dtype.itemsize
            * max(1, len(self.layer_names))
        )
        logger.info(
            "ZoomKV K+V CPU pool: slots=%d layers=%d ~%.2f GiB pinned",
            self.num_slots,
            len(self.layer_names),
            self.num_slots * bytes_per_slot / (1024**3),
        )

    def ensure_offload_mask(self, layer_name: str, num_blocks: int) -> torch.Tensor:
        mask = self.offloaded_mask.get(layer_name)
        if mask is None or mask.numel() != num_blocks:
            mask = torch.zeros(num_blocks, device=self.device, dtype=torch.bool)
            self.offloaded_mask[layer_name] = mask
        return mask

    def free_gpu_blocks(self, layer_name: str, gpu_block_ids: list[int]) -> None:
        if not gpu_block_ids:
            return
        with self._lock:
            warm = self._warm.setdefault(layer_name, set())
            cold = self._cold.setdefault(layer_name, set())
            for gpu_block in gpu_block_ids:
                key = (layer_name, int(gpu_block))
                slot = self._map.pop(key, None)
                warm.discard(int(gpu_block))
                cold.discard(int(gpu_block))
                if slot is None:
                    continue
                self._slot_to_block.pop((layer_name, slot), None)
                self.slot_valid[layer_name][slot] = False
                self._free_slots.append(slot)
            mask = self.offloaded_mask.get(layer_name)
            if mask is not None and gpu_block_ids:
                ids = torch.tensor(
                    [b for b in gpu_block_ids if 0 <= b < mask.numel()],
                    dtype=torch.int64,
                    device=self.device,
                )
                if ids.numel():
                    mask.index_fill_(0, ids, False)
            self.metrics.cpu_slots_in_use = self.num_slots - len(self._free_slots)

    def free_gpu_blocks_all_layers(self, gpu_block_ids: list[int]) -> None:
        for name in self.layer_names:
            self.free_gpu_blocks(name, gpu_block_ids)

    def lookup_slot(self, layer_name: str, gpu_block_id: int) -> int | None:
        return self._map.get((layer_name, int(gpu_block_id)))

    def slots_from_block_ids(
        self, layer_name: str, block_ids: list[int]
    ) -> tuple[torch.Tensor, list[int]]:
        """Map host-side physical block ids to CPU slots (-1 if absent).

        Returns both the GPU slot tensor and the host list so callers can
        branch on slot presence without a GPU->CPU synchronization.
        """
        slots = [
            self._map.get((layer_name, b), -1) if b >= 0 else -1 for b in block_ids
        ]
        return torch.tensor(slots, dtype=torch.int64, device=self.device), slots

    def lookup_slots_for_physical_ids(
        self,
        layer_name: str,
        phys_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Map physical GPU block ids to CPU slots (-1 if not offloaded)."""
        ids = phys_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        slots, _ = self.slots_from_block_ids(layer_name, [int(b) for b in ids.tolist()])
        return slots

    def offload_blocks_bulk(
        self,
        layer_name: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_summary,
        gpu_block_ids: torch.Tensor,
    ) -> int:
        """GPU-only -> warm: D2H-copy completed blocks; GPU pages stay intact.

        Callers pass blocks that just completed in this step's KV-cache
        update, so their block summaries are valid by construction. GPU pages
        are NOT zeroed here — zeroing is deferred to :meth:`mark_cold`, which
        only runs from the sparse decode path where no dense reader exists.
        """
        if gpu_block_ids is None or gpu_block_ids.numel() == 0:
            return 0
        ids = gpu_block_ids.detach().to(device="cpu", dtype=torch.int64).unique()
        num_gpu_blocks = key_cache.shape[0]
        new_blocks: list[int] = []
        new_slots: list[int] = []
        with self._lock:
            for b in ids.tolist():
                bi = int(b)
                if bi < 0 or bi >= num_gpu_blocks:
                    continue
                if (layer_name, bi) in self._map:
                    continue
                if not self._free_slots:
                    self.metrics.cpu_slots_in_use = self.num_slots
                    if self.strict:
                        raise RuntimeError(
                            f"ZoomKV CPU KV pool exhausted: capacity={self.num_slots}"
                        )
                    logger.warning_once(
                        "ZoomKV CPU KV pool exhausted: capacity=%d", self.num_slots
                    )
                    break
                slot = self._free_slots.pop()
                self._map[(layer_name, bi)] = slot
                self._slot_to_block[(layer_name, slot)] = bi
                new_blocks.append(bi)
                new_slots.append(slot)
            self._warm.setdefault(layer_name, set()).update(new_blocks)
            self.metrics.cpu_slots_in_use = self.num_slots - len(self._free_slots)
        if not new_blocks:
            return 0

        blocks_gpu = torch.tensor(new_blocks, device=self.device, dtype=torch.int64)
        slots_gpu = torch.tensor(new_slots, device=self.device, dtype=torch.int64)
        # Snapshot the summaries into slot-indexed buffers (all GPU-side).
        self.slot_min[layer_name].index_copy_(
            0, slots_gpu, block_summary.chunk_min.index_select(0, blocks_gpu)
        )
        self.slot_max[layer_name].index_copy_(
            0, slots_gpu, block_summary.chunk_max.index_select(0, blocks_gpu)
        )
        self.slot_centroid[layer_name].index_copy_(
            0, slots_gpu, block_summary.centroid.index_select(0, blocks_gpu)
        )
        self.slot_packed[layer_name].index_copy_(
            0, slots_gpu, block_summary.packed.index_select(0, blocks_gpu)
        )
        self.slot_valid[layer_name].index_fill_(0, slots_gpu, True)

        # D2H after this step's KV writes have been enqueued on the current
        # stream; the copies run in the background on d2h_stream.
        current = torch.cuda.current_stream(device=self.device)
        self.d2h_stream.wait_stream(current)
        bytes_one = (
            self.block_size * self.num_kv_heads * self.head_dim * self.dtype.itemsize
        )
        with torch.cuda.stream(self.d2h_stream):
            for bi, slot in zip(new_blocks, new_slots):
                self.key[layer_name][slot].copy_(key_cache[bi], non_blocking=True)
                self.value[layer_name][slot].copy_(value_cache[bi], non_blocking=True)
        self.metrics.d2h_bytes += 2 * bytes_one * len(new_blocks)
        self.metrics.d2h_events += 1
        return len(new_blocks)

    def mark_cold(
        self,
        layer_name: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        candidate_block_ids: list[int],
    ) -> int:
        """warm -> cold: zero GPU K/V pages of blocks that have a CPU copy.

        Pure GPU zeroing — no PCIe traffic. Callers must guarantee no dense
        reader needs these pages (the sparse decode path passes only its
        retrieval-zone blocks; sparse reads of cold blocks go through the
        hybrid gather which reads pinned memory directly).
        """
        warm = self._warm.get(layer_name)
        if not warm:
            return 0
        to_zero = [b for b in candidate_block_ids if b in warm]
        if not to_zero:
            return 0
        with self._lock:
            warm.difference_update(to_zero)
            self._cold.setdefault(layer_name, set()).update(to_zero)
        mask = self.ensure_offload_mask(layer_name, key_cache.shape[0])
        ids = torch.tensor(to_zero, device=self.device, dtype=torch.int64)
        # The D2H copies for these blocks were queued on d2h_stream when they
        # became warm; order the zeroing after them.
        torch.cuda.current_stream(device=self.device).wait_stream(self.d2h_stream)
        key_cache.index_fill_(0, ids, 0)
        value_cache.index_fill_(0, ids, 0)
        mask.index_fill_(0, ids, True)
        return len(to_zero)

    def has_cold_blocks(self, layer_name: str) -> bool:
        cold = self._cold.get(layer_name)
        return bool(cold)

    def restore_blocks(
        self,
        layer_name: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        candidate_block_ids: list[int],
    ) -> int:
        """cold -> warm: H2D-copy K/V pages back before a dense read.

        The CPU copy and slot mapping are retained (block content is
        immutable), so a later :meth:`mark_cold` re-zeroes without new D2H.
        """
        cold = self._cold.get(layer_name)
        if not cold:
            return 0
        to_restore = [b for b in candidate_block_ids if b in cold]
        if not to_restore:
            return 0
        with self._lock:
            cold.difference_update(to_restore)
            self._warm.setdefault(layer_name, set()).update(to_restore)
            slots = [self._map[(layer_name, b)] for b in to_restore]
        mask = self.ensure_offload_mask(layer_name, key_cache.shape[0])
        current = torch.cuda.current_stream(device=self.device)
        # Ensure the original D2H copies have landed in pinned memory before
        # reading it back.
        current.wait_stream(self.d2h_stream)
        key_pool = self.key[layer_name]
        value_pool = self.value[layer_name]
        # Pinned-source copies on the current stream: async DMA, and every
        # later kernel on this stream (the dense attention) is ordered after.
        for bi, slot in zip(to_restore, slots):
            key_cache[bi].copy_(key_pool[slot], non_blocking=True)
            value_cache[bi].copy_(value_pool[slot], non_blocking=True)
        ids = torch.tensor(to_restore, device=self.device, dtype=torch.int64)
        mask.index_fill_(0, ids, False)
        bytes_one = (
            self.block_size * self.num_kv_heads * self.head_dim * self.dtype.itemsize
        )
        self.metrics.h2d_bytes += 2 * bytes_one * len(to_restore)
        self.metrics.h2d_events += 1
        return len(to_restore)

    def gather_key_h2d(
        self,
        layer_name: str,
        cpu_slots: torch.Tensor,
        token_offsets: torch.Tensor,
        out_k: torch.Tensor,
    ) -> None:
        """Gather selected tokens from pinned CPU Key into GPU out_k [N,H,D]."""
        from vllm.v1.attention.ops.zoomkv.kernels import h2d_gather_keys

        h2d_gather_keys(
            self.key[layer_name],
            cpu_slots,
            token_offsets,
            out_k,
            stream=self.h2d_stream,
            strict=self.strict,
        )
        n = int(cpu_slots.numel())
        self.metrics.h2d_bytes += (
            n * self.num_kv_heads * self.head_dim * self.dtype.itemsize
        )
        self.metrics.h2d_events += 1

    def gather_block_summaries_by_physical_ids(
        self,
        layer_name: str,
        phys_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cpu_slots = self.lookup_slots_for_physical_ids(layer_name, phys_ids)
        ids = cpu_slots.to(torch.int64).clamp(0, self.num_slots - 1)
        n = ids.numel()
        packed = (
            self.slot_packed[layer_name]
            .index_select(0, ids)
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
            .contiguous()
        )
        chunk_min = (
            self.slot_min[layer_name]
            .index_select(0, ids)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        chunk_max = (
            self.slot_max[layer_name]
            .index_select(0, ids)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        centroid = (
            self.slot_centroid[layer_name]
            .index_select(0, ids)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        valid = self.slot_valid[layer_name].index_select(0, ids)
        missing = cpu_slots < 0
        valid = valid & (~missing.to(device=valid.device))
        valid = valid.view(1, 1, n).expand(1, self.num_kv_heads, n).contiguous()
        return packed, chunk_min, chunk_max, centroid, valid

    def reset(self) -> None:
        logger.info(
            "ZoomKV K+V offload metrics: D2H=%.2f MiB/%d, H2D=%.2f MiB/%d, "
            "slots_in_use=%d",
            self.metrics.d2h_bytes / (1024**2),
            self.metrics.d2h_events,
            self.metrics.h2d_bytes / (1024**2),
            self.metrics.h2d_events,
            self.metrics.cpu_slots_in_use,
        )
        with self._lock:
            self._free_slots = list(range(self.num_slots))
            self._map.clear()
            self._slot_to_block.clear()
            self._warm = {n: set() for n in self.layer_names}
            self._cold = {n: set() for n in self.layer_names}
            self.metrics = ZoomKVOffloadMetrics(cpu_slots_capacity=self.num_slots)
        for name in self.layer_names:
            # Avoid inplace writes on InferenceMode tensors during worker shutdown.
            self.slot_valid[name] = torch.zeros_like(self.slot_valid[name])
            mask = self.offloaded_mask.get(name)
            if mask is not None:
                self.offloaded_mask[name] = torch.zeros_like(mask)


_CPU_KEY_POOL: ZoomKVCpuKeyPool | None = None


def get_cpu_key_pool() -> ZoomKVCpuKeyPool | None:
    return _CPU_KEY_POOL


def set_cpu_key_pool(pool: ZoomKVCpuKeyPool | None) -> None:
    global _CPU_KEY_POOL
    _CPU_KEY_POOL = pool


# Back-compat aliases used by worker hooks.
def get_cpu_pool() -> ZoomKVCpuKeyPool | None:
    return get_cpu_key_pool()


def set_cpu_pool(pool: ZoomKVCpuKeyPool | None) -> None:
    set_cpu_key_pool(pool)
