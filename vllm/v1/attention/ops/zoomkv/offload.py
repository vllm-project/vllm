# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned CPU KV pool + async D2H/H2D for ZoomKV K+V offload.

How this maps onto vLLM's own paging (do not "adapt" chunks ad-hoc):

* Scheduler chunked prefill emits up to ``max_num_batched_tokens`` tokens,
  aligned down to ``cache_config.block_size`` so a finished page can be
  prefix-cached. Physical id = ``block_table[req][logical_block]`` from
  the shared ``BlockPool``. Slot = ``phys * block_size + offset``.
* ZoomKV child / Quest chunk is 16 tokens. That **is** vLLM's page, not a
  second paging scheme. Changing ``--block-size`` to 64 would desync Quest
  children, slot_mapping, and Qwen3.6 GDN layers that share the pool.
* Offload granularity is a **logical** 64-token unit (4 child pages,
  matching ``sink_size``). We still store one CPU slot per 16-token page
  so physical ids stay the gather key. D2H waits until a retrieval-zone
  unit is complete; sink, the sliding local window, and the in-flight
  write page stay GPU-only.
* Chunked prefill dense FA still needs the whole prefix on GPU, so D2H
  during prefill is warm-only (GPU pages stay). ``mark_cold`` (zero GPU
  pages) and CPU top-k gather run only on the sparse decode path.

Block lifecycle (three states per physical block, per layer):

1. **GPU-only** — no CPU copy. All reads hit the paged cache.
2. **warm** — a CPU copy exists (D2H issued when a retrieval-zone unit
   completed) but the GPU page is still intact, because a dense reader
   (the same step's prefill attention, later prefill chunks, or a mixed
   dense-decode batch) may still need it.
3. **cold** — the GPU page has been zeroed; the CPU copy is the only
   full-precision source. Only entered from the sparse decode path.
   Sparse attention GPU-gathers sink+local and CPU-gathers Top-K.

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
from vllm.v1.attention.ops.zoomkv import stage_timer as _zt

logger = init_logger(__name__)


def retrieval_zone_logical_range(
    seq_len: int,
    sink_size: int,
    local_size: int,
    block_size: int,
    unit_tokens: int = 64,
    *,
    flush: bool = False,
) -> tuple[int, int]:
    """Logical block range that is safe to mirror to CPU.

    Exclusive of sink and of the sliding local / in-flight write window.
    When ``flush`` is false, the end is rounded down to a complete
    ``unit_tokens`` unit (default 64 = 4 children). Sparse-decode entry
    sets ``flush=True`` so a leftover 16/32/48-token tail is copied too.
    """
    if seq_len <= 0 or block_size <= 0:
        return 0, 0
    start_b = sink_size // block_size
    local_start = max(sink_size, seq_len - local_size)
    end_b = local_start // block_size
    if end_b <= start_b:
        return start_b, start_b
    if not flush:
        unit_blocks = max(1, int(unit_tokens) // block_size)
        n = end_b - start_b
        end_b = start_b + (n - n % unit_blocks)
    return start_b, end_b


def physical_ids_in_retrieval_zone(
    block_table,
    seq_lens,
    *,
    sink_size: int,
    local_size: int,
    block_size: int,
    unit_tokens: int = 64,
    flush: bool = False,
    num_reqs: int | None = None,
) -> list[int]:
    """Host physical ids currently inside the offloadable retrieval zone."""
    if block_table is None or seq_lens is None:
        return []
    if hasattr(seq_lens, "tolist"):
        seq_list = seq_lens.detach().to(device="cpu").tolist()
    else:
        seq_list = list(seq_lens)
    n = len(seq_list) if num_reqs is None else min(int(num_reqs), len(seq_list))
    bt = block_table
    if hasattr(bt, "detach"):
        bt = bt.detach().to(device="cpu")
    seen: set[int] = set()
    out: list[int] = []
    for i in range(n):
        start_b, end_b = retrieval_zone_logical_range(
            int(seq_list[i]),
            sink_size,
            local_size,
            block_size,
            unit_tokens,
            flush=flush,
        )
        if end_b <= start_b:
            continue
        row = bt[i, start_b:end_b] if bt.ndim > 1 else bt[start_b:end_b]
        for raw in row.tolist() if hasattr(row, "tolist") else list(row):
            phys = int(raw)
            if phys < 0 or phys in seen:
                continue
            seen.add(phys)
            out.append(phys)
    return out


def filter_completed_for_offload(
    completed_phys: torch.Tensor,
    block_table,
    seq_lens,
    *,
    sink_size: int,
    local_size: int,
    block_size: int,
    unit_tokens: int = 64,
    num_reqs: int = 0,
) -> torch.Tensor:
    """Intersect this step's completed pages with 64-token retrieval units."""
    if completed_phys is None or completed_phys.numel() == 0:
        return completed_phys
    zone = set(
        physical_ids_in_retrieval_zone(
            block_table,
            seq_lens,
            sink_size=sink_size,
            local_size=local_size,
            block_size=block_size,
            unit_tokens=unit_tokens,
            flush=False,
            num_reqs=num_reqs,
        )
    )
    if not zone:
        return completed_phys.new_empty((0,))
    kept = [int(b) for b in completed_phys.detach().to(device="cpu").tolist() if int(b) in zone]
    if not kept:
        return completed_phys.new_empty((0,))
    return torch.tensor(kept, device=completed_phys.device, dtype=completed_phys.dtype)


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
    Block summaries stay in their original GPU physical-block layout.
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
        # Every layer owns a [num_slots, ...] CPU tensor, so num_slots is a
        # per-layer capacity. Sharing one free list across layers incorrectly
        # divides the configured capacity by the number of attention layers.
        self._free_slots: dict[str, list[int]] = {
            name: list(range(self.num_slots)) for name in self.layer_names
        }
        # (layer_name, gpu_block_id) -> cpu_slot
        self._map: dict[tuple[str, int], int] = {}
        self._slot_to_block: dict[tuple[str, int], int] = {}
        # Host mirrors of block state so hot paths never sync on the GPU
        # offloaded_mask. warm: CPU copy exists, GPU page intact.
        # cold: GPU page zeroed (mirrors offloaded_mask).
        self._warm: dict[str, set[int]] = {n: set() for n in layer_names}
        self._cold: dict[str, set[int]] = {n: set() for n in layer_names}

        self.key: dict[str, torch.Tensor] = {}
        # Value is offloaded alongside Key (symmetric K+V offload): each CPU
        # slot holds the full-precision Value page for its physical block.
        self.value: dict[str, torch.Tensor] = {}
        # GPU bool mask [num_gpu_blocks] — True when Key was offloaded.
        self.offloaded_mask: dict[str, torch.Tensor] = {}
        # Persistent GPU map [physical_block] -> CPU slot. This lets the final
        # gather resolve selected tokens entirely on GPU without rebuilding a
        # full block-table-sized slot tensor on every layer and decode step.
        self.physical_to_slot: dict[str, torch.Tensor] = {}

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

    def ensure_physical_to_slot(
        self, layer_name: str, num_blocks: int
    ) -> torch.Tensor:
        slot_map = self.physical_to_slot.get(layer_name)
        if slot_map is None or slot_map.numel() != num_blocks:
            slot_map = torch.full(
                (num_blocks,),
                -1,
                device=self.device,
                dtype=torch.int64,
            )
            self.physical_to_slot[layer_name] = slot_map
        return slot_map

    def _update_slots_in_use(self) -> None:
        self.metrics.cpu_slots_in_use = max(
            (
                self.num_slots - len(self._free_slots[name])
                for name in self.layer_names
            ),
            default=0,
        )

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
                self._free_slots[layer_name].append(slot)
            mask = self.offloaded_mask.get(layer_name)
            slot_map = self.physical_to_slot.get(layer_name)
            map_size = 0
            if slot_map is not None:
                map_size = slot_map.numel()
            elif mask is not None:
                map_size = mask.numel()
            if map_size and gpu_block_ids:
                ids = torch.tensor(
                    [b for b in gpu_block_ids if 0 <= b < map_size],
                    dtype=torch.int64,
                    device=self.device,
                )
                if ids.numel():
                    if mask is not None:
                        mask.index_fill_(0, ids, False)
                    if slot_map is not None:
                        slot_map.index_fill_(0, ids, -1)
            self._update_slots_in_use()

    def free_gpu_blocks_all_layers(
        self,
        gpu_block_ids: list[int],
        allocation_num_blocks: int | None = None,
    ) -> None:
        # Hybrid cache groups allocate several native 16-token attention
        # pages per scheduler block. The block table and CPU pool use those
        # expanded physical ids, while new_block_ids_to_zero contains base
        # scheduler ids. Mirror block-summary invalidation's expansion here.
        expanded = gpu_block_ids
        if allocation_num_blocks:
            physical_num_blocks = max(
                (slot_map.numel() for slot_map in self.physical_to_slot.values()),
                default=allocation_num_blocks,
            )
            factor = (
                physical_num_blocks // allocation_num_blocks
                if physical_num_blocks % allocation_num_blocks == 0
                else 1
            )
            if factor > 1:
                expanded = [
                    block_id * factor + offset
                    for block_id in gpu_block_ids
                    for offset in range(factor)
                ]
        for name in self.layer_names:
            self.free_gpu_blocks(name, expanded)

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
        gpu_block_ids: torch.Tensor | list[int],
    ) -> int:
        """GPU-only -> warm: D2H-copy completed blocks; GPU pages stay intact.

        Callers pass blocks that just completed in this step's KV-cache
        update, so their block summaries are valid by construction. GPU pages
        are NOT zeroed here — zeroing is deferred to :meth:`mark_cold`, which
        only runs from the sparse decode path where no dense reader exists.

        Accepts a host list to avoid a device->host sync on hot paths.
        """
        if gpu_block_ids is None:
            return 0
        if isinstance(gpu_block_ids, torch.Tensor):
            if gpu_block_ids.numel() == 0:
                return 0
            ids = gpu_block_ids.detach().to(device="cpu", dtype=torch.int64).unique()
        else:
            if not gpu_block_ids:
                return 0
            ids = torch.tensor(sorted(set(gpu_block_ids)), dtype=torch.int64)
        num_gpu_blocks = key_cache.shape[0]
        new_blocks: list[int] = []
        new_slots: list[int] = []
        with self._lock:
            free_slots = self._free_slots[layer_name]
            for b in ids.tolist():
                bi = int(b)
                if bi < 0 or bi >= num_gpu_blocks:
                    continue
                if (layer_name, bi) in self._map:
                    continue
                if not free_slots:
                    self.metrics.cpu_slots_in_use = self.num_slots
                    if self.strict:
                        raise RuntimeError(
                            f"ZoomKV CPU KV pool exhausted: capacity={self.num_slots}"
                        )
                    logger.warning_once(
                        "ZoomKV CPU KV pool exhausted: capacity=%d", self.num_slots
                    )
                    break
                slot = free_slots.pop()
                self._map[(layer_name, bi)] = slot
                self._slot_to_block[(layer_name, slot)] = bi
                new_blocks.append(bi)
                new_slots.append(slot)
            self._warm.setdefault(layer_name, set()).update(new_blocks)
            self._update_slots_in_use()
        if not new_blocks:
            return 0

        blocks_gpu = torch.tensor(new_blocks, device=self.device, dtype=torch.int64)
        slots_gpu = torch.tensor(new_slots, device=self.device, dtype=torch.int64)
        self.ensure_physical_to_slot(layer_name, num_gpu_blocks).index_copy_(
            0, blocks_gpu, slots_gpu
        )

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
        candidate_block_ids: list[int] | None,
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
        # All warm entries were admitted from the retrieval zone. The sparse
        # path can therefore transition the warm set directly, avoiding an
        # O(context) scan of the visible block table on every decode step.
        to_zero = (
            list(warm)
            if candidate_block_ids is None
            else [b for b in candidate_block_ids if b in warm]
        )
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

        with _zt.Stage("cpu_gather.keys_h2d"):
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
            self._free_slots = {
                name: list(range(self.num_slots)) for name in self.layer_names
            }
            self._map.clear()
            self._slot_to_block.clear()
            self._warm = {n: set() for n in self.layer_names}
            self._cold = {n: set() for n in self.layer_names}
            self.metrics = ZoomKVOffloadMetrics(cpu_slots_capacity=self.num_slots)
        for name in self.layer_names:
            mask = self.offloaded_mask.get(name)
            if mask is not None:
                self.offloaded_mask[name] = torch.zeros_like(mask)
            slot_map = self.physical_to_slot.get(name)
            if slot_map is not None:
                self.physical_to_slot[name] = torch.full_like(slot_map, -1)


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
