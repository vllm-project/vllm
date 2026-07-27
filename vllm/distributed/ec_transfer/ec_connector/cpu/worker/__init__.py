# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side of the ECCPUConnector.

Thin, stateless across steps: opens the shared mmap region and uses the
per-step connector metadata (`ECCPUConnectorMetadata`) to decide which
blocks to copy in each direction.
"""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm._custom_ops import swap_blocks_batch
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECCPUWorkerMetadata,
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker.descriptor_buffers import (
    DescriptorBufferPool,
    DescriptorBuffers,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class Transfer:
    """A batched copy in flight on a dedicated GPU stream.

    Its descriptor buffers and stream are held until `end_event` fires; only
    then are the buffers and stream safe to recycle and the `mm_hashes` safe to
    report as complete. `start_event`/`end_event` bracket the copy so its
    elapsed time can be measured on completion.
    """

    start_event: torch.Event
    end_event: torch.Event
    mm_hashes: list[str]
    bufs: DescriptorBuffers
    stream: torch.Stream
    num_bytes: int


class ECCPUWorker:
    """Worker-side delegate for the ECCPUConnector.

    - Producer role: copies `encoder_cache[mm_hash]` → `mmap[block_ids]`
      for each entry in `metadata.saves`. Descriptor buffers are filled
      directly in `save_caches`; the actual DMA is issued as a single
      batched call in `flush_saves`.
    - Consumer role: copies `mmap[block_ids]` → `encoder_cache[mm_hash]`
      for all entries in `metadata.loads` via a single `swap_blocks_batch`
      call on a dedicated stream.
    - On `ec_both` nodes both paths run back-to-back in a single step.

    Every batched copy runs on a stream drawn from a pool so it overlaps the
    compute stream, bracketed by a start/end event pair. The stream, events,
    and descriptor buffers are recycled once the end event fires.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        self._region = create_ec_shared_region(vllm_config)
        # Model dtype; used to reinterpret raw int8 blocks on load.
        self._dtype = vllm_config.model_config.dtype

        if is_pin_memory_available():
            self._region.pin_memory()

        # All TP/PCP ranks hold identical encoder output. Only one rank
        # per mmap needs to write — saves host memory bandwidth.
        # DCP is a subdivision of TP, so tp_rank==0 covers it.
        self._is_save_rank = (
            get_tensor_model_parallel_rank() == 0 and get_pcp_group().rank_in_group == 0
        )

        # Descriptor buffer pool (recycled across steps, shared by both paths).
        self._buf_pool = DescriptorBufferPool()

        # Active save buffer being filled during save_caches calls this step.
        self._save_bufs: DescriptorBuffers | None = None
        self._save_count: int = 0
        # mm_hashes accumulated into the active save buffer this step.
        self._save_mm_hashes: list[str] = []

        # Batched copies whose end event has not yet fired. Buffers and stream
        # stay held until the event fires; the mm_hashes are reported to the
        # scheduler only then. Separate per direction because saves and loads
        # can be in flight concurrently.
        self._inflight_saves: deque[Transfer] = deque()
        self._inflight_loads: deque[Transfer] = deque()

        # Streams and timing events recycled across steps to avoid per-step
        # create/destroy churn. Shared by both directions.
        self._stream_pool: list[torch.Stream] = []
        self._event_pool: list[torch.Event] = []

    def _acquire_stream(self) -> torch.Stream:
        if self._stream_pool:
            return self._stream_pool.pop()
        return current_platform.Stream()

    def _acquire_event(self) -> torch.Event:
        return (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

    def _collect_finished(self, inflight: deque[Transfer], direction: str) -> list[str]:
        """Pop transfers whose end event has fired, recycle their stream,
        events, and buffers, and return the mm_hashes that completed.

        The front check is conservative: a transfer is reported only once its
        own end event fires, so completed hashes are always genuinely done. A
        later transfer that happens to finish first simply waits behind the
        front and is reported on a subsequent poll.
        """
        done: list[str] = []
        while inflight and inflight[0].end_event.query():
            transfer = inflight.popleft()
            done.extend(transfer.mm_hashes)
            elapsed_ms = transfer.start_event.elapsed_time(transfer.end_event)
            logger.debug(
                "EC %s: %d entr%s (%d bytes) took %.3f ms",
                direction,
                len(transfer.mm_hashes),
                "y" if len(transfer.mm_hashes) == 1 else "ies",
                transfer.num_bytes,
                elapsed_ms,
            )
            self._buf_pool.release(transfer.bufs)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.start_event)
            self._event_pool.append(transfer.end_event)
        return done

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        """Fill descriptor buffers directly for batched flush."""
        if not self._is_save_rank:
            return
        block_ids = connector_metadata.saves.get(mm_hash)
        if block_ids is None:
            return

        src = encoder_cache[mm_hash]
        total_bytes = src.numel() * src.element_size()
        block_size = self._region.block_size_bytes
        allocated_bytes = len(block_ids) * block_size
        assert total_bytes <= allocated_bytes, (
            f"EC: encoder output exceeds allocated blocks for mm_hash={mm_hash}: "
            f"{total_bytes} bytes but only {allocated_bytes} allocated "
            f"({len(block_ids)} blocks × {block_size}). "
            f"shape={list(src.shape)}"
        )

        if self._save_bufs is None:
            total = sum(len(v) for v in connector_metadata.saves.values())
            self._save_bufs = self._buf_pool.acquire(total)
            self._save_mm_hashes = []

        assert self._save_count + len(block_ids) <= self._save_bufs.src_ptrs.numel()

        src_ptrs, dst_ptrs, sizes = self._save_bufs
        src_base = src.view(-1).view(torch.uint8).data_ptr()
        dst_base = self._region.blocks.data_ptr()
        idx = self._save_count

        for i, block_idx in enumerate(block_ids):
            start = i * block_size
            src_ptrs[idx] = src_base + start
            dst_ptrs[idx] = dst_base + block_idx * block_size
            sizes[idx] = min(block_size, total_bytes - start)
            idx += 1

        self._save_count = idx
        self._save_mm_hashes.append(mm_hash)

    def flush_saves(self) -> None:
        """Flush all accumulated saves in a single swap_blocks_batch call.

        Runs the copy on a pooled stream gated behind the compute stream that
        produced the encoder outputs, brackets it with start/end events, and
        enqueues the batch as in-flight. The stream, events, and descriptor
        buffers are recycled only once the end event fires (see
        `_collect_finished`), which is also when the saved mm_hashes become
        safe to mark ready.
        """
        if self._save_count == 0:
            return

        bufs = self._save_bufs
        assert bufs is not None
        src_ptrs, dst_ptrs, sizes = bufs
        n = self._save_count
        num_bytes = int(sizes[:n].sum().item())

        stream = self._acquire_stream()
        # Gate the GPU→CPU copy behind the compute stream: it reads the encoder
        # outputs the model just produced there.
        stream.wait_stream(current_platform.current_stream())
        start_event = self._acquire_event()
        end_event = self._acquire_event()
        with current_platform.stream(stream):
            start_event.record(stream)
            swap_blocks_batch(src_ptrs[:n], dst_ptrs[:n], sizes[:n])
            end_event.record(stream)

        self._inflight_saves.append(
            Transfer(
                start_event=start_event,
                end_event=end_event,
                mm_hashes=self._save_mm_hashes,
                bufs=bufs,
                stream=stream,
                num_bytes=num_bytes,
            )
        )

        self._save_bufs = None
        self._save_count = 0
        self._save_mm_hashes = []

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        """Consumer path: single batched copy of all loads from mmap→GPU."""
        if not connector_metadata.loads:
            return

        block_size = self._region.block_size_bytes
        blocks = self._region.blocks
        dtype = self._dtype
        device_type = current_platform.device_type
        src_base = blocks.data_ptr()

        # Copy every dispatched load. The scheduler routes an input here only
        # after missing the GPU encoder cache, so a resident hit is not expected;
        # copying unconditionally keeps every scheduler pin balanced by exactly
        # one completion report.
        load_items = connector_metadata.loads
        total_blocks = sum(len(idxs) for idxs in load_items.values())

        stream = self._acquire_stream()
        start_event = self._acquire_event()
        end_event = self._acquire_event()
        with current_platform.stream(stream):
            # Single contiguous destination buffer for all loads.
            dst_buf = torch.empty(
                total_blocks, block_size, dtype=torch.int8, device=device_type
            )
            dst_buf_base = dst_buf.data_ptr()

            bufs = self._buf_pool.acquire(total_blocks)
            src_ptrs = bufs.src_ptrs[:total_blocks]
            dst_ptrs = bufs.dst_ptrs[:total_blocks]
            sizes = bufs.sizes[:total_blocks]
            sizes[:] = block_size

            op_idx = 0
            for block_ids in load_items.values():
                for block_idx in block_ids:
                    src_ptrs[op_idx] = src_base + block_idx * block_size
                    dst_ptrs[op_idx] = dst_buf_base + op_idx * block_size
                    op_idx += 1

            start_event.record(stream)
            swap_blocks_batch(src_ptrs, dst_ptrs, sizes, is_src_access_order_any=True)
            end_event.record(stream)
            self._inflight_loads.append(
                Transfer(
                    start_event=start_event,
                    end_event=end_event,
                    mm_hashes=list(load_items.keys()),
                    bufs=bufs,
                    stream=stream,
                    num_bytes=total_blocks * block_size,
                )
            )

            # Slice contiguous buffer into per-hash views.
            offset = 0
            for mm_hash, block_ids in load_items.items():
                n = len(block_ids)
                encoder_cache[mm_hash] = (
                    dst_buf[offset : offset + n].view(dtype).reshape(n, -1)
                )
                offset += n

        current_platform.current_stream().wait_stream(stream)

    def build_connector_worker_meta(self) -> ECCPUWorkerMetadata | None:
        """Report the mm_hashes whose GPU copies have completed this step.

        Returns None when nothing finished, so the scheduler sees no payload.
        """
        completed_saves = self._collect_finished(self._inflight_saves, "save")
        completed_loads = self._collect_finished(self._inflight_loads, "load")
        if not completed_saves and not completed_loads:
            return None
        return ECCPUWorkerMetadata(
            completed_saves=completed_saves,
            completed_loads=completed_loads,
        )

    def shutdown(self) -> None:
        for transfer in (*self._inflight_saves, *self._inflight_loads):
            transfer.end_event.synchronize()
        self._save_bufs = None
        self._save_count = 0
        self._save_mm_hashes = []
        self._inflight_saves.clear()
        self._inflight_loads.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        try:
            self._region.cleanup()
        except Exception:
            logger.debug("EC: worker region cleanup failed", exc_info=True)
