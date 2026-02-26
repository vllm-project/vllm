# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async hidden states writer.

Simple async writer that copies GPU tensors to pinned memory on a
dedicated CUDA stream, then writes safetensors files in a background
thread. This keeps disk I/O off the forward-pass critical path.

Pipeline:
  GPU tensor → [transfer stream] pinned copy → [bg thread] save_file
"""

import os
import queue
import tempfile
import threading
import time
from collections import deque
from typing import Optional

import safetensors.torch
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Optional zstd support
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


class _PinnedBufferPool:
    """Simple pool of reusable pinned-memory tensors.

    Avoids repeated cudaHostAlloc/cudaFreeHost calls on the hot path.
    Buffers are keyed by (shape, dtype) and reused when available.
    """

    def __init__(self, max_per_key: int = 8):
        self._max_per_key = max_per_key
        self._pool: dict[tuple, deque[torch.Tensor]] = {}

    def get(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        key = (shape, dtype)
        pool = self._pool.get(key)
        if pool:
            return pool.popleft()
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    def put(self, tensor: torch.Tensor) -> None:
        key = (tuple(tensor.shape), tensor.dtype)
        pool = self._pool.setdefault(key, deque())
        if len(pool) < self._max_per_key:
            pool.append(tensor)


class HiddenStatesWriter:
    """Non-blocking safetensors writer with async GPU→CPU transfer."""

    def __init__(
        self,
        shared_storage_path: str = "/tmp",
        use_compression: bool = True,
        compression_level: int = 3,
    ):
        self.shared_storage_path = shared_storage_path
        self.use_compression = use_compression and ZSTD_AVAILABLE
        self.compression_level = compression_level
        os.makedirs(shared_storage_path, exist_ok=True)

        if use_compression and not ZSTD_AVAILABLE:
            logger.warning(
                "Compression requested but zstandard not installed. "
                "Install with: pip install zstandard"
            )

        self._transfer_stream: Optional[torch.cuda.Stream] = None
        self._write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._shutdown = False
        self._pinned_pool = _PinnedBufferPool()

        # Reusable compressor (thread-safe for single-thread use)
        self._compressor = None
        if self.use_compression:
            self._compressor = zstd.ZstdCompressor(
                level=self.compression_level,
            )

        # Stats
        self._total_writes = 0
        self._total_bytes = 0

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="hs-writer",
        )
        self._writer_thread.start()
        logger.info(
            "HiddenStatesWriter initialized: path=%s, compression=%s",
            shared_storage_path, self.use_compression,
        )

    def write_async(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        filename: str,
    ) -> None:
        """Queue hidden states for async write to disk.

        Copies tensors to CPU via pinned memory on a transfer stream,
        then hands off to the background writer thread.
        """
        device = hidden_states.device

        if device.type == "cuda":
            if self._transfer_stream is None:
                self._transfer_stream = torch.cuda.Stream(device=device)

            stream = self._transfer_stream
            # Get pinned buffers from pool (avoids cudaHostAlloc each time)
            hs_pinned = self._pinned_pool.get(
                tuple(hidden_states.shape), hidden_states.dtype,
            )
            tids_pinned = self._pinned_pool.get(
                tuple(token_ids.shape), token_ids.dtype,
            )

            with torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.default_stream(device))
                hs_pinned.copy_(hidden_states, non_blocking=True)
                tids_pinned.copy_(token_ids, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
        else:
            hs_pinned = hidden_states.clone()
            tids_pinned = token_ids.clone()
            event = None

        try:
            self._write_queue.put_nowait({
                "hidden_states": hs_pinned,
                "token_ids": tids_pinned,
                "filename": filename,
                "event": event,
            })
        except queue.Full:
            logger.warning("Write queue full, dropping %s", filename)

    def _writer_loop(self) -> None:
        """Background thread: wait for transfers, write to disk."""
        while not self._shutdown:
            try:
                item = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                event = item.get("event")
                if event is not None:
                    event.synchronize()

                # After event.synchronize(), pinned buffers are stable.
                # Pass directly to safetensors — no extra .clone() needed.
                hs_pinned = item["hidden_states"]
                tids_pinned = item["token_ids"]
                tensors = {
                    "hidden_states": hs_pinned,
                    "token_ids": tids_pinned,
                }
                filepath = item["filename"]

                if self.use_compression:
                    filepath += ".zst"
                    self._write_compressed(tensors, filepath)
                else:
                    safetensors.torch.save_file(tensors, filepath)

                file_size = os.path.getsize(filepath)
                self._total_writes += 1
                self._total_bytes += file_size

                # Return pinned buffers to pool for reuse
                if hs_pinned.is_pinned():
                    self._pinned_pool.put(hs_pinned)
                if tids_pinned.is_pinned():
                    self._pinned_pool.put(tids_pinned)
            except Exception as e:
                logger.error("Error writing %s: %s",
                             item.get("filename", "?"), e)

    def _write_compressed(
        self, tensors: dict[str, torch.Tensor], filepath: str,
    ) -> None:
        """Write safetensors with zstd compression via atomic rename.

        Serializes to bytes in memory, compresses, then writes once.
        No intermediate temp file for the uncompressed data.
        """
        # Serialize to bytes in memory (no disk I/O)
        raw_bytes = safetensors.torch.save(tensors)

        # Create temp file in the same directory as the target so that
        # os.replace() is an atomic same-filesystem rename.
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(filepath), prefix="tmp_hs_",
        )
        try:
            compressed = self._compressor.compress(raw_bytes)
            os.write(fd, compressed)
            os.close(fd)
            fd = -1  # mark as closed
            os.replace(tmp_path, filepath)
        except Exception:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def flush(self, timeout: float = 10.0) -> None:
        """Block until all pending writes complete."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._write_queue.empty():
                return
            time.sleep(0.01)
        logger.warning("Flush timed out after %.1fs", timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully stop the writer thread."""
        logger.info(
            "HiddenStatesWriter shutting down. "
            "Writes: %d, Bytes: %.1f MB",
            self._total_writes,
            self._total_bytes / 1024 / 1024,
        )
        self._shutdown = True
        self._writer_thread.join(timeout=timeout)
