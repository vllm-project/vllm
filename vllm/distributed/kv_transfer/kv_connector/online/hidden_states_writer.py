# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import queue
import tempfile
import threading
import time
from collections import deque

import safetensors.torch
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


class PinnedBufferPool:
    """Reusable pinned-memory tensors keyed by (shape, dtype)."""

    def __init__(self, max_per_key: int = 8):
        self.max_per_key = max_per_key
        self.pool: dict[tuple, deque[torch.Tensor]] = {}

    def get(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        key = (shape, dtype)
        if buffers := self.pool.get(key):
            return buffers.popleft()
        return torch.empty(shape, dtype=dtype, pin_memory=True)

    def put(self, tensor: torch.Tensor) -> None:
        key = (tuple(tensor.shape), tensor.dtype)
        buffers = self.pool.setdefault(key, deque())
        if len(buffers) < self.max_per_key:
            buffers.append(tensor)


class HiddenStatesWriter:
    """Non-blocking safetensors writer with async GPU→pinned transfer."""

    def __init__(
        self,
        storage_path: str = "/tmp",
        use_compression: bool = True,
        compression_level: int = 3,
    ):
        self.storage_path = storage_path
        self.compression_enabled = use_compression and ZSTD_AVAILABLE
        self.compression_level = compression_level
        os.makedirs(storage_path, exist_ok=True)

        if use_compression and not ZSTD_AVAILABLE:
            logger.warning(
                "Compression requested but zstandard not installed."
            )

        self.transfer_stream: torch.cuda.Stream | None = None
        self.write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.is_shutdown = False
        self.pinned_pool = PinnedBufferPool()
        self.compressor = None
        if self.compression_enabled:
            self.compressor = zstd.ZstdCompressor(level=compression_level)

        # Per-request decode accumulation buffers (pinned CPU tensors)
        self.decode_hidden_states: dict[str, list[torch.Tensor]] = {}
        self.decode_token_ids: dict[str, list[torch.Tensor]] = {}
        self.decode_events: dict[str, list[torch.cuda.Event]] = {}

        self.total_writes = 0
        self.total_bytes = 0

        self.writer_thread = threading.Thread(
            target=self.writer_loop, daemon=True, name="hs-writer",
        )
        self.writer_thread.start()
        logger.info(
            "HiddenStatesWriter initialized: path=%s, compression=%s",
            storage_path, self.compression_enabled,
        )

    def copy_to_pinned(
        self, hidden_states: torch.Tensor, token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event | None]:
        """Copy GPU tensors to pinned memory on the transfer stream."""
        device = hidden_states.device
        if device.type == "cuda":
            if self.transfer_stream is None:
                self.transfer_stream = torch.cuda.Stream(device=device)
            stream = self.transfer_stream
            hs_pinned = self.pinned_pool.get(
                tuple(hidden_states.shape), hidden_states.dtype,
            )
            tids_pinned = self.pinned_pool.get(
                tuple(token_ids.shape), token_ids.dtype,
            )
            with torch.cuda.stream(stream):
                stream.wait_stream(torch.cuda.default_stream(device))
                hs_pinned.copy_(hidden_states, non_blocking=True)
                tids_pinned.copy_(token_ids, non_blocking=True)
                event = torch.cuda.Event()
                event.record()
            return hs_pinned, tids_pinned, event
        else:
            return hidden_states.clone(), token_ids.clone(), None

    def write_async(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        filename: str,
    ) -> None:
        """Copy to pinned memory and queue for immediate write."""
        hs_pinned, tids_pinned, event = self.copy_to_pinned(
            hidden_states, token_ids,
        )
        try:
            self.write_queue.put_nowait({
                "hidden_states": hs_pinned,
                "token_ids": tids_pinned,
                "filename": filename,
                "event": event,
            })
        except queue.Full:
            # Return pinned buffers to pool to avoid leak
            if hs_pinned.is_pinned():
                self.pinned_pool.put(hs_pinned)
            if tids_pinned.is_pinned():
                self.pinned_pool.put(tids_pinned)
            logger.warning("Write queue full, dropping %s", filename)

    def accumulate_async(
        self,
        req_id: str,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> None:
        """Copy decode token to pinned memory and accumulate per-request."""
        hs_pinned, tids_pinned, event = self.copy_to_pinned(
            hidden_states, token_ids,
        )
        if req_id not in self.decode_hidden_states:
            self.decode_hidden_states[req_id] = []
            self.decode_token_ids[req_id] = []
            self.decode_events[req_id] = []
        self.decode_hidden_states[req_id].append(hs_pinned)
        self.decode_token_ids[req_id].append(tids_pinned)
        if event is not None:
            self.decode_events[req_id].append(event)

    def flush_request(self, req_id: str, filename: str) -> None:
        """Concatenate accumulated decode buffers and queue for write."""
        hs_chunks = self.decode_hidden_states.pop(req_id, None)
        tids_chunks = self.decode_token_ids.pop(req_id, None)
        events = self.decode_events.pop(req_id, None)

        if not hs_chunks:
            return

        if events:
            for e in events:
                e.synchronize()

        hs_concat = torch.cat(hs_chunks, dim=0)
        tids_concat = torch.cat(tids_chunks, dim=0)

        for buf in hs_chunks:
            if buf.is_pinned():
                self.pinned_pool.put(buf)
        for buf in tids_chunks:
            if buf.is_pinned():
                self.pinned_pool.put(buf)

        try:
            self.write_queue.put_nowait({
                "hidden_states": hs_concat,
                "token_ids": tids_concat,
                "filename": filename,
                "event": None,
            })
        except queue.Full:
            logger.warning("Write queue full, dropping decode %s", filename)

    def discard_request(self, req_id: str) -> None:
        """Discard accumulated decode buffers without writing."""
        hs_chunks = self.decode_hidden_states.pop(req_id, None)
        tids_chunks = self.decode_token_ids.pop(req_id, None)
        self.decode_events.pop(req_id, None)
        if hs_chunks:
            for buf in hs_chunks:
                if buf.is_pinned():
                    self.pinned_pool.put(buf)
        if tids_chunks:
            for buf in tids_chunks:
                if buf.is_pinned():
                    self.pinned_pool.put(buf)

    def writer_loop(self) -> None:
        while not self.is_shutdown:
            try:
                item = self.write_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                event = item.get("event")
                if event is not None:
                    event.synchronize()

                tensors = {
                    "hidden_states": item["hidden_states"],
                    "token_ids": item["token_ids"],
                }
                filepath = item["filename"]
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                if self.compression_enabled:
                    filepath += ".zst"
                    self.write_compressed(tensors, filepath)
                else:
                    safetensors.torch.save_file(tensors, filepath)

                self.total_writes += 1
                self.total_bytes += os.path.getsize(filepath)

                for key in ("hidden_states", "token_ids"):
                    t = item[key]
                    if t.is_pinned():
                        self.pinned_pool.put(t)
            except Exception as e:
                logger.error("Error writing %s: %s",
                             item.get("filename", "?"), e)

    def write_compressed(
        self, tensors: dict[str, torch.Tensor], filepath: str,
    ) -> None:
        raw_bytes = safetensors.torch.save(tensors)
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(filepath), prefix="tmp_hs_",
        )
        try:
            compressed = self.compressor.compress(raw_bytes)
            os.write(fd, compressed)
            os.close(fd)
            fd = -1
            os.replace(tmp_path, filepath)
        except Exception:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def flush(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.write_queue.empty():
                return
            time.sleep(0.01)
        logger.warning("Flush timed out after %.1fs", timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        # Discard any remaining decode buffers to free pinned memory
        for req_id in list(self.decode_hidden_states.keys()):
            self.discard_request(req_id)
        # Drain the write queue
        self.flush(timeout=timeout)
        logger.info(
            "HiddenStatesWriter shutting down. "
            "Writes: %d, Bytes: %.1f MB",
            self.total_writes, self.total_bytes / 1024 / 1024,
        )
        self.is_shutdown = True
        self.writer_thread.join(timeout=timeout)
