# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import torch

from vllm.config import ModelConfig
from vllm.distributed import get_tensor_model_parallel_rank

logger = logging.getLogger(__name__)

# Constants
_TMP_DIR = tempfile.gettempdir()
_LOCK_FILE_PREFIX = os.path.join(_TMP_DIR, "vllm_routed_experts")
_BUFFER_PREFIX = "vllm_routed_experts_buffer"
_MMAP_FILE_PREFIX = os.path.join(_TMP_DIR, "vllm_routed_experts_mmap")
_SHM_DIR = "/dev/shm"
_SHM_FREE_RATIO = 1.1
_UINT16_MAX = np.iinfo(np.uint16).max

# Global singleton instances
_global_experts_capturer: RoutedExpertsCapturer | None = None
_global_experts_reader: RoutedExpertsReader | None = None


@contextmanager
def _file_lock(lock_file: str, mode: str = "wb+") -> Generator[None, None, None]:
    """Context manager for file-based locking."""
    with open(lock_file, mode) as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)


def _create_or_attach_shared_memory(
    name: str, size: int, lock_file: str
) -> shared_memory.SharedMemory:
    """Create or attach to shared memory with proper locking."""
    # Ensure lock file exists before acquiring lock
    with open(lock_file, "wb"):
        pass

    with _file_lock(lock_file):
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=name, create=False, size=size)

        if shm.size != size:
            logger.warning(
                "Shared memory %s size mismatch; recreating",
                name,
            )
            shm.close()
            shm.unlink()
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                logger.info("Created shared memory %s", name)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name, create=False, size=size)
                logger.info("Linked to existing shared memory %s", name)

    return shm


def _get_shm_free_bytes() -> int | None:
    try:
        stat = os.statvfs(_SHM_DIR)
    except FileNotFoundError:
        return None
    return stat.f_bsize * stat.f_bavail


def _should_use_shared_memory(size: int) -> bool:
    free_bytes = _get_shm_free_bytes()
    if free_bytes is None:
        logger.warning(
            "Shared memory directory %s not found; "
            "falling back to file-backed buffer.",
            _SHM_DIR,
        )
        return False
    if free_bytes < int(size * _SHM_FREE_RATIO):
        logger.warning(
            "Insufficient /dev/shm free space (%d bytes) for routed experts "
            "buffer (%d bytes); falling back to file-backed buffer.",
            free_bytes,
            size,
        )
        return False
    return True


def _create_or_attach_mmap(
    path: str,
    size: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
    lock_file: str,
) -> np.memmap:
    # Ensure lock file exists before acquiring lock
    with open(lock_file, "wb"):
        pass

    with _file_lock(lock_file):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            current_size = os.path.getsize(path)
        except FileNotFoundError:
            current_size = -1
        if current_size != size:
            with open(path, "wb") as fp:
                fp.truncate(size)
        return np.memmap(path, dtype=dtype, mode="r+", shape=shape)


def _close_memmap(mem: np.memmap) -> None:
    try:
        mem.flush()
    except Exception:
        logger.debug("Exception during memmap flush", exc_info=True)
    try:
        mem._mmap.close()  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Exception during memmap close", exc_info=True)


class RoutedExpertsCapturer:
    """
    Capturer for routed experts with device and optional shared memory buffer.

    This class captures expert routing decisions during model forward passes
    and optionally stores them in shared memory for cross-process access.
    """

    _instance: RoutedExpertsCapturer | None = None

    def __init__(self) -> None:
        self._device_buffer: torch.Tensor | None = None
        self._shm: shared_memory.SharedMemory | None = None
        self._host_buffer_view: np.ndarray | None = None
        self._mmap: np.memmap | None = None
        self._lock_file: str | None = None
        self._shm_name: str | None = None
        self._mmap_path: str | None = None
        self._use_shm: bool = True
        self._buffer_shape: tuple[int, int, int] | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._pinned_buffers: list[torch.Tensor] = []
        self._buffer_events: list[torch.cuda.Event] = []
        self._next_buffer_idx: int = 0
        self._buffer_dtype_torch: torch.dtype = torch.int32
        self._buffer_dtype_np: np.dtype = np.dtype(np.int32)

    @classmethod
    def create(cls) -> RoutedExpertsCapturer:
        """Create a global singleton instance."""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            raise RuntimeError("Experts capturer already created.")

        _global_experts_capturer = cls()
        return _global_experts_capturer

    @staticmethod
    def get_instance() -> RoutedExpertsCapturer | None:
        """Get the global singleton instance."""
        return _global_experts_capturer

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        model_config: ModelConfig,
        instance_id: str,
    ) -> None:
        """
        Initialize the device buffer and optionally shared memory buffer.

        Args:
            max_num_batched_tokens: Maximum number of tokens in a batch.
            max_num_kv_tokens: Maximum number of KV tokens for shared memory.
            model_config: Model configuration containing layer and expert info.
            instance_id: Unique identifier for the shared memory buffer.
        """

        if self._device_buffer is not None:
            raise RuntimeError("Device buffer has already been initialized")

        hf_config = model_config.hf_text_config
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok
        self._buffer_dtype_torch, self._buffer_dtype_np = (
            self._select_buffer_dtype(model_config)
        )

        # Initialize device buffer
        self._device_buffer = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts_per_tok),
            dtype=self._buffer_dtype_torch,
            device="cuda",
        )

        if get_tensor_model_parallel_rank() != 0:
            return

        # Initialize shared memory
        shape = (max_num_kv_tokens, num_layers, num_experts_per_tok)
        buffer_size = int(np.prod(shape)) * self._buffer_dtype_np.itemsize

        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}.lock"
        self._shm_name = f"{_BUFFER_PREFIX}_{instance_id}"
        self._mmap_path = f"{_MMAP_FILE_PREFIX}_{instance_id}.dat"

        self._use_shm = _should_use_shared_memory(buffer_size)
        if self._use_shm:
            self._shm = _create_or_attach_shared_memory(
                self._shm_name, buffer_size, self._lock_file
            )
            self._host_buffer_view = np.ndarray(
                shape, dtype=self._buffer_dtype_np, buffer=self._shm.buf
            )
        else:
            self._mmap = _create_or_attach_mmap(
                path=self._mmap_path,
                size=buffer_size,
                shape=shape,
                dtype=self._buffer_dtype_np,
                lock_file=self._lock_file,
            )
            self._host_buffer_view = self._mmap
        self._buffer_shape = shape
        if self._use_shm:
            self._host_buffer_view.fill(0)
        self._init_async_copy_buffers(
            max_num_batched_tokens, num_layers, num_experts_per_tok
        )

        if self._use_shm and self._shm is not None:
            logger.debug(
                "Created shared memory buffer '%s' with shape %s",
                self._shm.name,
                shape,
            )
        else:
            logger.debug(
                "Created mmap buffer '%s' with shape %s",
                self._mmap_path,
                shape,
            )

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """
        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")

        if layer_id >= self._device_buffer.shape[1]:
            return

        batch_size = topk_ids.shape[0]
        self._device_buffer[:batch_size, layer_id, :] = topk_ids

    def clear_buffer(self) -> None:
        """Clear the device buffer."""
        if self._device_buffer is not None:
            self._device_buffer.zero_()

    def save_captured_experts(self, indices: np.ndarray) -> None:
        """
        Save captured experts from device buffer to shared memory.

        Args:
            indices: Array of indices indicating where to store the data.
        """
        if get_tensor_model_parallel_rank() != 0:
            return
        if self._lock_file is None:
            raise RuntimeError("Shared memory not initialized.")
        if self._host_buffer_view is None:
            return
        if self._device_buffer is None:
            raise RuntimeError("Device buffer not initialized.")

        indices = np.asarray(indices)
        num_tokens = len(indices)
        if num_tokens == 0:
            return

        if not self._pinned_buffers or self._copy_stream is None:
            data = self._device_buffer[:num_tokens, :, :].cpu().numpy()
            self._write_to_host(indices, data)
            return

        buf_idx = self._next_buffer_idx
        self._next_buffer_idx = (self._next_buffer_idx + 1) % len(
            self._pinned_buffers
        )
        buf = self._pinned_buffers[buf_idx]
        evt = self._buffer_events[buf_idx]
        with torch.cuda.stream(self._copy_stream):
            buf[:num_tokens].copy_(self._device_buffer[:num_tokens], non_blocking=True)
            evt.record(self._copy_stream)
        evt.synchronize()
        data = buf[:num_tokens].numpy()
        self._write_to_host(indices, data)

    def _write_to_host(self, indices: np.ndarray, data: np.ndarray) -> None:
        if self._host_buffer_view is None:
            return
        if indices.size == 0:
            return

        if np.any(indices < 0):
            valid_mask = indices >= 0
            indices = indices[valid_mask]
            data = data[valid_mask]
            if indices.size == 0:
                return

        max_index = int(indices.max())
        if max_index >= self._host_buffer_view.shape[0]:
            if self._use_shm:
                logger.warning(
                    "Routed experts buffer too small for index %d with shared "
                    "memory; falling back to mmap and resizing.",
                    max_index,
                )
                # Switch to mmap so we can grow the buffer.
                self._use_shm = False
                if self._shm is not None:
                    try:
                        self._shm.close()
                    except Exception:
                        logger.debug(
                            "Exception during shm close for capturer", exc_info=True
                        )
                    finally:
                        self._shm = None
                self._mmap = None
                self._host_buffer_view = None

            if not self._use_shm:
                self._resize_mmap_buffer(
                    required_tokens=max_index + 1,
                    num_layers=data.shape[1],
                    num_experts=data.shape[2],
                )

        with _file_lock(self._lock_file):
            self._host_buffer_view[indices, :, :] = data

    def _init_async_copy_buffers(
        self, max_num_batched_tokens: int, num_layers: int, num_experts: int
    ) -> None:
        if get_tensor_model_parallel_rank() != 0:
            return
        if self._copy_stream is not None:
            return
        self._copy_stream = torch.cuda.Stream()
        self._pinned_buffers = [
            torch.empty(
                (max_num_batched_tokens, num_layers, num_experts),
                dtype=self._buffer_dtype_torch,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self._buffer_events = [torch.cuda.Event() for _ in range(2)]

    def _resize_mmap_buffer(
        self, required_tokens: int, num_layers: int, num_experts: int
    ) -> None:
        if self._mmap_path is None or self._lock_file is None:
            raise RuntimeError("Mmap path or lock file not initialized.")

        old_tokens = self._buffer_shape[0] if self._buffer_shape else 0
        if required_tokens <= old_tokens:
            return

        growth_tokens = max(int(old_tokens * 1.5), old_tokens + 1)
        new_tokens = max(required_tokens, growth_tokens)
        new_shape = (new_tokens, num_layers, num_experts)
        buffer_size = int(np.prod(new_shape)) * self._buffer_dtype_np.itemsize
        if self._mmap is not None:
            _close_memmap(self._mmap)
            self._mmap = None
        self._mmap = _create_or_attach_mmap(
            path=self._mmap_path,
            size=buffer_size,
            shape=new_shape,
            dtype=self._buffer_dtype_np,
            lock_file=self._lock_file,
        )
        self._host_buffer_view = self._mmap
        # Avoid zero-filling for mmap; new regions are logically zeroed by OS.
        self._buffer_shape = new_shape
        logger.info(
            "Expanded routed experts buffer to %d tokens.",
            new_tokens,
        )

    def cleanup(self) -> None:
        """Explicitly clean up shared memory resources."""
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                logger.debug("Exception during cleanup for capturer", exc_info=True)
            finally:
                self._shm = None
        if self._mmap is not None:
            _close_memmap(self._mmap)
            self._mmap = None
        if self._mmap_path is not None:
            try:
                os.remove(self._mmap_path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug("Exception during mmap cleanup", exc_info=True)

    def __del__(self) -> None:
        """Clean up shared memory on destruction."""
        self.cleanup()

    @staticmethod
    def _select_buffer_dtype(model_config: ModelConfig) -> tuple[torch.dtype, np.dtype]:
        hf_config = model_config.hf_text_config
        max_experts = getattr(
            hf_config,
            "n_routed_experts",
            getattr(
                hf_config,
                "num_experts",
                getattr(hf_config, "num_local_experts", None),
            ),
        )
        if max_experts is not None and max_experts > _UINT16_MAX:
            logger.warning(
                "Routed experts count %d exceeds uint16 capacity; using int32.",
                max_experts,
            )
            return torch.int32, np.dtype(np.int32)
        return torch.uint16, np.dtype(np.uint16)


class RoutedExpertsReader:
    """
    Reader for routed experts from shared memory.

    This class attaches to shared memory created by RoutedExpertsCapturer
    and reads expert routing decisions.
    """

    _instance: RoutedExpertsReader | None = None

    def __init__(self) -> None:
        self._shm: shared_memory.SharedMemory | None = None
        self._host_buffer_view: np.ndarray | None = None
        self._mmap: np.memmap | None = None
        self._lock_file: str | None = None
        self._mmap_path: str | None = None
        self._use_shm: bool = True
        self._num_layers: int | None = None
        self._num_experts: int | None = None
        self._buffer_dtype_np: np.dtype = np.dtype(np.int32)

    @classmethod
    def create(cls) -> RoutedExpertsReader:
        """Create a global singleton instance."""
        global _global_experts_reader
        if _global_experts_reader is not None:
            raise RuntimeError("Experts reader already created.")

        _global_experts_reader = cls()
        return _global_experts_reader

    @staticmethod
    def get_instance() -> RoutedExpertsReader | None:
        """Get the global singleton instance."""
        if _global_experts_reader is None:
            logger.info("Experts reader not initialized.")
        return _global_experts_reader

    def attach_buffer(
        self,
        max_num_kv_tokens: int,
        model_config: ModelConfig,
        instance_id: str,
    ) -> None:
        """
        Attach to an existing shared memory buffer.

        Args:
            max_num_kv_tokens: Maximum number of KV tokens.
            model_config: Model configuration.
            instance_id: Unique identifier for the shared memory buffer.
        """
        if self._shm is not None:
            logger.warning("Already attached to shared memory buffer.")
            return  # Already attached

        hf_config = model_config.hf_text_config
        shape = (
            max_num_kv_tokens,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
        )
        _, self._buffer_dtype_np = RoutedExpertsCapturer._select_buffer_dtype(
            model_config
        )

        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}.lock"
        shm_name = f"{_BUFFER_PREFIX}_{instance_id}"
        self._mmap_path = f"{_MMAP_FILE_PREFIX}_{instance_id}.dat"
        buffer_size = int(np.prod(shape)) * self._buffer_dtype_np.itemsize
        self._use_shm = _should_use_shared_memory(buffer_size)
        self._num_layers = shape[1]
        self._num_experts = shape[2]

        if self._use_shm:
            with _file_lock(self._lock_file, mode="rb+"):
                # Avoid resource_tracker registering the shared memory
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    self._shm = shared_memory.SharedMemory(name=shm_name)

                self._host_buffer_view = np.ndarray(
                    shape, dtype=self._buffer_dtype_np, buffer=self._shm.buf
                )
        else:
            self._mmap = _create_or_attach_mmap(
                path=self._mmap_path,
                size=buffer_size,
                shape=shape,
                dtype=self._buffer_dtype_np,
                lock_file=self._lock_file,
            )
            self._host_buffer_view = self._mmap

    def get_routed_experts(self, indices: np.ndarray) -> np.ndarray:
        """
        Read routed expert data from shared memory.

        Args:
            indices: Array of indices to read.

        Returns:
            Copy of the expert routing data for the given indices.
        """
        if self._host_buffer_view is None:
            raise RuntimeError("Buffer not attached. Call attach_buffer() first.")
        if self._lock_file is None:
            raise RuntimeError("Lock file not initialized.")
        indices = np.asarray(indices)
        if indices.size == 0:
            return self._host_buffer_view[:0, :, :].copy()

        max_index = int(indices.max())
        if max_index >= self._host_buffer_view.shape[0]:
            if self._mmap_path is not None and not self._use_shm:
                self._ensure_mmap_capacity(max_index + 1)
            else:
                raise RuntimeError(
                    "Routed experts buffer too small for requested indices."
                )

        with _file_lock(self._lock_file, mode="rb+"):
            return self._host_buffer_view[indices, :, :].copy()

    def _ensure_mmap_capacity(self, required_tokens: int) -> None:
        if self._mmap_path is None or self._lock_file is None:
            raise RuntimeError("Mmap path or lock file not initialized.")
        if self._num_layers is None or self._num_experts is None:
            raise RuntimeError("Buffer shape not initialized.")

        tokens_per_entry = self._num_layers * self._num_experts
        try:
            file_size = os.path.getsize(self._mmap_path)
        except FileNotFoundError:
            file_size = 0
        current_tokens = file_size // (
            tokens_per_entry * self._buffer_dtype_np.itemsize
        )
        if (
            required_tokens <= current_tokens
            and self._host_buffer_view is not None
            and self._host_buffer_view.shape[0] >= current_tokens
        ):
            return

        new_tokens = max(required_tokens, current_tokens)
        new_shape = (new_tokens, self._num_layers, self._num_experts)
        buffer_size = int(np.prod(new_shape)) * self._buffer_dtype_np.itemsize
        if self._mmap is not None:
            _close_memmap(self._mmap)
            self._mmap = None
        self._mmap = _create_or_attach_mmap(
            path=self._mmap_path,
            size=buffer_size,
            shape=new_shape,
            dtype=self._buffer_dtype_np,
            lock_file=self._lock_file,
        )
        self._host_buffer_view = self._mmap

    def cleanup(self) -> None:
        """Explicitly clean up resources (close without unlink)."""
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                logger.debug("Exception during cleanup for reader", exc_info=True)
            finally:
                self._shm = None
        if self._mmap is not None:
            _close_memmap(self._mmap)
            self._mmap = None

    def __del__(self) -> None:
        """Close shared memory on destruction (do not unlink)."""
        self.cleanup()
