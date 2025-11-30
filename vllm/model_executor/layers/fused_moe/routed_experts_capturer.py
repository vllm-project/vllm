# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
import logging
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from typing import Optional
from unittest.mock import patch

import numpy as np
import torch

from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

LOCK_FILE_PREFIX = "/tmp/vllm_routed_experts"  # Shared lock file path
BUFFER_PREFIX = "vllm_routed_experts_buffer"


def lock_file(fp):
    fcntl.flock(fp, fcntl.LOCK_EX)


def unlock_file(fp):
    fcntl.flock(fp, fcntl.LOCK_UN)


# Global singleton instances (annotated)
_global_experts_capturer: Optional["RoutedExpertsCapturer"] = None
_global_experts_reader: Optional["RoutedExpertsReader"] = None


class RoutedExpertsCapturer(ABC):
    """Abstract interface for capturer (host side)."""

    @staticmethod
    def create(enable: bool) -> "RoutedExpertsCapturer":
        """Create a global singleton instance"""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            raise RuntimeError("Experts capturer already created.")

        if enable:
            _global_experts_capturer = _RoutedExpertsCapturerReal()
        else:
            _global_experts_capturer = _RoutedExpertsCapturerNoop()
        return _global_experts_capturer

    @staticmethod
    def get_instance() -> Optional["RoutedExpertsCapturer"]:
        return _global_experts_capturer

    @abstractmethod
    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        model_config: ModelConfig,
        instance_id: str,
        enable_shared_memory: bool,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_buffer(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_captured_experts(self, indices: np.ndarray) -> None:
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""

    def __init__(self) -> None:
        self._experts_capturer_device_buffer: torch.Tensor | None = None
        self._shm: shared_memory.SharedMemory | None = None
        self._host_buffer_view: np.ndarray | None = None

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        model_config: ModelConfig,
        instance_id: str,
        enable_shared_memory: bool,
    ) -> None:
        if (
            model_config.enable_return_routed_experts
            and self._experts_capturer_device_buffer is None
        ):
            self._experts_capturer_device_buffer = torch.zeros(
                (
                    max_num_batched_tokens,
                    model_config.hf_text_config.num_hidden_layers,
                    model_config.hf_text_config.num_experts_per_tok,
                ),
                dtype=torch.int32,
                device="cuda",
            )

            if enable_shared_memory:
                # Compute required shared memory size
                shape = (
                    max_num_kv_tokens,
                    model_config.hf_text_config.num_hidden_layers,
                    model_config.hf_text_config.num_experts_per_tok,
                )
                self.dest_size = int(np.prod(shape)) * np.dtype(np.int32).itemsize
                self.lock_file = f"{LOCK_FILE_PREFIX}_{instance_id}.lock"
                self.shm_name = f"{BUFFER_PREFIX}_{instance_id}"

                with open(self.lock_file, "wb") as fp:
                    lock_file(fp)
                    try:
                        shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.dest_size)
                    except:
                        shm = shared_memory.SharedMemory(name=self.shm_name, create=False, size=self.dest_size)

                    if shm.size != self.dest_size:
                        logger.warning(f"size not same, unlink shm {self.shm_name} and create again")
                        shm.close()
                        shm.unlink()
                        try:
                            shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.dest_size)
                            logger.info(f"create shm {self.shm_name}")
                        except:
                            shm = shared_memory.SharedMemory(name=self.shm_name, create=False, size=self.dest_size)
                            logger.info(f"link shm {self.shm_name}")

                    self._shm = shm
                    self._host_buffer_view = np.ndarray(
                        shape, dtype=np.int32, buffer=self._shm.buf
                    )
                    # init 0
                    self._host_buffer_view.fill(0)
                    unlock_file(fp)

                # parameterized logging (avoid f-strings in logging)
                logger.debug(
                    "Created shared memory buffer '%s' with shape %s",
                    self._shm.name if self._shm is not None else "None",
                    shape,
                )
            else:
                self._shm = None
                self._host_buffer_view = None

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if self._experts_capturer_device_buffer is None:
            raise RuntimeError("Buffer not initialized.")
        batch_size, num_routed_experts = topk_ids.shape
        # copy into device buffer (ensure shapes are compatible)
        self._experts_capturer_device_buffer[:batch_size, layer_id, :] = topk_ids

    def clear_buffer(self) -> None:
        if self._experts_capturer_device_buffer is not None:
            self._experts_capturer_device_buffer.zero_()

    def save_captured_experts(self, indices: np.ndarray) -> None:
        # Copy the entire batch from GPU to shared memory (via numpy view)
        with open(self.lock_file, "wb+") as fp:
            lock_file(fp)
            try:
                if self._host_buffer_view is not None:
                    num_tokens = len(indices)
                    # Ensure device buffer exists
                    if self._experts_capturer_device_buffer is None:
                        raise RuntimeError("Device buffer not initialized.")
                    data = (
                        self._experts_capturer_device_buffer[:num_tokens, :, :]
                        .cpu()
                        .numpy()
                    )
                    # indices should be valid for host buffer
                    self._host_buffer_view[indices, :, :] = data
            finally:
                unlock_file(fp)

    def __del__(self) -> None:
        """Clean up shared memory"""
        try:
            if self._shm is not None:
                self._shm.close()
                # Only creator should unlink
                self._shm.unlink()
        except Exception:
            # Avoid raising in destructor
            logger.debug("Exception during __del__ cleanup for capturer", exc_info=True)


class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        model_config: ModelConfig,
        instance_id: str,
        enable_shared_memory: bool,
    ) -> None:
        return None

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        return None

    def clear_buffer(self) -> None:
        return None

    def save_captured_experts(self, indices: np.ndarray) -> None:
        return None


class RoutedExpertsReader(ABC):
    """Abstract interface for reader (worker side)."""

    @staticmethod
    def create(enable: bool) -> "RoutedExpertsReader":
        """Create a global singleton instance"""
        global _global_experts_reader
        if _global_experts_reader is not None:
            raise RuntimeError("Experts Reader already created.")

        if enable:
            _global_experts_reader = _RoutedExpertsReaderReal()
        else:
            _global_experts_reader = _RoutedExpertsReaderNoop()
        return _global_experts_reader

    @staticmethod
    def get_instance() -> Optional["RoutedExpertsReader"]:
        if _global_experts_reader is None:
            logger.info("Experts reader not initialized.")
        return _global_experts_reader

    @abstractmethod
    def attach_buffer(
        self, max_num_kv_tokens: int, model_config: ModelConfig, instance_id: str
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_routed_experts(self, indices: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError


class _RoutedExpertsReaderReal(RoutedExpertsReader):
    """Reader class in worker process"""

    def __init__(self) -> None:
        self._shm: shared_memory.SharedMemory | None = None
        self._host_buffer_view: np.ndarray | None = None

    def attach_buffer(
        self, max_num_kv_tokens: int, model_config: ModelConfig, instance_id: str
    ) -> None:
        if self._shm is None:
            shape = (
                max_num_kv_tokens,
                model_config.hf_text_config.num_hidden_layers,
                model_config.hf_text_config.num_experts_per_tok,
            )

            self.lock_file = f"{LOCK_FILE_PREFIX}_{instance_id}.lock"

            # Attach to existing shared memory
            with open(self.lock_file, "rb+") as fp:
                lock_file(fp)
                try:
                    # avoid resource_tracker registering the shared memory
                    with patch(
                        "multiprocessing.resource_tracker.register",
                        lambda *args, **kwargs: None,
                    ):
                        # This will raise if the shared memory doesn't exist
                        self._shm = shared_memory.SharedMemory(
                            name=f"{BUFFER_PREFIX}_{instance_id}"
                        )

                    self._host_buffer_view = np.ndarray(
                        shape, dtype=np.int32, buffer=self._shm.buf
                    )
                finally:
                    unlock_file(fp)

    def get_routed_experts(self, indices: np.ndarray) -> np.ndarray | None:
        """
        Read routed expert data from shared memory for the given request.
        """

        with open(self.lock_file, "rb+") as fp:
            lock_file(fp)
            try:
                if self._host_buffer_view is None:
                    raise RuntimeError("Buffer not attached.")
                # Return a copy to avoid referencing shared memory buffer directly
                return self._host_buffer_view[indices, :, :].copy()
            finally:
                unlock_file(fp)

    def __del__(self) -> None:
        """Only close, do not delete shared memory"""
        try:
            if self._shm is not None:
                self._shm.close()  # Note: reader does not call unlink()
        except Exception:
            logger.debug("Exception during __del__ cleanup for reader", exc_info=True)


class _RoutedExpertsReaderNoop(RoutedExpertsReader):
    def attach_buffer(
        self, max_num_kv_tokens: int, model_config: ModelConfig, instance_id: str
    ) -> None:
        return None

    def get_routed_experts(self, indices: np.ndarray) -> np.ndarray | None:
        return None
