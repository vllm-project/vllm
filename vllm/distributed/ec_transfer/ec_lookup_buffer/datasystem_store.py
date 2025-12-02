# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Datasystem ECMooncakeStore adaptor.
This adaptor provides an interface to interact with Mooncake storage for vLLM,
supporting both regular and zero-copy data transfer modes.
"""

import asyncio
import json
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.utils import split_host_port

METADATA_SIZE_LEN = 4

# View map for unsupported dtypes (add more as needed, e.g., for future types)
VIEW_MAP = {
    torch.bfloat16: torch.uint16,
    torch.complex32: torch.uint32,
    torch.float8_e4m3fn: torch.uint8,
    torch.float8_e4m3fnuz: torch.uint8,
    torch.float8_e5m2: torch.uint8,
    torch.float8_e5m2fnuz: torch.uint8,
    torch.float8_e8m0fnu: torch.uint8,
}

logger = init_logger(__name__)


@dataclass
class DatasystemStoreConfig:
    """Configuration class for Datasystem ECMooncakeStore"""
    fast_transfer: bool  # Whether to use zero-copy fast transfer mode
    transfer_timeout: int  # Timeout (seconds) for data transfer operations
    ds_worker_addr: str  # Address (ip:port) of datasystem worker

    @classmethod
    def from_env(cls) -> "DatasystemStoreConfig":
        """
        Create DatasystemStoreConfig instance from environment variables.
        Reads:
            - FAST_TRANSFER: Enable fast zero-copy transfer
            - TRANSFER_TIMEOUT: Transfer timeout in seconds
            - DS_WORKER_ADDR: Datasystem worker address
        Returns:
            DatasystemStoreConfig: Initialized config object
        """
        return cls(
            fast_transfer=(os.getenv("FAST_TRANSFER", "false").lower()
                           in ("true", "1", "yes")),
            transfer_timeout=int(os.getenv("TRANSFER_TIMEOUT", "1")),
            ds_worker_addr=os.getenv("DS_WORKER_ADDR", "127.0.0.1:31501"),
        )


class ECMooncakeStore:
    """
    Adaptor for Mooncake storage connector integration with vLLM.
    Provides key-value storage operations with support for:
    - Regular data transfer (serialization/deserialization)
    - Zero-copy fast transfer (direct device-to-host/host-to-device)
    - Asynchronous batch put operations with queue management
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        """
        Initialize ECMooncakeStore adaptor.
        Args:
            vllm_config: vLLM configuration
        Raises:
            ImportError: If datasystem library is not installed
            ValueError: If ec_transfer_config is not set in vllm_config
            Exception: For other initialization errors
        """
        try:
            # Import datasystem clients (lazy import to avoid dependency issues)
            from datasystem.kv_client import KVClient, SetParam, WriteMode
        except ImportError as e:
            raise ImportError(
                "Please install yuanrong-datasystem at "
                "https://gitee.com/openeuler/yuanrong-datasystem "
                "to run vLLM with DatasystemStore.") from e

        try:
            if vllm_config.ec_transfer_config is None:
                raise ValueError(
                    "ec_transfer_config must be set for ECConnectorBase")

            self.config = DatasystemStoreConfig.from_env()

            ip, port = split_host_port(self.config.ds_worker_addr)

            # Use local rank as device ID for datasystem client
            device = get_world_group().local_rank

            # Configure write parameters (disable L2 cache eviction)
            self.set_param = SetParam()
            self.set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

            # Initialize fast transfer client if enabled
            if self.config.fast_transfer:
                from datasystem.ds_tensor_client import DsTensorClient
                self.ds_tensor_client = DsTensorClient(ip, port, device)
                self.ds_tensor_client.init()

            # Initialize regular KV client
            self.kv_client = KVClient(ip, port)
            self.kv_client.init()

            logger.info(
                "Datasystem Client initialized successfully. "
                "IP: %s, Port: %d, Device: %d", ip, port, device)

        except Exception as e:
            logger.error("Failed to initialize the Datasystem Client: %s",
                         str(e))
            raise

        # Initialize async put queue and event loop infrastructure
        # queue of unfinished put requests stored by keys
        self.put_queue: set[str] = set()  # Track pending put keys
        self.put_queue_cv = asyncio.Condition()  # Condition var for queue sync
        self.put_loop = asyncio.new_event_loop(
        )  # Dedicated loop for async puts
        self.put_thread = threading.Thread(  # Thread to run async loop
            target=self.put_loop.run_forever,
            daemon=True)
        self.put_thread.start()

    def batch_exists(self, keys: list[str]) -> list[bool]:
        """
        Check if multiple keys exist in the datasystem.
        Args:
            keys: List of keys to check existence for
        Returns:
            list[bool]: List of booleans indicating existence
        """
        if not keys:
            return []

        if self.config.fast_transfer:
            return self.ds_tensor_client.exist(keys)

        return self.kv_client.exist(keys)

    def metadata_key(self, key: str) -> str:
        """
        Generate metadata key for a given data key.
        Args:
            key: Original data key
        Returns:
            str: Metadata key (original key + "_metadata")
        """
        return key + "_metadata"

    def batch_get(self, keys: list[str],
                  device) -> list[Optional[torch.Tensor]]:
        """
        Retrieve multiple tensors from datasystem by keys.
        Automatically routes to zero-copy or regular get based on config.
        Args:
            keys: List of keys to retrieve
            device: Target device (torch.device) to place tensors on
        Returns:
            list[Optional[torch.Tensor]]: List of tensors
        """
        if not keys:
            return []

        if self.config.fast_transfer:
            return self._zero_copy_batch_get(keys, device)

        return self._batch_get(keys, device)

    def _zero_copy_batch_get(self, keys: list[str],
                             device) -> list[Optional[torch.Tensor]]:
        """
        Zero-copy batch get implementation (host-to-device direct transfer).
        Retrieves metadata first, then transfers data directly.
        Args:
            keys: List of keys to retrieve
            device: Target device for tensors
        Returns:
            list[Optional[torch.Tensor]]: List of tensors
        """
        # Retrieve metadata for all keys first
        try:
            meta_keys = [self.metadata_key(k) for k in keys]
            meta_bytes = self.kv_client.get(meta_keys)
        except Exception as e:
            logger.error("Failed to zero_copy_batch_get for metadata: %s",
                         str(e))
            return [None] * len(keys)

        tensors: list[Optional[torch.Tensor]] = []
        valid_keys: list[str] = []
        valid_tensors: list[torch.Tensor] = []

        # Process metadata and allocate tensors
        for key, meta_byte in zip(keys, meta_bytes):
            # Skip non-existent keys
            if not meta_byte:
                tensors.append(None)
                continue

            try:
                meta_out = json.loads(meta_byte.decode("utf-8"))
                buffer_dtype = getattr(torch, meta_out["dtype"].split(".")[1])
                buffer_shape = tuple(meta_out["shape"])
                tensor = torch.empty(buffer_shape,
                                     dtype=buffer_dtype,
                                     device=device)
                tensors.append(tensor)
                valid_keys.append(key)
                valid_tensors.append(tensor)

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.error("Failed to parse metadata or create tensor: %s",
                             str(e))
                tensors.append(None)

        # Transfer data to pre-allocated tensors (zero-copy)
        try:
            if valid_keys:
                self.ds_tensor_client.mget_h2d(valid_keys, valid_tensors)
        except Exception as e:
            logger.error("Failed to zero_copy_batch_get for tensors: %s",
                         str(e))
            return [None] * len(keys)

        return tensors

    def _batch_get(self, keys: list[str],
                   device) -> list[Optional[torch.Tensor]]:
        """
        Regular batch get implementation (serialized data transfer).
        Retrieves serialized bytes, then converts to torch tensors.
        Args:
            keys: List of keys to retrieve
            device: Target device for tensors
        Returns:
            list[Optional[torch.Tensor]]: List of tensors
        """
        try:
            bytes_list = self.kv_client.get(keys)
        except Exception as e:
            logger.error("Failed to batch_get for bytes_list of tensors: %s",
                         str(e))
            return [None] * len(keys)

        tensors: list[Optional[torch.Tensor]] = []
        for bytes_data in bytes_list:
            # Skip non-existent keys
            if not bytes_data:
                tensors.append(None)
                continue

            # Parse metadata
            len_meta = int.from_bytes(bytes_data[:METADATA_SIZE_LEN], "big")
            meta = json.loads(bytes_data[METADATA_SIZE_LEN:METADATA_SIZE_LEN +
                                         len_meta].decode("utf-8"))
            data = bytes_data[METADATA_SIZE_LEN + len_meta:]
            arr_loaded = np.frombuffer(
                data, dtype=meta["serialized_dtype"]).reshape(meta["shape"])
            tensor_loaded = torch.from_numpy(arr_loaded)

            # Convert back to original dtype if view was used for serialization
            if meta["original_dtype"] != meta["serialized_dtype"]:
                tensor_loaded = tensor_loaded.view(
                    getattr(torch, meta["original_dtype"].split(".")[-1]))

            tensors.append(tensor_loaded.to(device))

        return tensors

    def wait_for_put(self):
        """
        Block until all pending asynchronous put operations are completed.
        Uses asyncio to wait for put queue to be empty.
        """
        future = asyncio.run_coroutine_threadsafe(self._wait_for_put_async(),
                                                  self.put_loop)
        future.result()  # Block until async wait completes

    async def _wait_for_put_async(self):
        """
        Async helper to wait for all pending put operations to complete.
        Uses condition variable to wait until put queue is empty.
        """
        async with self.put_queue_cv:
            while self.put_queue:
                await self.put_queue_cv.wait()

    def batch_put(self, keys: list[str], tensors: list[torch.Tensor]) -> None:
        """
        Submit asynchronous batch put operation for multiple key-tensor pairs.
        Routes to zero-copy or regular put based on config.
        Args:
            keys: List of keys to store
            tensors: List of torch tensors to store
        """
        if not keys:
            return

        # Submit async task to dedicated put loop
        self.put_loop.call_soon_threadsafe(lambda: self.put_loop.create_task(
            self._batch_put_async(keys, tensors)))

    async def _batch_put_async(self, keys: list[str],
                               tensors: list[torch.Tensor]) -> None:
        """
        Async batch put implementation with queue management.
        Tracks pending keys in put queue and notifies waiters when complete.
        Args:
            keys: List of keys to store
            tensors: List of torch tensors to store
        """
        # Add keys to pending queue
        async with self.put_queue_cv:
            self.put_queue.update(keys)

        try:
            if self.config.fast_transfer:
                await self._zero_copy_batch_put(keys, tensors)
            else:
                await self._batch_put(keys, tensors)
        finally:
            async with self.put_queue_cv:
                self.put_queue.difference_update(keys)
                if not self.put_queue:
                    self.put_queue_cv.notify()

    async def _zero_copy_batch_put(self, keys: list[str],
                                   tensors: list[torch.Tensor]) -> None:
        """
        Zero-copy batch put implementation (device-to-host direct transfer).
        Stores metadata first, then transfers tensor data directly.
        Args:
            keys: List of keys to store
            tensors: List of torch tensors to store
        """
        # Prepare metadata for each tensor
        meta_keys = []
        meta_values = []
        for key, tensor in zip(keys, tensors):
            meta = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            meta_str = json.dumps(meta)
            meta_bytes = meta_str.encode("utf-8")
            key_meta = self.metadata_key(key)
            meta_keys.append(key_meta)
            meta_values.append(meta_bytes)

        # Store metadata with timeout
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.kv_client.mset, meta_keys, meta_values,
                                  self.set_param.write_mode),
                timeout=self.config.transfer_timeout,
            )
        except Exception as e:
            logger.error(
                "Failed to put metadata for keys %s with error %s",
                ",".join(keys),
                str(e),
            )
            return

        # Store tensor data via zero-copy transfer (device-to-host)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.ds_tensor_client.mset_d2h, keys,
                                  tensors, self.set_param),
                timeout=self.config.transfer_timeout,
            )
        except Exception as e:
            logger.error(
                "Failed to put tensors for keys %s with error %s",
                ",".join(keys),
                str(e),
            )

    async def _batch_put(self, keys: list[str],
                         tensors: list[torch.Tensor]) -> None:
        """
        Regular batch put implementation (serialized data transfer).
        Serializes tensors to bytes and stores in KV store.
        Handles dtype conversion for unsupported types using VIEW_MAP.
        Args:
            keys: List of keys to store
            tensors: List of torch tensors to store
        """
        bytes_list = []
        for tensor in tensors:
            # Move tensor to CPU if on GPU (required for serialization)
            if tensor.get_device() != -1:
                tensor = tensor.cpu()

            # Track original dtype and handle unsupported dtypes
            original_dtype_str = str(tensor.dtype)
            if tensor.dtype in VIEW_MAP:
                view_tensor = tensor.view(VIEW_MAP[tensor.dtype])
                arr = view_tensor.numpy()
                serialized_dtype_str = str(arr.dtype)
            else:
                arr = tensor.numpy()
                serialized_dtype_str = original_dtype_str

            data_bytes = arr.tobytes()
            meta = {
                "shape": list(arr.shape),
                "original_dtype": original_dtype_str,
                "serialized_dtype": serialized_dtype_str,
            }
            meta_bytes = json.dumps(meta).encode("utf-8")
            len_bytes = len(meta_bytes).to_bytes(METADATA_SIZE_LEN, "big")
            bytes_list.append(len_bytes + meta_bytes + data_bytes)

        # Store serialized bytes with timeout
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.kv_client.mset, keys, bytes_list,
                                  self.set_param.write_mode),
                timeout=self.config.transfer_timeout,
            )
        except Exception as e:
            logger.error(
                "Failed to put bytes_list for keys %s with error %s",
                ",".join(keys),
                str(e),
            )
