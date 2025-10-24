# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains a new class `MooncakeStore` that allows developers to
think of EC cache transfer operations as putting new EC cache entries
into a remote ECStore-based lookup buffer and getting existing EC caches
from this remote lookup buffer.
"""

import asyncio
import json
import math
import os
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.utils.tensor_memory_pool import (
    InsufficientMemoryError,
    TensorMemoryPool,
)
from vllm.logger import init_logger

DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB
DEFAULT_TENSOR_POOL_SIZE = 1073741824  # 1.0 GiB

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
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    storage_root_dir: str
    transfer_timeout: int
    replica_num: int
    fast_transfer: bool
    fast_transfer_buffer_size: int

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get(
                "global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE
            ),
            local_buffer_size=config.get(
                "local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE
            ),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
            storage_root_dir=config.get("storage_root_dir", ""),
            transfer_timeout=int(config.get("transfer_timeout", 1)),
            replica_num=int(config.get("replica_num", 1)),
            fast_transfer=bool(config.get("fast_transfer", True)),
            fast_transfer_buffer_size=int(
                float(config.get("fast_transfer_buffer_size", 1))
                * DEFAULT_TENSOR_POOL_SIZE
            ),
        )


@dataclass
class ECMooncakeTensorPoolMetadata:
    """
    Metadata element for a buffer in the tensor pool. This stores:

        key: ECMooncakeStore key of (key, value) pair
        addr: addr of the buffer in the tensor pool

    Those elements are maintained for zero-copy put method (fast_transfer mode),
    and evicted by FIFO eviction policy.
    """

    key: str
    addr: int


class ECMooncakeStore:
    """
    Currently, it only supports zero-copy get/put with
    following data path gpu->cpu->cpu->gpu
    TODO: remove by keys, non-blocking
    """

    def __init__(self, vllm_config: "VllmConfig"):
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        try:
            if vllm_config.ec_transfer_config is None:
                raise ValueError("ec_transfer_config must be set for ECConnectorBase")

            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.from_file(
                vllm_config.ec_transfer_config.ec_connector_extra_config[
                    "ec_mooncake_config_file_path"
                ]
            )
            logger.debug("Mooncake Configuration loaded successfully.")

            # Check if storage_root_dir exists and set environment variable
            if (
                self.config.storage_root_dir is not None
                and self.config.storage_root_dir != ""
            ):
                os.environ["MOONCAKE_STORAGE_ROOT_DIR"] = self.config.storage_root_dir
                logger.info(
                    "Set MOONCAKE_STORAGE_ROOT_DIR to: %s", self.config.storage_root_dir
                )

            logger.info("Setting up Mooncake store with parameters:")
            logger.info("  local_hostname: %s", self.config.local_hostname)
            logger.info("  metadata_server: %s", self.config.metadata_server)
            logger.info("  global_segment_size: %s", self.config.global_segment_size)
            logger.info("  local_buffer_size: %s", self.config.local_buffer_size)
            logger.info("  protocol: %s", self.config.protocol)
            logger.info("  device_name: %s", self.config.device_name)
            logger.info(
                "  master_server_address: %s", self.config.master_server_address
            )
            logger.info("  transfer_timeout: %s", self.config.transfer_timeout)
            logger.info("  replica_num: %s", self.config.replica_num)
            logger.info("  fast_transfer: %s", self.config.fast_transfer)
            logger.info(
                "  fast_transfer_buffer_size: %s", self.config.fast_transfer_buffer_size
            )

            self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        # Initialize ReplicateConfig
        self.replica_config = ReplicateConfig()
        self.replica_config.replica_num = self.config.replica_num

        logger.info("MooncakeConnector initialized successfully.")

        # Fast transfer init (Use zero-copy methods of mooncake)
        if self.config.fast_transfer:
            self.tensor_pool = TensorMemoryPool(
                max_block_size=self.config.fast_transfer_buffer_size
            )
            self.store.register_buffer(
                self.tensor_pool.base_address, self.config.fast_transfer_buffer_size
            )
            self.fifo_pool_queue: deque[ECMooncakeTensorPoolMetadata] = deque()

        # Put async init
        # queue of unfinished put requests stored by keys
        self.put_queue: set[str] = set()
        self.put_queue_cv = asyncio.Condition()
        self.put_loop = asyncio.new_event_loop()
        self.put_thread = threading.Thread(
            target=self.put_loop.run_forever, daemon=True
        )
        self.put_thread.start()

    def close(self):
        if self.config.fast_transfer:
            self.store.unregister_buffer(
                self.tensor_pool.base_address, self.config.fast_transfer_buffer_size
            )
            self.tensor_pool.cleanup()

        self.put_loop.call_soon_threadsafe(self.put_loop.stop)
        self.put_thread.join()
        self.put_loop.close()

        self.store.close()
        logger.info("Closed the mooncake store connection")

    def batch_exists(self, keys: list[str]) -> list[bool]:
        if not keys:
            return []
        return self.store.batch_is_exist(keys)

    def metadata_key(self, key: str) -> str:
        # TODO: no guarantee that there is no (k,v) with this key
        return key + "_metadata"

    def get(self, key: str) -> torch.Tensor | None:
        logger.error("Single get operation is not supported. Use batch_get instead.")
        raise NotImplementedError(
            "Single get is not supported. Use batch_get([key]) instead."
        )

    def batch_get(self, keys: list[str]) -> list[torch.Tensor | None]:
        if self.config.fast_transfer:
            return self._zero_copy_batch_get(keys)

        return self._batch_get(keys)

    def _zero_copy_batch_get(self, keys: list[str]) -> list[torch.Tensor | None]:
        if not keys:
            return []

        # Retrieve metadata
        try:
            meta_keys = [self.metadata_key(k) for k in keys]
            meta_bytes = self.store.get_batch(meta_keys)
        except Exception as e:
            logger.error("get_batch for metadata failed: %s", str(e))
            return [None] * len(keys)

        buffer_shapes = []
        buffer_addrs = []
        buffer_dtypes = []
        sizes = []
        exist_ids = []
        for id, meta_byte in enumerate(meta_bytes):
            # non-exist object
            if not meta_byte:
                continue

            exist_ids.append(id)
            meta_out = json.loads(meta_byte.decode("utf-8"))

            # Retrieve metadata (dtype, shape)
            buffer_dtype = getattr(torch, meta_out["dtype"].split(".")[1])
            buffer_shape = tuple(meta_out["shape"])
            element_size = torch.tensor([], dtype=buffer_dtype).element_size()
            num_elem = math.prod(buffer_shape)
            buffer_size = num_elem * element_size

            buffer_addr = self._pool_allocate(buffer_size)
            buffer_addrs.append(buffer_addr)
            buffer_dtypes.append(buffer_dtype)
            sizes.append(buffer_size)
            buffer_shapes.append(buffer_shape)

        # Fill None first and
        # replace valid keys with corresponding buffers
        results = [None] * len(keys)
        try:
            valid_keys = [keys[id] for id in exist_ids]
            read_bytes = self.store.batch_get_into(valid_keys, buffer_addrs, sizes)
        except Exception as e:
            logger.error("batch_get_into failed: %s", str(e))

        # NOTE: should I delay free buffer
        for id, addr, dtype, shape, read_byte in zip(
            exist_ids, buffer_addrs, buffer_dtypes, buffer_shapes, read_bytes
        ):
            if read_byte > 0:
                results[id] = self.tensor_pool.load_tensor(addr, dtype, shape, "cuda")

            self.tensor_pool.free(addr)

        return results

    def _batch_get(self, keys: list[str]) -> list[torch.Tensor | None]:
        try:
            bytes_list = self.store.get_batch(keys)
        except Exception as e:
            logger.error("batch_get_into failed: %s", str(e))
            return [None] * len(keys)

        tensors: list[torch.Tensor | None] = []
        for bytes_data in bytes_list:
            if not bytes_data:
                tensors.append(None)
                continue

            len_meta = int.from_bytes(bytes_data[:4], "big")
            meta = json.loads(bytes_data[4 : 4 + len_meta].decode("utf-8"))
            data = bytes_data[4 + len_meta :]
            arr_loaded = np.frombuffer(data, dtype=meta["serialized_dtype"]).reshape(
                meta["shape"]
            )
            tensor_loaded = torch.from_numpy(arr_loaded)

            if meta["original_dtype"] != meta["serialized_dtype"]:
                tensor_loaded = tensor_loaded.view(
                    getattr(torch, meta["original_dtype"].split(".")[-1])
                )  # e.g., 'torch.bfloat16' -> torch.bfloat16
            tensors.append(tensor_loaded.cuda())

        return tensors

    def put(self, key: str, tensor: torch.Tensor) -> None:
        logger.error("Single put operation is not supported. Use batch_put instead.")
        raise NotImplementedError(
            "Single put is not supported. Use batch_put([key], [tensor]) instead."
        )

    def wait_for_put(self):
        future = asyncio.run_coroutine_threadsafe(
            self._wait_for_put_async(), self.put_loop
        )
        future.result()  # wait until complete

    async def _wait_for_put_async(self):
        async with self.put_queue_cv:
            while self.put_queue:
                await self.put_queue_cv.wait()

    def batch_put(self, keys: list[str], tensors: list[torch.Tensor]) -> None:
        self.put_loop.call_soon_threadsafe(
            lambda: self.put_loop.create_task(self._batch_put_async(keys, tensors))
        )

    async def _batch_put_async(
        self, keys: list[str], tensors: list[torch.Tensor]
    ) -> None:
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

    async def _zero_copy_batch_put(
        self, keys: list[str], tensors: list[torch.Tensor]
    ) -> None:
        if not keys:
            return

        # Prepair metadata
        meta_keys = []
        meta_values = []
        buffer_addrs = []
        buffer_sizes = []
        for key, tensor in zip(keys, tensors):
            buffer_addr = self._pool_store_tensor(tensor)
            self.fifo_pool_queue.append(ECMooncakeTensorPoolMetadata(key, buffer_addr))
            buffer_size = tensor.numel() * tensor.element_size()
            buffer_addrs.append(buffer_addr)
            buffer_sizes.append(buffer_size)

            meta = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            meta_str = json.dumps(meta)
            meta_bytes = meta_str.encode("utf-8")
            key_meta = self.metadata_key(key)
            meta_keys.append(key_meta)
            meta_values.append(meta_bytes)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.put_batch, meta_keys, meta_values, self.replica_config
                ),
                timeout=self.config.transfer_timeout,
            )
        except Exception as e:
            logger.error(
                "Failed to put metadata for keys %s using put_batch with error %s",
                ",".join(keys),
                str(e),
            )

        try:
            # Zero-copy put
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.batch_put_from,
                    keys,
                    buffer_addrs,
                    buffer_sizes,
                    self.replica_config,
                ),
                timeout=self.config.transfer_timeout,
            )
        except Exception as e:
            logger.error(
                "Failed to put keys %s using batch_put_from with error %s",
                ",".join(keys),
                str(e),
            )

    async def _batch_put(self, keys: list[str], tensors: list[torch.Tensor]) -> None:
        bytes_list = []
        for tensor in tensors:
            if tensor.get_device() != -1:
                tensor = tensor.cpu()

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
            len_bytes = len(meta_bytes).to_bytes(4, "big")  # Prefix metadata length
            bytes_list.append(len_bytes + meta_bytes + data_bytes)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.store.put_batch, keys, bytes_list, self.replica_config
                ),
                timeout=self.config.transfer_timeout,
            )
        except Exception as e:
            logger.error(
                "Failed to put keys %s using put_batch with error %s",
                ",".join(keys),
                str(e),
            )

    # ==============================
    # Tensor pool helper functions
    # ==============================

    def _pool_eviction(self) -> None:
        evicted_buffer = self.fifo_pool_queue.popleft()
        self.tensor_pool.free(evicted_buffer.addr)
        key = re.escape(evicted_buffer.key)
        meta_key = re.escape(self.metadata_key(evicted_buffer.key))
        count = self.store.remove_by_regex(f"^(?:{key}|{meta_key})$")
        assert count <= 2

    def _pool_allocate(self, size: int) -> int:
        while True:
            try:
                return self.tensor_pool.allocate(size)
            except InsufficientMemoryError:
                if not self.fifo_pool_queue:
                    raise

                self._pool_eviction()

    def _pool_store_tensor(self, tensor: torch.Tensor) -> int:
        while True:
            try:
                return self.tensor_pool.store_tensor(tensor)
            except InsufficientMemoryError:
                if not self.fifo_pool_queue:
                    raise

                self._pool_eviction()
