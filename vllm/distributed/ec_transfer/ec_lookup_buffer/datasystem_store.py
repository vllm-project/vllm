# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Datasystem ECMooncakeStore adaptor.
"""

import json
import os
from queue import Queue
from typing import Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.utils import split_host_port

logger = init_logger(__name__)


class ECMooncakeStore:
    """
    Adaptor for Mooncake storage.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        try:
            from datasystem.ds_tensor_client import DsTensorClient
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

            ds_worker_addr = os.getenv("DS_WORKER_ADDR", "127.0.0.1:31501")
            ip, port = split_host_port(ds_worker_addr)

            # Get local rank as device ID
            device = get_world_group().local_rank

            # Setup parameters
            self._set_param = SetParam()
            self._set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

            # Initialize clients
            self._ds_tensor_client = DsTensorClient(ip, port, device)
            self._ds_tensor_client.init()

            self._kv_client = KVClient(ip, port)
            self._kv_client.init()

            logger.info(
                "DatasystemStore initialized successfully. "
                "DS_WORKER_ADDR: %s, IP: %s, Port: %d, Device: %d",
                ds_worker_addr, ip, port, device)

        except Exception as e:
            logger.error(
                "An error occurred while initializing DatasystemStore: %s", e)
            raise

        # Queue for handling async put futures
        self._put_queue: Queue = Queue()

    def batch_get(
            self,
            keys: list[str],
            device: Optional[torch.device] = None
    ) -> list[Optional[torch.Tensor]]:
        """
        Retrieves a batch of tensors from the store.
        """
        if not keys:
            return []

        meta_keys = [f"{key}_meta" for key in keys]
        try:
            meta_bytes_list = self._kv_client.get(meta_keys)
        except Exception as e:
            logger.error("batch_get metadata failed: %s", e)
            return [None] * len(keys)

        tensors: list[Optional[torch.Tensor]] = []

        # Pre-allocate empty tensors based on metadata
        for meta_bytes_data in meta_bytes_list:
            if not meta_bytes_data:
                tensors.append(None)
                continue

            try:
                meta = json.loads(meta_bytes_data.decode("utf-8"))
                shape = tuple(meta["shape"])
                dtype_str = meta["dtype"]
                tensors.append(
                    torch.empty(size=shape,
                                dtype=getattr(torch, dtype_str),
                                device=device))
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.error("Failed to parse metadata or create tensor: %s",
                             e)
                tensors.append(None)

        # Fill the allocated tensors with data from the store
        try:
            self._ds_tensor_client.mget_h2d(keys, tensors)
        except Exception as e:
            logger.error("batch_get mget_h2d failed: %s", e)
            # Note: Tensors might be partially filled or garbage if this fails

        return tensors

    def batch_put(self, keys: list[str], tensors: list[torch.Tensor]) -> None:
        """
        Stores a batch of tensors asynchronously.
        """
        if not keys:
            return

        meta_bytes_list = []
        meta_keys = [f"{key}_meta" for key in keys]

        for tensor in tensors:
            meta = {
                "shape": list(tensor.shape),
                # Robustly get dtype string (e.g., 'float16')
                "dtype": str(tensor.dtype).split(".")[-1]
            }
            meta_bytes_list.append(json.dumps(meta).encode("utf-8"))

        # 1. Put metadata (Sync)
        try:
            self._kv_client.mset(meta_keys, meta_bytes_list,
                                 self._set_param.write_mode)
        except Exception as e:
            logger.error("batch_put metadata mset failed: %s", e)
            raise

        # 2. Put data (Async init, returns future)
        future = self._ds_tensor_client.async_mset_d2h(keys, tensors,
                                                       self._set_param)
        self._put_queue.put(future)

    def wait_for_put(self) -> None:
        """
        Waits for all pending put operations to complete.
        """
        while not self._put_queue.empty():
            future = self._put_queue.get()
            try:
                # Block until transfer completes
                failed_list = future.get()
                if failed_list:
                    logger.error("Async put transfer failed for keys: %s",
                                 failed_list)
            except Exception as e:
                logger.error("Error waiting for put future: %s", e)
            finally:
                # If using queue.join(), task_done is needed,
                # but standard Queue usage here is fine.
                pass

    def batch_exists(self, keys: list[str]) -> list[bool]:
        """
        Checks if keys exist in the store.
        """
        if not keys:
            return []
        return self._ds_tensor_client.exist(keys)
