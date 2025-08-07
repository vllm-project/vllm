# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains a new class `MooncakeStore` that allows developers to
think of KV cache transfer operations as putting new KV cache entries
into a remote KVStore-based lookup buffer and getting existing KV caches
from this remote lookup buffer.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVStoreBufferBase)
from vllm.logger import init_logger

DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB

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

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeStoreConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size",
                                           DEFAULT_GLOBAL_SEGMENT_SIZE),
            local_buffer_size=config.get("local_buffer_size",
                                         DEFAULT_LOCAL_BUFFER_SIZE),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeStoreConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_file_path)


class MooncakeStore(KVStoreBufferBase):

    def __init__(
        self,
        config: VllmConfig,
    ):

        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")

            self.store.setup(self.config.local_hostname,
                             self.config.metadata_server,
                             self.config.global_segment_size,
                             self.config.local_buffer_size,
                             self.config.protocol, self.config.device_name,
                             self.config.master_server_address)

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def put(
        self,
        key: str,
        value: Optional[torch.Tensor],
    ) -> None:
        # A message queue needs to be introduced before making it asynchronous.
        if value is not None:
            self._put_impl(key, value)

    def get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        # A message queue needs to be introduced before making it asynchronous.
        value = self._get_impl(key)
        return value

    def _put_impl(
        self,
        key: str,
        value: torch.Tensor,
    ) -> None:
        """Put KVCache to Mooncake Store"""
        device_id = value.device.index if value.device.type == 'cuda' else -1
        device_tensor = torch.tensor(device_id, dtype=torch.int32)
        value_bytes = safetensors_save({
            "tensor": value,
            "device_id": device_tensor
        })
        try:
            self.store.put(key, value_bytes)
        except TypeError as err:
            logger.error("Failed to put value into Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Put Type Error.") from err

    def _get_impl(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        """Get KVCache from Mooncake Store"""
        try:
            data = self.store.get(key)
        except TypeError as err:
            logger.error("Failed to get value from Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Get Type Error.") from err

        if data:
            loaded_tensors = safetensors_load(data)
            tensor = loaded_tensors["tensor"]
            device_id_tensor = loaded_tensors["device_id"]
            device_id = int(device_id_tensor.item())
            device = torch.device(
                'cuda', device_id) if device_id >= 0 else torch.device('cpu')
            return tensor.to(device)

        return None
