# SPDX-License-Identifier: Apache-2.0
"""Local file system based KV store implementation."""
import os
from typing import Optional

import torch
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVStoreBufferBase)
from vllm.logger import init_logger

logger = init_logger(__name__)


class FileStore(KVStoreBufferBase):
    """KV store implementation using local filesystem with safetensors."""

    def __init__(
        self,
        config: VllmConfig,
    ):
        self.storage_path = config.kv_transfer_config.get_from_extra_config(
            "fs_storage_path", "/tmp/vllm_kv_cache")
        os.makedirs(self.storage_path, exist_ok=True)

    def close(self):
        """No resources to clean up for file storage"""
        pass

    def put(self, key: str, value: Optional[torch.Tensor]) -> None:
        """Save tensor to file with key as filename."""
        if value is None:
            return

        file_path = os.path.join(self.storage_path, f"{key}.safetensors")
        device_id = value.device.index if value.device.type == 'cuda' else -1
        device_tensor = torch.tensor(device_id, dtype=torch.int32)

        safetensors_save({
            "tensor": value.cpu(),
            "device_id": device_tensor
        }, file_path)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Load tensor from file with key as filename."""
        file_path = os.path.join(self.storage_path, f"{key}.safetensors")
        if not os.path.exists(file_path):
            return None

        try:
            data = safetensors_load(file_path)
            tensor = data["tensor"]
            device_id = int(data["device_id"].item())

            device = torch.device(
                'cuda', device_id) if device_id >= 0 else torch.device('cpu')
            return tensor.to(device)
        except Exception as e:
            logger.error("Error loading tensor %s: %s", key, str(e))
            return None
