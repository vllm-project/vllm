# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import aiofiles
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorage,
    ExOffloadingStorageKVCacheConfig,
    copy_data_d2h,
    copy_data_h2d,
    get_mem_tensors,
    tensors_total_numel,
)

DEFAULT_FILE_ROOT_PATH = "/dev/shm/vllm_file_storage"


class FileStorage(ExOffloadingStorage):
    def __init__(self, extra_config: dict[str, str]):
        self.root_path = extra_config.get("root_path")
        if self.root_path is None:
            raise ValueError("not found File root root_path")

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[dict[str, str], str]:
        if not uri.startswith("file://"):
            raise ValueError("invalid File URI format")

        filepath = uri.replace("file://", "")

        root_path = os.getenv("FILE_ROOT_PATH", None)
        if root_path is not None:
            if not os.path.exists(root_path):
                raise ValueError(f"FILE_ROOT_PATH {root_path} not exists")
        else:
            root_path = DEFAULT_FILE_ROOT_PATH
            os.makedirs(root_path, exist_ok=True)

        return {"root_path": root_path}, filepath

    def register_kvcache(self, config: ExOffloadingStorageKVCacheConfig) -> None:
        self.kvcache_config = config

    async def load(self, filepath: str, offset: int, block_ids: list[int]) -> None:
        if not block_ids:
            return

        assert self.root_path is not None

        mem_tensors = get_mem_tensors(
            self.kvcache_config.kv_caches, block_ids, self.kvcache_config.split_k_and_v
        )
        host_buffer = torch.empty(
            tensors_total_numel(mem_tensors), dtype=mem_tensors[0].dtype, device="cpu"
        )

        tensor_path = os.path.realpath(
            os.path.join(self.root_path, filepath.lstrip("/"))
        )
        if not tensor_path.startswith(os.path.realpath(self.root_path) + os.sep):
            raise ValueError(f"Path traversal detected: {filepath}")

        async with aiofiles.open(tensor_path, "rb", buffering=0) as f:
            await f.seek(offset)

            buf = memoryview(host_buffer.flatten().view(torch.uint8).numpy())
            bytes_read = await f.readinto(buf)
            if bytes_read is None or bytes_read != len(buf):
                raise OSError(
                    f"Short read for {filepath} at offset {offset}. "
                    f"Expected {len(buf)} bytes, got {bytes_read} bytes."
                )

        copy_data_h2d(host_buffer, mem_tensors)

    async def save(self, filepath: str, offset: int, block_ids: list[int]) -> None:
        if not block_ids:
            return

        assert self.root_path is not None

        mem_tensors = get_mem_tensors(
            self.kvcache_config.kv_caches, block_ids, self.kvcache_config.split_k_and_v
        )
        host_buffer = torch.empty(
            tensors_total_numel(mem_tensors), dtype=mem_tensors[0].dtype, device="cpu"
        )
        copy_data_d2h(mem_tensors, host_buffer)

        tensor_path = os.path.realpath(
            os.path.join(self.root_path, filepath.lstrip("/"))
        )
        if not tensor_path.startswith(os.path.realpath(self.root_path) + os.sep):
            raise ValueError(f"Path traversal detected: {filepath!r}")

        async with aiofiles.open(tensor_path, "r+b", buffering=0) as f:
            await f.seek(offset)

            buf = memoryview(host_buffer.flatten().view(torch.uint8).numpy())
            bytes_written = await f.write(buf)
            if bytes_written is None or bytes_written != len(buf):
                raise OSError(
                    f"Short write for {filepath} at offset {offset}. "
                    f"Expected to write {len(buf)} bytes, wrote {bytes_written} bytes."
                )
