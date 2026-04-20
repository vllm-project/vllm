# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoadStoreSpec for file-based KV offloading.
"""
from vllm.v1.kv_offload.abstract import LoadStoreSpec


class FileLoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading/storing KV blocks from/to files.

    file_paths: list of file paths for the blocks.
    block_offsets: byte offsets within each file (for mmap-style access).
    """

    def __init__(self, file_paths: list[str], block_offsets: list[int] | None = None):
        self.file_paths = file_paths
        self.block_offsets = block_offsets or [0] * len(file_paths)

    @staticmethod
    def medium() -> str:
        return "FILE"
