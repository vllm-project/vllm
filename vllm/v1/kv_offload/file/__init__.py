# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
File-based KV cache offloading.

This module provides file-based offloading for KV cache data,
storing blocks as binary files on disk.
"""
from vllm.v1.kv_offload.file.handler import FileOffloadingHandler
from vllm.v1.kv_offload.file.load_store_spec import FileLoadStoreSpec
from vllm.v1.kv_offload.file.manager import FileOffloadingManager
from vllm.v1.kv_offload.file.spec import FileOffloadingSpec

__all__ = [
    "FileOffloadingHandler",
    "FileOffloadingManager",
    "FileOffloadingSpec",
    "FileLoadStoreSpec",
]
