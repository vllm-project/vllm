# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the KV cache block copy kernel used by prefix-cache copy-on-write."""

from typing import TYPE_CHECKING

import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def kv_cache_block_copy_warmup(runner: "GPUModelRunner") -> None:
    """Compile the block copy kernel, which copy-on-write first runs mid-serving."""
    config = runner.vllm_config
    if not (
        config.kernel_config.enable_jit_warmup
        and config.cache_config.enable_prefix_caching
    ):
        return
    blocks = torch.zeros(2, 1, dtype=torch.int64, device=runner.device)
    copy_kv_cache_blocks_inplace([blocks], 2, [KVCacheBlockCopy(0, 1)])
