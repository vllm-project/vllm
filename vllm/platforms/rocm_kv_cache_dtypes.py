# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm-only custom ``--kv-cache-dtype`` values.

Imported by ``RocmPlatform.pre_register_and_update`` before ``CacheConfig``
validates the dtype, and by the attention modules that use them, so every
worker registers them too.
"""

import torch

from vllm.config.cache import register_kv_cache_dtype
from vllm.v1.kv_cache_interface import KVQuantMode


@register_kv_cache_dtype("mxfp4_qsa")
class MXFP4QSAKVCacheDType:
    """OCP MXFP4 K and V records for the AMD QSA main attention cache."""

    name = "mxfp4_qsa"

    def torch_dtype(self) -> torch.dtype:
        return torch.uint8

    def quant_mode(self) -> KVQuantMode:
        # The QSA backend owns every read and write of this cache, so no
        # generic quantized-KV path (query quantization, fp8 cache views,
        # per-mode workspaces) may apply to it.
        return KVQuantMode.NONE
