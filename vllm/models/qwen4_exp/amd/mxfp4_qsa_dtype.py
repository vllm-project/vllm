# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registers the ``mxfp4_qsa`` ``--kv-cache-dtype`` (see ``ops/mxfp4_kv.py``).

Imported by the ROCm platform before ``CacheConfig`` validates the dtype, and
by the QSA attention module so every worker registers it too.
"""

import torch

from vllm.config.cache import register_kv_cache_dtype
from vllm.v1.kv_cache_interface import KVQuantMode


@register_kv_cache_dtype("mxfp4_qsa")
class MXFP4QSAKVCacheDType:
    name = "mxfp4_qsa"

    def torch_dtype(self) -> torch.dtype:
        return torch.uint8

    def quant_mode(self) -> KVQuantMode:
        # The QSA backend owns both the writer and the reader.
        return KVQuantMode.CUSTOM
