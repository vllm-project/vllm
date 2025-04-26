# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

try:
    __version__ = importlib.metadata.version("vllm-flash-attn")
except importlib.metadata.PackageNotFoundError:
    # in this case, vllm-flash-attn is built from installing vllm editable
    __version__ = "0.0.0.dev0"

from .flash_attn_interface import (fa_version_unsupported_reason,
                                   flash_attn_varlen_func,
                                   flash_attn_with_kvcache,
                                   get_scheduler_metadata,
                                   is_fa_version_supported, sparse_attn_func,
                                   sparse_attn_varlen_func)

__all__ = [
    'flash_attn_varlen_func', 'flash_attn_with_kvcache',
    'get_scheduler_metadata', 'sparse_attn_func', 'sparse_attn_varlen_func',
    'is_fa_version_supported', 'fa_version_unsupported_reason'
]
