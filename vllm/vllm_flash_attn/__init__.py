# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
import types

# Set up a virtual `flash_attn` package so that `flash_attn.cute.*` imports
# resolve without requiring the flash-attn pip package to be installed.
# In symlink mode (VLLM_FLASH_ATTN_SRC_DIR), __path__ points at the real
# source tree; in copy mode, it points at this package's directory.
_cute_dir = os.path.join(os.path.dirname(__file__), "cute")
if os.path.isdir(_cute_dir) and "flash_attn" not in sys.modules:
    _fa_mod = types.ModuleType("flash_attn")
    if os.path.islink(_cute_dir):
        _fa_mod.__path__ = [os.path.dirname(os.path.realpath(_cute_dir))]
    else:
        _fa_mod.__path__ = [os.path.dirname(__file__)]
    _fa_mod.__package__ = "flash_attn"
    sys.modules["flash_attn"] = _fa_mod

from vllm.vllm_flash_attn.flash_attn_interface import (  # noqa: E402
    FA2_AVAILABLE,
    FA3_AVAILABLE,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

if not (FA2_AVAILABLE or FA3_AVAILABLE):
    raise ImportError(
        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "
        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."
    )

__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
