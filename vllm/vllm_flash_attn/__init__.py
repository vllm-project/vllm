# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.vllm_flash_attn.flash_attn_interface import (
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
