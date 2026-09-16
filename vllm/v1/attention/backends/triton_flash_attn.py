# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton image-mask and FlashAttention causal attention composite."""

import torch

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.composite import (
    MMPrefixAttentionRouting,
    create_composite_attention_backend,
)
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend


class _FA4Backend(FlashAttentionBackend):
    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        reason = super().supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )
        if reason is not None:
            return reason
        if (
            get_flash_attn_version(
                head_size=head_size,
                has_sinks=has_sink,
                kv_cache_block_size=block_size,
                supports_fa4_hd256=True,
            )
            != 4
        ):
            return "causal route requires FlashAttention v4"
        return None


TritonFlashAttentionBackend = create_composite_attention_backend(
    TritonAttentionBackend,
    _FA4Backend,
    name="TritonFlashAttentionBackend",
    backend_name="TRITON_FLASH_ATTN",
    module=__name__,
    routing_policy=MMPrefixAttentionRouting,
    head_sizes=(256, 512),
    device_major=9,
)
