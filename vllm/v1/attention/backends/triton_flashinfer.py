# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton image-mask and FlashInfer causal attention composite."""

from vllm.v1.attention.backends.composite import (
    MMPrefixAttentionRouting,
    create_composite_attention_backend,
)
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

TritonFlashInferBackend = create_composite_attention_backend(
    TritonAttentionBackend,
    FlashInferBackend,
    name="TritonFlashInferBackend",
    backend_name="TRITON_FLASHINFER",
    module=__name__,
    routing_policy=MMPrefixAttentionRouting,
    head_sizes=(256, 512),
    kernel_block_sizes=(64,),
    device_major=10,
    # Use FlashInfer's KV writer to keep the following PDL decode compatible.
    kv_cache_update_variant=1,
)
