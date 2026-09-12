# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton image-mask and FlashInfer causal attention composite."""

from vllm.v1.attention.backends.composite import (
    MMPrefixAttentionRouting,
    create_composite_attention_backend,
)
from vllm.v1.attention.backends.flashinfer import FlashInferBackend, FlashInferImpl
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend


class _CompositeFlashInferImpl(FlashInferImpl):
    # Cache updates from Triton do not participate in FlashInfer's PDL chain.
    trtllm_decode_enable_pdl = False


class _CompositeFlashInferBackend(FlashInferBackend):
    @staticmethod
    def get_impl_cls() -> type[FlashInferImpl]:
        return _CompositeFlashInferImpl


TritonFlashInferBackend = create_composite_attention_backend(
    TritonAttentionBackend,
    _CompositeFlashInferBackend,
    name="TritonFlashInferBackend",
    backend_name="TRITON_FLASHINFER",
    module=__name__,
    routing_policy=MMPrefixAttentionRouting,
    head_sizes=(256, 512),
    kernel_block_sizes=(64,),
    device_major=10,
)
