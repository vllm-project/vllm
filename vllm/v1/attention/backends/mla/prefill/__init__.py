# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA prefill backend implementations.

This package provides modular prefill backends for MLA (Multi-head Latent
Attention) with priority-based selection.

Available backends:
- FlashAttention (default, works on all hardware)
- FlashInfer (optimized for Blackwell)
- cuDNN (optimized for Blackwell, requires NVIDIA artifactory)
- TRT-LLM Ragged (optimized for Blackwell, DeepSeek-specific)

Metadata classes (FlashInferPrefillMetadata, CudnnPrefillMetadata) are available
through their respective backend modules to avoid circular imports.
"""

from vllm.v1.attention.backends.mla.prefill.base import (
    MLAPrefillBackend,
    MLAPrefillBuilderState,
    MLAPrefillImpl,
)
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.mla.prefill.selector import get_mla_prefill_backend

__all__ = [
    "MLAPrefillBackend",
    "MLAPrefillBuilderState",
    "MLAPrefillImpl",
    "MLAPrefillBackendEnum",
    "get_mla_prefill_backend",
]
