# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from vllm.platforms import current_platform


def supports_turing_indexer_fallback() -> bool:
    """True when this CUDA device needs the portable (Turing/SM75) indexer
    logits fallback instead of the DeepGEMM kernels."""
    return current_platform.is_cuda() and current_platform.is_device_capability((7, 5))
