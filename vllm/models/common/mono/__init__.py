# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernels (RFC #60904): a decoder layer's decode step in one launch."""

from vllm.models.common.mono.guards import configure
from vllm.models.common.mono.runtime import attach, kernel_for_layer
from vllm.models.common.mono.spec import MonoSpec

__all__ = [
    "MonoSpec",
    "attach",
    "configure",
    "kernel_for_layer",
]
