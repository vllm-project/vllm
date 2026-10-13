# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MonoKernels (RFC #60904): a decoder layer's decode step in one launch."""

from vllm.models.common.mono.guards import configure
from vllm.models.common.mono.layer import attach, kernel_for_layer

__all__ = [
    "attach",
    "configure",
    "kernel_for_layer",
]
