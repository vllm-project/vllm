# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wrapper that delegates to the shared DFlash CUDA graph manager."""

from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager


def DominoCudaGraphManager(*args, **kwargs):
    """Create a CUDA graph manager for Domino (delegates to DFlash)."""
    return DFlashCudaGraphManager(*args, **kwargs)
