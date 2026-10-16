# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm entry point for Beam.

Beam's baseline model and Triton kernels are portable to ROCm once
CUDA-only launch options are omitted. Keep a platform-specific entry point so
CUDA-only optimizations and speculative decoding are not selected on ROCm.
"""

from ..nvidia.model import BeamForCausalLM

__all__ = ["BeamForCausalLM"]
