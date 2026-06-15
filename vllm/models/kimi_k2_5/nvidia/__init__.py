# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K2.5 NVFP4 (NVIDIA) specialized package.

Currently this package only ships the custom CuTe DSL kernels under
:mod:`vllm.models.kimi_k2_5.nvidia.ops`. The kernels are imported explicitly
from their submodules (rather than re-exported here) so that importing this
package does not require the optional ``cutlass`` dependency or a Blackwell GPU.
"""
