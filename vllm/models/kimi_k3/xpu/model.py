# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 / Kimi-Linear on XPU.

The NVIDIA implementation's MLA epilogues (``nvidia/mla.py``) call the fused
``fused_kimi_k3_mla_*_concat_kv_cache_insert`` custom ops through
``torch.ops._C``, with no CUDA-specific Python branching. Once an XPU SYCL
implementation of those ops is registered under the same schema (see
``vllm-xpu-kernels``), that code dispatches correctly on XPU tensors as-is —
so this module re-exports the NVIDIA classes rather than forking them.
"""

from vllm.models.kimi_k3.nvidia.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
)

__all__ = ["KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"]
