# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""
import torch

import vllm.envs as envs
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.utils.deep_gemm import is_deep_gemm_supported


def kernel_warmup(model: torch.nn.Module, max_tokens: int):
    do_deep_gemm_warmup = (envs.VLLM_USE_DEEP_GEMM
                           and is_deep_gemm_supported()
                           and not envs.VLLM_SKIP_DEEP_GEMM_WARMUP)
    if do_deep_gemm_warmup:
        deep_gemm_warmup(model, max_tokens)
