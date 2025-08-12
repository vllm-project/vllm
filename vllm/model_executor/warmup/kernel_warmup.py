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
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def kernel_warmup(model: torch.nn.Module, max_tokens: int):
    do_deep_gemm_warmup = (envs.VLLM_USE_DEEP_GEMM
                           and is_deep_gemm_supported()
                           and not envs.VLLM_SKIP_DEEP_GEMM_WARMUP)
    if do_deep_gemm_warmup:
        deep_gemm_warmup(model, max_tokens)


def flashinfer_autotune(runner: GPUModelRunner) -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    from flashinfer.autotuner import autotune

    with torch.inference_mode(), autotune():
        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        runner._dummy_run(runner.scheduler_config.max_num_batched_tokens,
                          skip_eplb=True,
                          is_profile=True)
