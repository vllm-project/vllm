# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest
import torch

from vllm.model_executor.warmup.logprob_triton_warmup import (
    _warm_topk_log_softmax_kernel,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

_VOCAB_SIZE = 128


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_topk_log_softmax_kernel_compiles_on_gpu() -> None:
    _warm_topk_log_softmax_kernel(torch.device("cuda"), torch.bfloat16, _VOCAB_SIZE)
    torch.accelerator.synchronize("cuda")


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_warmup_precompiles_every_num_logprobs_bucket() -> None:
    """A `num_logprobs` count the fixed sampler warmup never exercises (see
    `SamplingParams.for_sampler_warmup`, which only reaches two of the
    kernel's topk-width buckets) must not pay a JIT compile inline once this
    warmup has run -- for any `topk` value that maps to an already-warmed
    bucket, not just the exact values the warmup happened to call with.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    _warm_topk_log_softmax_kernel(device, dtype, _VOCAB_SIZE)

    logits = torch.randn(4, _VOCAB_SIZE, device=device, dtype=dtype)

    def assert_cached(topk: int) -> None:
        token_ids = torch.randint(
            0, _VOCAB_SIZE, (4, topk), device=device, dtype=torch.int64
        )
        torch.accelerator.synchronize()
        start = time.perf_counter()
        compute_token_logprobs(logits, token_ids)
        torch.accelerator.synchronize()
        elapsed = time.perf_counter() - start
        # A cache hit is sub-millisecond; a fresh compile measures in the
        # hundreds of milliseconds (see the kernel_warmup module docstring).
        assert elapsed < 0.05, (
            f"compute_token_logprobs(topk={topk}) took {elapsed * 1e3:.1f}ms "
            "after warmup; expected a cached kernel, not a fresh JIT compile"
        )

    # topk=33 -> TOPK_BLOCK_SIZE=64, a bucket the fixed warmup path in
    # `SamplingParams.for_sampler_warmup` (logprobs=5, prompt_logprobs=1)
    # never reaches -- and, unlike a clean power of 2, isn't 16-byte
    # aligned. This is exactly the shape a real `logprobs=32` request
    # produces (1 sampled token + 32 top-k columns).
    assert_cached(33)
    # A 16-byte-aligned topk in the same bucket must hit the same compiled
    # kernel too: Triton specializes plain (non-constexpr) int arguments on
    # alignment by default, which is exactly what
    # `do_not_specialize_on_alignment=["topk"]` on `_topk_log_softmax_kernel`
    # disables. Without it, this call recompiles even though topk=33 above
    # already warmed the same `TOPK_BLOCK_SIZE` bucket.
    assert_cached(64)
