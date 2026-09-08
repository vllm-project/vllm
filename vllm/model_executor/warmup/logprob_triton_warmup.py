# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the logprobs top-k/log-softmax Triton kernel across every
`TOPK_BLOCK_SIZE` bucket it can be specialized for.

`SamplingParams.for_sampler_warmup()` only exercises two column counts (via
its fixed `logprobs=5` and `prompt_logprobs=1`), so a request with a
different `num_logprobs` count reaches a `TOPK_BLOCK_SIZE` bucket that was
never compiled and pays the Triton JIT compile inline instead of at startup.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.logprob import MAX_TOPK_BLOCK, compute_token_logprobs

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def _warm_topk_log_softmax_kernel(
    device: torch.device, dtype: torch.dtype, vocab_size: int
) -> None:
    """Compile `_topk_log_softmax_kernel` for every power-of-2 topk width.

    `vocab_size` must be the model's real vocab size: Triton also
    specializes plain (non-`constexpr`) integer arguments on whether their
    runtime value is exactly 1, and `vocab_size`/`logits_stride` are
    otherwise constant across every real call for a given model, so warming
    with a placeholder value like 1 would compile a variant production never
    reuses. `topk` doesn't need to match exactly: `_topk_log_softmax_kernel`
    is declared with `do_not_specialize_on_alignment=["topk"]` so it isn't
    also specialized per-call on whether that value happens to be 16-byte
    aligned (`num_logprobs`, which drives it, is user-controlled and
    unbounded, so no finite warmup could otherwise cover every alignment).
    """
    logits = torch.zeros(1, vocab_size, device=device, dtype=dtype)
    topk = 1
    while topk <= MAX_TOPK_BLOCK:
        token_ids = torch.zeros(1, topk, device=device, dtype=torch.int64)
        compute_token_logprobs(logits, token_ids)
        topk *= 2


@torch.inference_mode()
def logprob_triton_warmup(runner: "GPUModelRunner") -> None:
    """Warm the logprobs Triton kernel for every reachable `num_logprobs`."""
    device = runner.device
    if device.type != "cuda":
        return
    vocab_size = runner.vocab_size
    # `--hf-overrides '{"head_dtype": "float32"}'` computes logits in fp32
    # regardless of the model's compute dtype; warm both so neither can hit
    # an uncompiled bucket.
    for dtype in {runner.dtype, torch.float32}:
        _warm_topk_log_softmax_kernel(device, dtype, vocab_size)
    logger.info("Warmed logprobs Triton kernel for every num_logprobs bucket.")
