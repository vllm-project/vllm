# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from vllm.config import LogprobsMode
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def __init__(
        self,
        logprobs_mode: LogprobsMode = LogprobsMode.PROCESSED_LOGPROBS,
    ):
        super().__init__()
        assert logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # Divide logits by temperature, in FP32.
        logits = apply_temperature(logits, sampling_metadata.temperature)

        # Apply top_k and/or top_p.
        logits = apply_top_k_top_p(
            logits,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # Sample the next token (int64).
        sampled = gumbel_sample(
            probs,
            sampling_metadata.temperature,
            None,  # seeds
            None,  # pos
        )

        logprobs_tensors = None
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            assert num_logprobs >= 0
            # Compute the logprobs.
            logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
            # Gather the logprobs of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                logprobs,
                num_logprobs,
                sampled,
            )

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        sampled: torch.Tensor,
    ) -> LogprobsTensors:
        sampled = sampled.unsqueeze(-1)
        sampled_logprobs = logprobs.gather(-1, sampled)
        sampled_ranks = (logprobs > sampled_logprobs).sum(-1)
        if num_logprobs == 0:
            # Return the logprobs of the sampled token.
            logprobs_tensors = LogprobsTensors(
                sampled,
                sampled_logprobs,
                sampled_ranks,
            )
        else:
            # Return (num_logprobs + 1) logprobs.
            topk_logprobs, topk_indices = torch.topk(
                logprobs,
                num_logprobs,
                dim=-1,
            )
            logprobs_tensors = LogprobsTensors(
                torch.cat((sampled, topk_indices), dim=1),
                torch.cat((sampled_logprobs, topk_logprobs), dim=1),
                sampled_ranks,
            )
        return logprobs_tensors


@triton.jit
def _apply_temp_kernel(
    logits,  # bf16[batch_size, vocab_size]
    logits_stride,
    output,  # fp32[batch_size, vocab_size]
    output_stride,
    temperature,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    temp = tl.load(temperature + batch_idx)
    if temp < EPSILON:
        # Greedy sampling. Don't apply temperature.
        # NOTE(woosuk): In this case, we assume that its logprobs are not used.
        temp = tl.ones([1], dtype=tl.float32)

    offset = tl.arange(0, BLOCK_SIZE)
    block = block_idx * BLOCK_SIZE + offset

    # Load the logits.
    x = tl.load(logits + batch_idx * logits_stride + block,
                mask=block < vocab_size)
    x = x.to(tl.float32)
    x = x / temp
    tl.store(output + batch_idx * output_stride + block,
             x,
             mask=block < vocab_size)


def apply_temperature(
    logits: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    output = torch.empty_like(logits, dtype=torch.float32)
    BLOCK_SIZE = 8192
    _apply_temp_kernel[(batch_size, triton.cdiv(vocab_size, BLOCK_SIZE))](
        logits,
        logits.stride(0),
        output,
        output.stride(0),
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        EPSILON=_SAMPLING_EPS,
    )
    return output


@triton.jit
def _apply_gumbel_kernel(
    probs_ptr,
    probs_stride,
    seeds_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    req_idx = tl.program_id(0)
    seed = tl.load(seeds_ptr + req_idx)
    temp = tl.load(temp_ptr + req_idx)

    if temp < EPSILON:
        # Greedy sampling. Don't apply gumbel noise.
        return

    block_id = tl.program_id(1)
    r_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q = tl.rand(seed, r_offset)

    # NOTE(woosuk): This logic makes sure q is not 0.
    RMAX = 0.9999999403953552
    RMAX_LOG = -5.960464477539063e-08
    q = tl.where(q >= RMAX, RMAX_LOG, tl.math.log(q))
    q = -1.0 * q

    p = tl.load(probs_ptr + req_idx * probs_stride + r_offset,
                mask=r_offset < vocab_size)
    p = p / q

    tl.store(probs_ptr + req_idx * probs_stride + r_offset,
             p,
             mask=r_offset < vocab_size)


def gumbel_sample(
    # fp32[num_reqs, vocab_size]
    probs: torch.Tensor,
    # fp32[num_reqs]
    temperature: torch.Tensor,
    # int64[num_reqs]
    seeds: Optional[torch.Tensor],
    # int64[num_reqs]
    pos: Optional[torch.Tensor],
) -> torch.Tensor:
    num_reqs = probs.shape[0]
    vocab_size = probs.shape[1]
    if seeds is not None:
        # Per-request seed.
        assert pos is not None
        gumbel_seeds = seeds + pos
    else:
        # Global seed.
        assert pos is None
        seed_dtype = torch.int64
        gumbel_seeds = torch.randint(
            torch.iinfo(seed_dtype).min,
            torch.iinfo(seed_dtype).max,
            (num_reqs, ),
            dtype=seed_dtype,
            device=probs.device,
        )

    # Update the probs in-place.
    BLOCK_SIZE = 8192
    _apply_gumbel_kernel[(num_reqs, triton.cdiv(vocab_size, BLOCK_SIZE))](
        probs,
        probs.stride(0),
        gumbel_seeds,
        temperature,
        vocab_size,
        BLOCK_SIZE,
        EPSILON=_SAMPLING_EPS,
    )
    # Sample the next token.
    return probs.argmax(dim=-1).view(-1)
