# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
import triton
import triton.language as tl

from vllm.config import LogprobsMode
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.states import SamplingMetadata

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "processed_logprobs",
    ):
        super().__init__()
        assert logprobs_mode == "processed_logprobs"
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
            sampling_metadata.seeds,
            sampling_metadata.pos,
        )

        logprobs_tensors = None
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            logprobs_tensors = compute_logprobs(
                logits,
                num_logprobs,
                sampled,
            )

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output


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
        temp = 1.0

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
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    req_idx = tl.program_id(0)
    temp = tl.load(temp_ptr + req_idx)

    if temp < EPSILON:
        # Greedy sampling. Don't apply gumbel noise.
        return

    seed = tl.load(seeds_ptr + req_idx).to(tl.uint64)
    pos = tl.load(pos_ptr + req_idx).to(tl.uint64)
    gumbel_seed = seed ^ (pos * 0x9E3779B97F4A7C15)

    block_id = tl.program_id(1)
    r_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q = tl.rand(gumbel_seed, r_offset)

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
    seeds: torch.Tensor,
    # int64[num_reqs]
    pos: torch.Tensor,
) -> torch.Tensor:
    num_reqs = probs.shape[0]
    vocab_size = probs.shape[1]

    # Update the probs in-place.
    BLOCK_SIZE = 8192
    _apply_gumbel_kernel[(num_reqs, triton.cdiv(vocab_size, BLOCK_SIZE))](
        probs,
        probs.stride(0),
        seeds,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE,
        EPSILON=_SAMPLING_EPS,
    )
    # Sample the next token.
    return probs.argmax(dim=-1).view(-1)


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        l = tl.load(row_ptr + block,
                    mask=block < vocab_size,
                    other=float("-inf"))
        max_val = tl.max(tl.maximum(l, max_val))

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        l = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        e = tl.exp(l - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask)

    l = tl.load(row_ptr + topk_ids, mask=k_mask)
    o = l - max_val - lse
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


def compute_topk_logprobs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    topk = topk_ids.shape[1]
    output = torch.empty(
        batch_size,
        topk,
        dtype=torch.float32,
        device=logits.device,
    )
    _topk_log_softmax_kernel[(batch_size, )](
        output,
        logits,
        logits.stride(0),
        topk_ids,
        topk,
        vocab_size,
        BLOCK_SIZE=1024,
        PADDED_TOPK=triton.next_power_of_2(topk),
    )
    return output


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        l = tl.load(row_ptr + block,
                    mask=block < vocab_size,
                    other=float("-inf"))
        n += tl.sum((l > x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape
    if num_logprobs == 0:
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
    else:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        logprob_token_ids = torch.cat(
            (sampled_token_ids.unsqueeze(-1), topk_indices), dim=1)

    # NOTE(woosuk): Here, to save GPU memory, we do not materialize the full
    # logprobs tensor. Instead, we only compute and return the logprobs of
    # the topk + 1 tokens.
    logprobs = compute_topk_logprobs(
        logits,
        logprob_token_ids,
    )

    token_ranks = torch.empty(
        batch_size,
        dtype=torch.int64,
        device=logits.device,
    )
    _ranks_kernel[(batch_size, )](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
    )
