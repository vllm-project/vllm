# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
import triton
import triton.language as tl

from vllm.config.model import LogprobsMode
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p


class Sampler:
    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
    ):
        if logprobs_mode not in ["processed_logprobs", "raw_logprobs"]:
            raise NotImplementedError(f"Unsupported logprobs_mode: {logprobs_mode}")
        self.logprobs_mode = logprobs_mode

    def __call__(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        if sampling_metadata.max_num_logprobs is not None:
            if self.logprobs_mode == "processed_logprobs":
                sampled, logits = self.sample(
                    logits, sampling_metadata, return_logits=True
                )
            else:
                assert self.logprobs_mode == "raw_logprobs"
                sampled, _ = self.sample(logits, sampling_metadata, return_logits=False)

            logprobs_tensors = compute_topk_logprobs(
                logits,
                sampling_metadata.max_num_logprobs,
                sampled,
            )
        else:
            sampled, _ = self.sample(logits, sampling_metadata, return_logits=False)
            logprobs_tensors = None

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_greedy = sampling_metadata.temperature == 0
        temp = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
        logits = logits / temp.view(-1, 1)
        logits = apply_top_k_top_p(
            logits, sampling_metadata.top_k, sampling_metadata.top_p
        )

        sampled = gumbel_sample(
            logits,
            is_greedy,
            sampling_metadata.seeds,
            sampling_metadata.pos,
        )
        return sampled, logits if return_logits else None


@triton.jit
def _gumbel_sample_kernel(
    sampled_ptr,
    logits_ptr,
    logits_stride,
    seeds_ptr,
    pos_ptr,
    is_greedy_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)

    if is_greedy:
        # Greedy sampling. Don't apply gumbel noise.
        max_val = float("-inf")
        max_idx = 0
        for i in range(0, vocab_size, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < vocab_size
            logits = tl.load(
                logits_ptr + req_idx * logits_stride + block,
                mask=mask,
                other=float("-inf"),
            )

            idx = tl.argmax(logits, axis=0)
            value = tl.max(logits, axis=0)
            is_greater = value > max_val
            max_val = tl.where(is_greater, value, max_val)
            max_idx = tl.where(is_greater, i + idx, max_idx)
        tl.store(sampled_ptr + req_idx, max_idx)
        return

    # Random sampling.
    # Calculate gumbel seed.
    seed = tl.load(seeds_ptr + req_idx)
    pos = tl.load(pos_ptr + req_idx)
    gumbel_seed = tl.randint(seed, pos)

    max_val = float("-inf")
    max_idx = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size

        # Generate gumbel noise.
        r = tl.rand(gumbel_seed, block).to(tl.float64)
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
        gumbel_noise = gumbel_noise.to(tl.float32)

        # Apply gumbel noise.
        logits = tl.load(logits_ptr + req_idx * logits_stride + block, mask=mask)
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

        # Argmax to get the sampled token.
        idx = tl.argmax(logits, axis=0)
        value = tl.max(logits, axis=0)
        is_greater = value > max_val
        max_val = tl.where(is_greater, value, max_val)
        max_idx = tl.where(is_greater, i + idx, max_idx)
    tl.store(sampled_ptr + req_idx, max_idx)


def gumbel_sample(
    logits: torch.Tensor,  # [num_reqs, vocab_size]
    is_greedy: torch.Tensor,  # [num_reqs]
    seed: torch.Tensor,  # [num_reqs]
    pos: torch.Tensor,  # [num_reqs]
) -> torch.Tensor:
    num_reqs, vocab_size = logits.shape
    # NOTE(woosuk): Use int64 for later indexing.
    sampled = torch.empty(
        num_reqs,
        dtype=torch.int64,
        device=logits.device,
    )
    _gumbel_sample_kernel[(num_reqs,)](
        sampled,
        logits,
        logits.stride(0),
        seed,
        pos,
        is_greedy,
        vocab_size,
        num_warps=8,
        BLOCK_SIZE=16384,  # type: ignore
    )
    return sampled


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
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)

    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - max_val - lse
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


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
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits > x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size = logits.shape[0]
    vocab_size = logits.shape[1]
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = torch.empty(
        batch_size,
        num_logprobs,
        dtype=torch.float32,
        device=logits.device,
    )
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        PADDED_TOPK=triton.next_power_of_2(num_logprobs),
    )
    return logprobs


def compute_topk_logprobs(
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
            (sampled_token_ids.unsqueeze(-1), topk_indices), dim=1
        )

    # NOTE(woosuk): Here, to save GPU memory, we do not materialize the full
    # logprobs tensor. Instead, we only compute and return the logprobs of
    # the topk + 1 tokens.
    logprobs = compute_token_logprobs(logits, logprob_token_ids)
    token_ranks = torch.empty(
        batch_size,
        dtype=torch.int64,
        device=logits.device,
    )
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
    )


def compute_prompt_logprobs(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    CHUNK_SIZE = 1024
    logprobs = []
    ranks = []
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        prompt_logprobs = compute_topk_logprobs(
            prompt_logits,
            0,  # num_logprobs
            prompt_token_ids[start_idx:end_idx],
        )
        logprobs.append(prompt_logprobs.logprobs)
        ranks.append(prompt_logprobs.selected_token_ranks)

    logprobs = torch.cat(logprobs, dim=0) if len(logprobs) > 1 else logprobs[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return logprobs, ranks
