# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.v1.outputs import SamplerOutput


def evenly_split(
    n: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[int, int]:
    q = n // tp_size
    r = n % tp_size
    start = q * tp_rank + min(tp_rank, r)
    end = start + q + (1 if tp_rank < r else 0)
    return start, end


def pad_and_all_gather(
    x: torch.Tensor,
    padded_size: int,
) -> torch.Tensor:
    n = x.shape[0]
    if n != padded_size:
        padded_x = torch.empty(
            (padded_size, *x.shape[1:]),
            dtype=x.dtype,
            device=x.device,
        )
        padded_x[:n] = x
    else:
        padded_x = x

    x = tensor_model_parallel_all_gather(padded_x)
    return x


def all_gather_sampler_output(
    sampler_output: SamplerOutput,
    num_reqs: int,
    tp_size: int,
) -> SamplerOutput:
    n = (num_reqs + tp_size - 1) // tp_size
    sampler_output.sampled_token_ids = pad_and_all_gather(
        sampler_output.sampled_token_ids, n)[:num_reqs]

    # TODO(woosuk): 3 small all-gathers, could be merged into one.
    logprobs_tensors = sampler_output.logprobs_tensors
    if logprobs_tensors is not None:
        logprobs_tensors.logprob_token_ids = pad_and_all_gather(
            logprobs_tensors.logprob_token_ids, n)[:num_reqs]
        logprobs_tensors.logprobs = pad_and_all_gather(
            logprobs_tensors.logprobs, n)[:num_reqs]
        logprobs_tensors.selected_token_ranks = pad_and_all_gather(
            logprobs_tensors.selected_token_ranks, n)[:num_reqs]
    return sampler_output
