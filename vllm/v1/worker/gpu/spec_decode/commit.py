# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The accepted path of a speculative round, as the commit kernels get it.
Moving the KV of path row ``path[j]`` to chain slot ``j`` is the
``layout:paged_token`` commit kernel's job; recurrent state resumes from the
last accepted row (:func:`accepted_rows`)."""

import torch

from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.round import AcceptResult


def accept_result_of(
    sampler_output: SamplerOutput, chain_path: torch.Tensor
) -> AcceptResult:
    """The round's :class:`AcceptResult`; a chain's path is a view of the
    constant ``chain_path``, so it costs no launch."""
    sampled = sampler_output.sampled_token_ids
    path = sampler_output.accept_path
    if path is None:
        path = chain_path[: sampled.shape[0], : sampled.shape[1]]
    return AcceptResult(path=path, path_len=sampler_output.num_sampled, sampled=sampled)


def accepted_rows(result: AcceptResult, num_reqs: int) -> torch.Tensor:
    """``[num_reqs]`` int32: the verify row whose recurrent state each request resumes
    from, ``path[path_len - 1]`` (row 0 when nothing was committed)."""
    last = (result.path_len[:num_reqs].to(torch.int64) - 1).clamp_(min=0)
    return result.path[:num_reqs].gather(1, last.unsqueeze(1)).squeeze(1).int()
