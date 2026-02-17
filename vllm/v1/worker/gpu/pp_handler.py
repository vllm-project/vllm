# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism handler for V2 Model Runner."""

import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class PPHandler:
    """Pipeline parallelism handler for Model Runner V2.

    Manages sampled token synchronization between PP ranks.
    Only instantiated when PP is enabled (pp_size > 1).
    """

    def __init__(self, device: torch.device):
        self.device = device

    def maybe_broadcast_sampled_tokens(
        self,
        sampler_output: SamplerOutput,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """Broadcast sampled tokens from the last PP rank to all other ranks.

        No-ops if this is not the last rank.

        Broadcasts sampled_token_ids [num_reqs, max_sample_len], num_sampled
        [num_reqs], and num_rejected [num_reqs] to support both regular decode
        and speculative decoding.

        Args:
            sampler_output: SamplerOutput from sampling.
            num_sampled: Number of accepted tokens per request.
            num_rejected: Number of rejected tokens per request.
        """
        pp = get_pp_group()
        if not pp.is_last_rank:
            return

        torch.distributed.broadcast(
            sampler_output.sampled_token_ids.contiguous(),
            src=pp.last_rank,
            group=pp.device_group,
        )
        # NOTE: num_sampled/num_rejected are only needed
        # for speculative decoding.
        torch.distributed.broadcast(
            num_sampled.contiguous(),
            src=pp.last_rank,
            group=pp.device_group,
        )
        torch.distributed.broadcast(
            num_rejected.contiguous(),
            src=pp.last_rank,
            group=pp.device_group,
        )

    def maybe_receive_sampled_tokens(
        self,
        num_reqs: int,
        max_sample_len: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Receive sampled tokens broadcast by the last PP rank.

        Returns None if this is the last rank (which samples, not receives).

        Args:
            num_reqs: Number of requests in the batch.
            max_sample_len: Maximum number of tokens sampled per request
                (1 for regular decode, >1 for speculative decoding).

        Returns:
            None if called on last rank.
            Otherwise, tuple of (sampled_tokens, num_sampled, num_rejected):
            - sampled_tokens: shape [num_reqs, max_sample_len]
            - num_sampled: shape [num_reqs]
            - num_rejected: shape [num_reqs]
        """
        pp = get_pp_group()
        if pp.is_last_rank:
            return None

        sampled_tokens = torch.empty(
            num_reqs, max_sample_len, dtype=torch.int64, device=self.device
        )
        torch.distributed.broadcast(
            sampled_tokens,
            src=pp.last_rank,
            group=pp.device_group,
        )
        # NOTE: num_sampled/num_rejected are only needed
        # for speculative decoding.
        num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(
            num_sampled,
            src=pp.last_rank,
            group=pp.device_group,
        )
        num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(
            num_rejected,
            src=pp.last_rank,
            group=pp.device_group,
        )
        return sampled_tokens, num_sampled, num_rejected
