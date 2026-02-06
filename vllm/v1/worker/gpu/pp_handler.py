# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism handler for V2 Model Runner."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class PPRole(Enum):
    """Role of this rank in pipeline parallelism."""

    NO_PP = auto()  # No PP (pp_size=1), full model on one rank
    FIRST = auto()  # First stage
    MIDDLE = auto()  # Middle stage
    LAST = auto()  # Last stage


@dataclass
class PPConfig:
    """Pipeline parallelism configuration for this rank."""

    role: PPRole
    pp_size: int
    pp_rank: int

    @property
    def is_first_rank(self) -> bool:
        return self.role in (PPRole.NO_PP, PPRole.FIRST)

    @property
    def is_last_rank(self) -> bool:
        return self.role in (PPRole.NO_PP, PPRole.LAST)

    @classmethod
    def from_parallel_config(cls, parallel_config) -> "PPConfig":
        """Build PPConfig from vLLM parallel config."""
        from vllm.distributed.parallel_state import get_pp_group

        pp_size = parallel_config.pipeline_parallel_size
        if pp_size <= 1:
            return cls(role=PPRole.NO_PP, pp_size=1, pp_rank=0)

        pp = get_pp_group()
        if pp.is_first_rank:
            role = PPRole.FIRST
        elif pp.is_last_rank:
            role = PPRole.LAST
        else:
            role = PPRole.MIDDLE

        return cls(role=role, pp_size=pp_size, pp_rank=pp.rank_in_group)


class PPHandler:
    """Pipeline parallelism handler for Model Runner V2.

    Manages sampled token synchronization between PP ranks and prepares
    model inputs/outputs for pipeline-parallel execution.

    All public methods no-op when PP is disabled or called on an
    inapplicable rank, so callers don't need guard conditions.
    """

    def __init__(self, config: PPConfig):
        self.config = config

    @property
    def is_enabled(self) -> bool:
        """Is pipeline parallelism enabled (pp_size > 1)?"""
        return self.config.pp_size > 1

    @property
    def receives_raw_inputs(self) -> bool:
        """Does this rank receive tokens/embeddings (vs intermediate tensors)?

        True for first rank or when PP is disabled.
        """
        return self.config.is_first_rank

    @property
    def produces_final_output(self) -> bool:
        """Does this rank produce output for sampling?

        True for last rank or when PP is disabled.
        """
        return self.config.is_last_rank

    # === Sampled token synchronization across PP ranks ===

    def maybe_broadcast_sampled_tokens(
        self,
        sampler_output: SamplerOutput,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """Broadcast sampled tokens from the last PP rank to all other ranks.

        No-ops if PP is disabled or if this is not the last rank.

        Broadcasts sampled_token_ids [num_reqs, max_sample_len], num_sampled
        [num_reqs], and num_rejected [num_reqs] to support both regular decode
        and speculative decoding.

        Args:
            sampler_output: SamplerOutput from sampling.
            num_sampled: Number of accepted tokens per request.
            num_rejected: Number of rejected tokens per request.
        """
        if not self.is_enabled or not self.produces_final_output:
            return

        from vllm.distributed.parallel_state import get_pp_group

        pp = get_pp_group()
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
        device: torch.device,
        max_sample_len: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Receive sampled tokens broadcast by the last PP rank.

        No-ops (returns None) if PP is disabled or if this is the last rank.

        Args:
            num_reqs: Number of requests in the batch.
            device: Device to create tensors on.
            max_sample_len: Maximum number of tokens sampled per request
                (1 for regular decode, >1 for speculative decoding).

        Returns:
            None if PP disabled or called on last rank.
            Otherwise, tuple of (sampled_tokens, num_sampled, num_rejected):
            - sampled_tokens: shape [num_reqs, max_sample_len]
            - num_sampled: shape [num_reqs]
            - num_rejected: shape [num_reqs]
        """
        if not self.is_enabled or self.produces_final_output:
            return None

        from vllm.distributed.parallel_state import get_pp_group

        pp = get_pp_group()
        sampled_tokens = torch.empty(
            num_reqs, max_sample_len, dtype=torch.int64, device=device
        )
        torch.distributed.broadcast(
            sampled_tokens,
            src=pp.last_rank,
            group=pp.device_group,
        )
        # NOTE: num_sampled/num_rejected are only needed
        # for speculative decoding.
        num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
        torch.distributed.broadcast(
            num_sampled,
            src=pp.last_rank,
            group=pp.device_group,
        )
        num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)
        torch.distributed.broadcast(
            num_rejected,
            src=pp.last_rank,
            group=pp.device_group,
        )
        return sampled_tokens, num_sampled, num_rejected

    # === Model input/output preparation ===

    def prepare_model_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
    ) -> dict[str, Any]:
        """
        Prepare inputs for model.forward().

        Returns dict with keys: input_ids, positions, inputs_embeds,
        and optionally intermediate_tensors.
        """
        if self.receives_raw_inputs:
            return {
                "input_ids": input_ids,
                "positions": positions,
                "inputs_embeds": inputs_embeds,
            }
        else:
            assert intermediate_tensors is not None, (
                "Non-first PP rank requires intermediate_tensors"
            )
            return {
                "input_ids": None,
                "positions": positions,
                "inputs_embeds": None,
                "intermediate_tensors": intermediate_tensors,
            }

    def prepare_output(
        self,
        hidden_states: torch.Tensor | IntermediateTensors,
        kv_connector_output: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Prepare output after model forward.

        Returns:
            - Last rank: torch.Tensor (hidden states for sampling)
            - Non-last rank: IntermediateTensors (to send to next rank)
        """
        if self.produces_final_output:
            # Last rank: extract hidden states for sampling
            if isinstance(hidden_states, IntermediateTensors):
                if "hidden_states" not in hidden_states.tensors:
                    raise ValueError(
                        "IntermediateTensors from model on the last PP rank "
                        "must contain 'hidden_states' tensor."
                    )
                return hidden_states["hidden_states"]
            return hidden_states
        else:
            # Non-last rank: wrap as IntermediateTensors
            if isinstance(hidden_states, IntermediateTensors):
                intermediate = hidden_states
            else:
                intermediate = IntermediateTensors({"hidden_states": hidden_states})
            intermediate.kv_connector_output = kv_connector_output
            return intermediate


def get_pp_handler(parallel_config) -> PPHandler:
    """Factory function to create PPHandler."""
    config = PPConfig.from_parallel_config(parallel_config)
    return PPHandler(config)


# No-op singleton for when PP is disabled
NO_OP_PP_HANDLER = PPHandler(PPConfig(PPRole.NO_PP, 1, 0))
