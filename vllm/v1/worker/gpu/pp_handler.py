# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism handler for V2 Model Runner."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch

from vllm.sequence import IntermediateTensors


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
    """Handles PP input/output transformations for V2 Model Runner.

    Design principle: All public methods are safe to call regardless of PP state.
    Methods check `is_enabled` internally and return appropriate no-op values
    when PP is disabled or when called on an inapplicable rank. This makes the
    handler easier to use and harder to misuse - callers don't need to guard
    every call with `if pp_handler.is_enabled`.
    """

    def __init__(self, config: PPConfig):
        self.config = config
        # Async broadcast state (for future optimization)
        self._broadcast_handle: torch.distributed.Work | None = None
        self._broadcast_buffer: torch.Tensor | None = None
        self._broadcast_device: torch.device | None = None
        self._broadcast_num_reqs: int = 0

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

    # === Sync API (blocking, used by default) ===

    def blocking_broadcast_tokens(self, sampler_output) -> None:
        """Last rank broadcasts sampled tokens to all PP ranks (blocking).

        Safe to call on any rank - no-ops if PP is disabled or if called
        on a non-last rank.

        Args:
            sampler_output: SamplerOutput from sampling. Only used by last rank.
        """
        # No-op if PP disabled or not the last rank (only last rank broadcasts)
        if not self.is_enabled or not self.produces_final_output:
            return

        from vllm.distributed.parallel_state import get_pp_group

        pp = get_pp_group()
        # Broadcast the first sampled token per request (main decode token)
        tokens_to_broadcast = sampler_output.sampled_token_ids[:, 0].contiguous()
        torch.distributed.broadcast(
            tokens_to_broadcast,
            src=pp.last_rank,
            group=pp.device_group,
        )

    def blocking_receive_tokens(
        self,
        num_reqs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Non-last ranks receive sampled tokens from last PP rank (blocking).

        Safe to call on any rank - returns None if PP is disabled or if called
        on the last rank (which doesn't need to receive).

        Args:
            num_reqs: Number of requests in the batch.
            device: Device to create tensors on.

        Returns:
            None if PP disabled or called on last rank.
            Otherwise, tuple of (sampled_tokens, num_sampled, num_rejected):
            - sampled_tokens: shape [num_reqs, 1] for postprocess
            - num_sampled: shape [num_reqs], all ones
            - num_rejected: shape [num_reqs], all zeros
        """
        # No-op if PP disabled or on last rank (last rank samples, doesn't receive)
        if not self.is_enabled or self.produces_final_output:
            return None

        from vllm.distributed.parallel_state import get_pp_group

        pp = get_pp_group()
        recv = torch.empty(num_reqs, dtype=torch.int64, device=device)
        torch.distributed.broadcast(
            recv,
            src=pp.last_rank,
            group=pp.device_group,
        )

        # Format for postprocess: [num_reqs, 1] sampled tokens
        sampled_tokens = recv.unsqueeze(1)
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)
        num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        return sampled_tokens, num_sampled, num_rejected

    # === Async API (for future optimization with overlap) ===
    # TODO: Use this API to overlap token broadcast with prompt_logprobs
    # computation or other CPU work. Currently unused - blocking API is used.

    def start_token_broadcast(
        self,
        sampler_output,
        num_reqs: int,
        device: torch.device,
    ) -> None:
        """Start async broadcast of sampled tokens from last rank.

        Safe to call on any rank - no-ops if PP is disabled. All PP ranks
        should call this together (collective operation). Last rank broadcasts,
        non-last ranks receive. Call wait_token_broadcast() to complete.

        Args:
            sampler_output: SamplerOutput from sampling (only used by last rank,
                can be None for non-last ranks).
            num_reqs: Number of requests in the batch.
            device: Device to create receive buffer on.
        """
        # No-op if PP disabled
        if not self.is_enabled:
            self._broadcast_handle = None
            return

        from vllm.distributed.parallel_state import get_pp_group

        pp = get_pp_group()

        if self.produces_final_output:
            # Last rank: broadcast sampled tokens
            self._broadcast_buffer = sampler_output.sampled_token_ids[:, 0].contiguous()
        else:
            # Non-last ranks: prepare receive buffer
            self._broadcast_buffer = torch.empty(
                num_reqs, dtype=torch.int64, device=device
            )

        self._broadcast_handle = torch.distributed.broadcast(
            self._broadcast_buffer,
            src=pp.last_rank,
            group=pp.device_group,
            async_op=True,
        )
        self._broadcast_device = device
        self._broadcast_num_reqs = num_reqs

    def wait_token_broadcast(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Wait for async token broadcast to complete.

        Safe to call always - returns None if no pending broadcast (PP disabled
        or start_token_broadcast not called).

        Returns:
            None if PP disabled, no pending broadcast, or called on last rank.
            For non-last ranks: tuple of (sampled_tokens, num_sampled, num_rejected)
                in the format expected by postprocess().
        """
        # No-op if no pending broadcast
        if self._broadcast_handle is None:
            return None

        self._broadcast_handle.wait()
        self._broadcast_handle = None

        if self.produces_final_output:
            # Last rank: nothing to return (it did the sampling)
            return None

        # Non-last ranks: format for postprocess
        assert self._broadcast_buffer is not None
        sampled_tokens = self._broadcast_buffer.unsqueeze(1)
        num_sampled = torch.ones(
            self._broadcast_num_reqs, dtype=torch.int32, device=self._broadcast_device
        )
        num_rejected = torch.zeros(
            self._broadcast_num_reqs, dtype=torch.int32, device=self._broadcast_device
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
