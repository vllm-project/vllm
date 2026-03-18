# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import (
    SpecDecodeInput,
    SpecDecodePrepareOutput,
    SpecDecodeProposer,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

# Initialize logger
logger = init_logger(__name__)


class MedusaProposer(SpecDecodeProposer):
    """
    Medusa proposer class for generating token sequences
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # Save config parameters
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None, (
            "Speculative config must be set"
        )
        self.spec_config = vllm_config.speculative_config
        self.device = device
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = self.spec_config.draft_model_config.get_hidden_size()
        self.dtype = vllm_config.model_config.dtype

    def prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: "SamplingMetadata",
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: "SpecDecodeMetadata | None",
        common_attn_metadata: "CommonAttentionMetadata",
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
        input_batch: "InputBatch",
    ) -> SpecDecodePrepareOutput:
        """
        Prepare inputs for Medusa draft token proposal.

        Medusa needs to extract hidden states at specific positions based on
        the spec decode metadata.
        """
        # Medusa needs special handling for hidden states
        if sample_hidden_states.shape[0] == len(sampled_token_ids):
            # The input to the target model does not include draft tokens.
            target_hidden_states = sample_hidden_states
        else:
            indices = []
            offset = 0
            assert spec_decode_metadata is not None, (
                "No spec decode metadata for medusa"
            )
            for num_draft, tokens in zip(
                spec_decode_metadata.num_draft_tokens, sampled_token_ids
            ):
                indices.append(offset + len(tokens) - 1)
                offset += num_draft + 1
            indices = torch.tensor(indices, device=hidden_states.device)
            target_hidden_states = sample_hidden_states[indices]

        inputs = SpecDecodeInput(
            target_hidden_states=target_hidden_states,
            sampling_metadata=sampling_metadata,
            slot_mappings=slot_mappings,
        )
        return SpecDecodePrepareOutput(inputs=inputs)

    def propose(
        self,
        inputs: SpecDecodeInput,
    ) -> torch.Tensor:
        """
        Propose draft tokens using Medusa heads.

        Args:
            inputs: Unified input container. Required fields:
                - target_hidden_states: torch.Tensor [num_tokens, hidden_size]
                - sampling_metadata: SamplingMetadata

        Returns:
            Draft tokens tensor of shape [batch_size, num_heads].
        """
        # Extract fields from unified input
        target_hidden_states = inputs.target_hidden_states
        sampling_metadata = inputs.sampling_metadata

        assert isinstance(target_hidden_states, torch.Tensor)
        assert sampling_metadata is not None

        # Generate blocks and compute logits
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)

        # Compute argmax for each Medusa head and stack into a single tensor
        # Shape: [batch_size, num_heads]
        draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)

        return draft_tokens

    def load_model(self, target_model: nn.Module) -> None:
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("medusa_head"):
            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.spec_config.draft_model_config,
            )
        assert not (
            is_mixture_of_experts(self.model)
            and self.vllm_config.parallel_config.enable_eplb
        ), "EPLB for Medusa is not supported"

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):
            self.model(hidden_states)
