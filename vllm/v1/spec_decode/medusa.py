# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.v1.spec_decode.metadata import ProposeInput, SpecDecodeProposer

if typing.TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

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
        sampled_token_ids: torch.Tensor | list[list[int]],
        input_batch: "InputBatch",
        **kwargs,
    ) -> ProposeInput:
        sample_hidden_states = kwargs["sample_hidden_states"]
        spec_decode_metadata = kwargs["spec_decode_metadata"]

        if sample_hidden_states.shape[0] == len(sampled_token_ids):
            # The input to the target model does not include draft tokens.
            hidden_states = sample_hidden_states
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
            indices = torch.tensor(indices, device=self.device)
            hidden_states = sample_hidden_states[indices]

        return ProposeInput(
            sampled_token_ids=sampled_token_ids,
            input_batch=input_batch,
            hidden_states=hidden_states,
        )

    def propose(self, inputs: ProposeInput) -> tuple[torch.Tensor, None]:
        # Generate blocks and compute logits
        blocks = self.model(inputs.hidden_states)
        logits = self.model.compute_logits(blocks)

        # Compute argmax for each Medusa head and stack into a single tensor
        # Shape: [batch_size, num_heads]
        draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)

        return draft_tokens, None

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
