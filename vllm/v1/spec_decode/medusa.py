# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.sample.metadata import SamplingMetadata

# Initialize logger
logger = init_logger(__name__)


class MedusaProposer:
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
        self.device = device
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        self.hidden_size = vllm_config.speculative_config.\
            draft_model_config.get_hidden_size(
        )
        self.dtype = vllm_config.model_config.dtype

    def propose(
        self,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> list[list[int]]:
        # Generate blocks and compute logits
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)

        # Get draft tokens and transpose the result
        # TODO(woosuk): OPTIMIZATION: Return GPU tensor without GPU-CPU
        # synchronization.
        draft_tokens = [logit.argmax(dim=-1).tolist() for logit in logits]
        return [list(row) for row in zip(*draft_tokens)]

    def load_model(self, target_model: nn.Module) -> None:
        from vllm.compilation.backends import set_model_tag
        with set_model_tag("medusa_head"):
            self.model = get_model(vllm_config=self.vllm_config,
                                   model_config=self.vllm_config.
                                   speculative_config.draft_model_config)

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size),
                                    dtype=self.dtype,
                                    device=self.device)
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            self.model(hidden_states)
