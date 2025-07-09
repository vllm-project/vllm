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


class MLPSpeculatorProposer:
    """
    MLPSpeculator proposer class for generating token sequences
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # Save config parameters
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.hidden_size = (vllm_config.speculative_config.
            draft_model_config.get_hidden_size())
        self.num_speculative_tokens = (vllm_config.speculative_config.
            num_speculative_tokens)
        self.dtype = vllm_config.model_config.dtype

    def propose(
        self,
        input_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
        sampling_metadata: SamplingMetadata,
    ) -> list[list[int]]:
        # Generate blocks and compute logits
        draft_tokens = self.model.generate_proposals(input_ids, previous_hidden_states, num_predict_tokens,sampling_metadata)
        return list(map(lambda x: x[0], zip(*[i.sampled_token_ids.tolist() for i in draft_tokens])))

    def load_model(self, target_model: nn.Module) -> None:
        self.model = get_model(vllm_config=self.vllm_config,
                               model_config=self.vllm_config.
                               speculative_config.draft_model_config)

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        input_ids = torch.zeros((self.max_num_seqs, 1), device=self.device)
        hidden_states = torch.zeros((self.max_num_seqs, self.hidden_size),
                            dtype=self.dtype,
                            device=self.device)
        num_predict_tokens = self.num_speculative_tokens
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            self.model.generate_proposals(input_ids, hidden_states, num_predict_tokens, None)