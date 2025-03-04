# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import CompilationLevel, VllmConfig
from vllm.forward_context import set_forward_context

class MultiTokenProposer:

    def __init__(
        self, mtp_model:nn.Module, mtp_model_vllm_config: VllmConfig):
        self.mtp_model = mtp_model
        self.mtp_model_vllm_config = mtp_model_vllm_config


    def propose(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: FlashAttentionMetadata,
    ) -> Optional[np.ndarray]:
    
        with set_forward_context(attention_metadata, self.mtp_model_vllm_config):
            mtp_hidden_states = self.mtp_model(
                input_ids=input_ids,
                positions=positions,
                previous_hidden_states=hidden_states,
                intermediate_tensors=None,
                inputs_embeds=None,
            )
            print('mtp_hidden_states ' + str(mtp_hidden_states))
        return np.array([])
