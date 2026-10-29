# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.model_executor.models.opt import OPTForCausalLM
from vllm.sequence import IntermediateTensors


class MyOPTForCausalLM(OPTForCausalLM):
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            logits.zero_()
            logits[:, 0] += 1.0
        return logits


class MyOPTRawInputTokensForCausalLM(MyOPTForCausalLM):
    # Like models whose decoder layers read token ids, this model needs
    # input_ids on every pipeline-parallel stage, not only the first.
    requires_raw_input_tokens = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        assert input_ids is not None, "requires_raw_input_tokens model got no input_ids"
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
