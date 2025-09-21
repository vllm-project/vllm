# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch

from vllm.model_executor.models.opt import OPTForCausalLM


class MyOPTForCausalLM(OPTForCausalLM):

    def compute_logits(self,
                       hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        # this dummy model always predicts the first token
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            logits.zero_()
            logits[:, 0] += 1.0
        return logits
