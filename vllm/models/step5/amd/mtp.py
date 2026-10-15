# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 MTP: Step-3.5's MTP with a Step-5 decoder block."""

import torch

from vllm.model_executor.models.step3p5_mtp import (
    Step3p5AMultiTokenPredictor,
    Step3p5AMultiTokenPredictorLayer,
    Step3p5MTP,
)
from vllm.models.step5.amd.model import Step5DecoderLayer


class Step5MultiTokenPredictorLayer(Step3p5AMultiTokenPredictorLayer):
    decoder_layer_cls = Step5DecoderLayer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                dim=-1,
            )
        )
        return self.mtp_block(positions, hidden_states.float()).to(hidden_states.dtype)


class Step5MultiTokenPredictor(Step3p5AMultiTokenPredictor):
    layer_cls = Step5MultiTokenPredictorLayer


class Step5MTP(Step3p5MTP):
    predictor_cls = Step5MultiTokenPredictor
