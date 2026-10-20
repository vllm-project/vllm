# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 MTP: Step-3.5's MTP with a Step-5 decoder block."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from vllm.model_executor.models.step3p5_mtp import (
    Step3p5AMultiTokenPredictor,
    Step3p5AMultiTokenPredictorLayer,
    Step3p5MTP,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.models.step5.amd.model import Step5DecoderLayer, _use_fp32_norm_weights


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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _use_fp32_norm_weights(self, vllm_config.model_config.hf_config)

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )


class Step5MTP(Step3p5MTP, SupportsMultiModal):
    supports_multimodal_embeddings = True
    predictor_cls = Step5MultiTokenPredictor

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids, self.model.embed_input_ids, is_multimodal=is_multimodal
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=_require_is_multimodal(is_multimodal),
        )

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)
