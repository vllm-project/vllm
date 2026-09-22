# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only IBM Granite Speech Plus model."""

import torch
from transformers import PretrainedConfig

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from .granite_speech import (
    GraniteSpeechCTCEncoder,
    GraniteSpeechDummyInputsBuilder,
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechMultiModalProcessingInfo,
    GraniteSpeechMultiModalProcessor,
)

ISO639_1_SUPPORTED_LANGS = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "es": "Spanish",
}


class GraniteSpeechPlusCTCEncoder(GraniteSpeechCTCEncoder):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_linear(hidden_states)
        # cat_hidden_layers selects non-negative layer indices (0 = encoder
        # input, N = output of layer N) whose hidden states are concatenated
        # along the feature dim *in addition to* the final hidden states,
        # which are always appended last.
        cat_layers = set(self.config.cat_hidden_layers or [])
        exported_hidden_states = []

        if 0 in cat_layers:
            exported_hidden_states.append(hidden_states)

        for idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=self.attention_dists)

            # Skip the final layer here since its output is always appended
            # below; capturing it twice would double-append.
            if idx in cat_layers and idx != self.num_layers:
                exported_hidden_states.append(hidden_states)

            if idx == self.num_layers // 2:
                hidden_states_mid = hidden_states.clone()
                hidden_states_mid, _ = self.out(hidden_states_mid)
                hidden_states_mid = self.softmax(hidden_states_mid)
                hidden_states_mid, _ = self.out_mid(hidden_states_mid)
                hidden_states += hidden_states_mid

        if exported_hidden_states:
            hidden_states = torch.cat([*exported_hidden_states, hidden_states], dim=-1)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    GraniteSpeechMultiModalProcessor,
    info=GraniteSpeechMultiModalProcessingInfo,
    dummy_inputs=GraniteSpeechDummyInputsBuilder,
)
class GraniteSpeechPlusForConditionalGeneration(GraniteSpeechForConditionalGeneration):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    def _build_encoder(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> GraniteSpeechCTCEncoder:
        return GraniteSpeechPlusCTCEncoder(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )
