# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kanana-2-v-decide configuration: Qwen3-VL with a gated vision attention and
the settings structured decisions read the checkpoint with. Loads
kakaocorp/kanana-2-v-decide-3b-2611 without trust_remote_code."""

from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
)


class KananaVDecideVisionConfig(Qwen3VLVisionConfig):
    model_type = "kanana_v_decide_vision"

    def __init__(self, gated_attention: bool = True, **kwargs):
        self.gated_attention = gated_attention
        super().__init__(**kwargs)


class KananaVDecideConfig(Qwen3VLConfig):
    model_type = "kanana_v_decide"
    sub_configs = {
        **Qwen3VLConfig.sub_configs,
        "vision_config": KananaVDecideVisionConfig,
    }

    def __init__(
        self,
        embedding_dimension: int = 2560,
        decision_temperature: float = 0.056139365295102696,
        max_length: int = 20480,
        truncate_inputs: bool = True,
        show_criteria: bool = True,
        **kwargs,
    ):
        #: Cosine similarities are divided by this before the softmax.
        self.decision_temperature = decision_temperature
        self.embedding_dimension = embedding_dimension
        self.max_length = max_length
        self.truncate_inputs = truncate_inputs
        self.show_criteria = show_criteria
        super().__init__(**kwargs)
