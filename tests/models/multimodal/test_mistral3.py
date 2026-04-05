# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.model_executor.models.mistral3 import Mistral3ForConditionalGeneration


@pytest.mark.skip_global_cleanup
def test_mistral3_passes_inner_lm_architecture():
    text_config = SimpleNamespace(
        architectures=None,
        model_type="ministral3",
        hidden_size=5120,
    )
    vision_config = SimpleNamespace(
        hidden_act="gelu",
        hidden_size=1024,
        patch_size=14,
    )
    hf_config = SimpleNamespace(
        text_config=text_config,
        vision_config=vision_config,
        projector_hidden_act="gelu",
        spatial_merge_size=2,
        multimodal_projector_bias=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            multimodal_config=SimpleNamespace(
                mm_encoder_only=False,
                get_limit_per_prompt=lambda _: 1,
            ),
        ),
        quant_config=None,
    )
    language_model = SimpleNamespace(
        make_empty_intermediate_tensors=Mock(),
    )

    with (
        patch(
            "vllm.model_executor.models.mistral3.init_vision_tower_for_mistral3",
            return_value=Mock(),
        ),
        patch(
            "vllm.model_executor.models.mistral3.Mistral3MultiModalProjector",
            return_value=Mock(),
        ),
        patch(
            "vllm.model_executor.models.mistral3.init_vllm_registered_model",
            return_value=language_model,
        ) as init_language_model,
    ):
        Mistral3ForConditionalGeneration(vllm_config=vllm_config)

    init_language_model.assert_called_once_with(
        vllm_config=vllm_config,
        hf_config=text_config,
        architectures=["Ministral3ForCausalLM"],
        prefix="language_model",
    )
