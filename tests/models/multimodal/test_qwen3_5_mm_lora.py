# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "model_cls",
    [Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration],
)
def test_qwen3_5_mm_lora_helpers(model_cls):
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(
                spatial_merge_size=2,
            ),
        ),
    )

    mapping = model_cls.get_mm_mapping(model)
    assert mapping.language_model == ["language_model"]
    assert mapping.connector == ["visual.merger", "visual.deepstack_merger_list"]
    assert mapping.tower_model == ["visual."]

    assert model_cls.get_num_mm_encoder_tokens(model, 7) == 28
    assert model_cls.get_num_mm_connector_tokens(model, 28) == 7
