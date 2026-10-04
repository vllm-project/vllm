# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)


@pytest.mark.parametrize(
    "model_cls",
    [Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration],
    ids=["dense", "moe"],
)
@pytest.mark.parametrize("video_pruning_method", ["evs", "vidcom2"])
def test_video_pruning_config_is_initialized(model_cls, video_pruning_method):
    multimodal_config = MultiModalConfig(
        video_pruning_rate=0.3,
        video_pruning_method=video_pruning_method,
    )
    model_config = Mock(
        hf_config=SimpleNamespace(
            vision_config=SimpleNamespace(out_hidden_size=64),
        ),
        multimodal_config=multimodal_config,
    )
    vllm_config = Mock(model_config=model_config, quant_config=None)

    with (
        patch("vllm.model_executor.models.qwen3_5.cached_tokenizer_from_config"),
        patch("vllm.model_executor.models.qwen3_5.Qwen3_VisionTransformer"),
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5ForCausalLM") as dense_lm,
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5MoeForCausalLM") as moe_lm,
        patch.object(model_cls, "_mark_tower_model", return_value=nullcontext()),
        patch.object(model_cls, "_mark_language_model", return_value=nullcontext()),
        patch.object(Qwen3_5MoeForConditionalGeneration, "set_moe_parameters"),
    ):
        dense_lm.return_value.make_empty_intermediate_tensors = Mock()
        moe_lm.return_value.make_empty_intermediate_tensors = Mock()
        model = model_cls(vllm_config=vllm_config)

    assert model.video_pruning_method == video_pruning_method
    assert model.video_pruning_rate == 0.3
    assert model.is_multimodal_pruning_enabled
