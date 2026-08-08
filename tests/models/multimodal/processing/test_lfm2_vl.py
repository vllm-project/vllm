# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import ReplicatedLinearWithLoRA
from vllm.model_executor.models.lfm2_vl import (
    Lfm2VLForConditionalGeneration,
    Lfm2VLMultiModalProjector,
)

from ...utils import build_model_context

get_num_mm_encoder_tokens = Lfm2VLForConditionalGeneration.get_num_mm_encoder_tokens
get_num_mm_connector_tokens = Lfm2VLForConditionalGeneration.get_num_mm_connector_tokens


@pytest.mark.parametrize("model_id", ["LiquidAI/LFM2-VL-450M"])
def test_num_mm_tokens_match_real_config(
    model_id,
    monkeypatch: pytest.MonkeyPatch,
):
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 1})
    config = ctx.model_config.hf_config
    stub = SimpleNamespace(config=config)
    factor = config.downsample_factor
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    projector = Lfm2VLMultiModalProjector(config)
    lora_config = LoRAConfig(max_loras=1, max_lora_rank=8)

    for layer in (projector.linear_1, projector.linear_2):
        assert ReplicatedLinearWithLoRA.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=[],
            model_config=config,
        )

    for num_image_tokens in (1, 17, 256, 1024):
        encoder_tokens = get_num_mm_encoder_tokens(stub, num_image_tokens)
        assert encoder_tokens == num_image_tokens * factor**2

        connector_tokens = get_num_mm_connector_tokens(stub, encoder_tokens)
        assert connector_tokens == num_image_tokens


@pytest.mark.parametrize(
    ("downsample_factor", "num_image_tokens"),
    [
        (1, 1),
        (2, 17),
        (3, 256),
        (4, 1024),
    ],
)
def test_num_mm_tokens_roundtrip(downsample_factor, num_image_tokens):
    stub = SimpleNamespace(config=SimpleNamespace(downsample_factor=downsample_factor))

    encoder_tokens = get_num_mm_encoder_tokens(stub, num_image_tokens)
    assert encoder_tokens == num_image_tokens * downsample_factor**2

    connector_tokens = get_num_mm_connector_tokens(stub, encoder_tokens)
    assert connector_tokens == num_image_tokens


def test_num_mm_tokens_zero():
    stub = SimpleNamespace(config=SimpleNamespace(downsample_factor=2))

    assert get_num_mm_encoder_tokens(stub, 0) == 0
    assert get_num_mm_connector_tokens(stub, 0) == 0
