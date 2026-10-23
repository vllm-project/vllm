# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs


@pytest.fixture
def processor():
    config = ModelConfig(
        model="deepseek-ai/deepseek-vl2-tiny",
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    return MULTIMODAL_REGISTRY.create_processor(config)


@pytest.mark.parametrize("use_uuid", [False, True])
def test_image_cache_identity_depends_on_cropping(processor, use_uuid: bool):
    tokenizer = processor.info.get_tokenizer()
    images = [
        Image.new("RGB", (768, 768), color)
        for color in ("red", "green", "blue", "yellow")
    ]
    results = []
    for count in (1, 2, 3, 4):
        inputs = ProcessorInputs(
            prompt=tokenizer.encode("<image>\n" * count + "Describe."),
            mm_data_items=processor.info.parse_mm_data({"image": images[:count]}),
            mm_uuid_items={"image": [f"image-{i}" for i in range(count)]}
            if use_uuid
            else None,
        )
        results.append(processor.apply(inputs, TimingContext(enabled=False)))

    hashes = [result["mm_hashes"]["image"][0] for result in results]
    lengths = [result["mm_placeholders"]["image"][0].length for result in results]
    assert lengths[0] == lengths[1] > lengths[2] == lengths[3]
    assert hashes[0] == hashes[1]
    assert hashes[2] == hashes[3]
    assert hashes[0] != hashes[2]


@pytest.mark.parametrize("use_uuid", [False, True])
def test_image_embedding_cache_identity_ignores_image_count(processor, use_uuid: bool):
    tokenizer = processor.info.get_tokenizer()
    embedding = torch.ones(8, processor.info.ctx.model_config.get_hidden_size())
    hashes = []
    for count in (1, 3):
        inputs = ProcessorInputs(
            prompt=tokenizer.encode("<image>\n" * count + "Describe."),
            mm_data_items=processor.info.parse_mm_data({"image": [embedding] * count}),
            mm_uuid_items={"image": [f"embedding-{i}" for i in range(count)]}
            if use_uuid
            else None,
        )
        result = processor.apply(inputs, TimingContext(enabled=False))
        assert result["mm_placeholders"]["image"][0].length == embedding.shape[0]
        hashes.append(result["mm_hashes"]["image"][0])

    assert hashes[0] == hashes[1]
    if use_uuid:
        assert hashes[0] == "embedding-0"
