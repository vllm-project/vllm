# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.transformers_utils.processors.deepseek_ocr import calculate_aspect_ratios


@pytest.mark.parametrize("overrides, expected", [({}, 128), ({"rswa_window": 0}, None)])
def test_v2_output_window_reaches_runtime_config(overrides, expected):
    """Disabling output eviction must also disable it in the derived runtime config."""
    config = ModelConfig(model="baidu/Unlimited-OCR-2", hf_overrides=overrides)
    assert config.rswa_window == expected
    assert config.hf_text_config.rswa_window == expected


def test_v2_base_mode_preserves_page_aspect_ratio():
    config = ModelConfig(
        model="baidu/Unlimited-OCR-2", max_model_len=4096, enforce_eager=True
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    hf_processor = processor.info.get_hf_processor()
    image = Image.new("RGB", (384, 768), "red")
    _, pixels, *_ = hf_processor.tokenize_with_images(
        "<PAGE><image>Multi page parsing.", [image], cropping=False
    )
    # A 1:2 page occupies the middle 512 columns of a 1024-square global view.
    assert pixels.shape == (1, 3, 1024, 1024)
    assert torch.all(pixels[0, 0, :, 256:768] == 1)
    assert torch.all(pixels[0, 1, :, 256:768] == -1)
    assert torch.all(pixels[0, :, :, :256].abs() < 0.01)
    assert torch.all(pixels[0, :, :, 768:].abs() < 0.01)


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("model", ["baidu/Unlimited-OCR", "baidu/Unlimited-OCR-2"])
def test_profiling_tokens_cover_all_crop_grids(model):
    """Every supported crop grid fits the profiled image token budget."""
    config = ModelConfig(
        model=model, max_model_len=8192, limit_mm_per_prompt={"image": 1}
    )
    mm_processor = MULTIMODAL_REGISTRY.create_processor(config)
    hf_processor = mm_processor.info.get_hf_processor()
    dummy_inputs = mm_processor.get_dummy_mm_inputs({"image": 1})
    budget = dummy_inputs["mm_placeholders"]["image"][0].get_num_embeds()

    for width_tiles, height_tiles in calculate_aspect_ratios(
        max_num=hf_processor.max_crops
    ):
        num_tokens = mm_processor.info.get_num_image_tokens(
            image_width=width_tiles * hf_processor.image_size,
            image_height=height_tiles * hf_processor.image_size,
        )
        assert num_tokens <= budget, (width_tiles, height_tiles, num_tokens, budget)


@pytest.mark.parametrize("model", ["baidu/Unlimited-OCR", "baidu/Unlimited-OCR-2"])
@pytest.mark.parametrize("use_uuid", [False, True])
@pytest.mark.parametrize("use_embeds", [False, True])
def test_image_cache_identity_across_image_counts(
    model: str, use_uuid: bool, use_embeds: bool
):
    config = ModelConfig(
        model=model,
        limit_mm_per_prompt={"image": 3},
        enable_mm_embeds=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    tokenizer = processor.info.get_tokenizer()
    if use_embeds:
        images = [torch.ones(8, config.get_hidden_size())] * 3
    else:
        images = [
            Image.new("RGB", (1280, 800), color) for color in ("red", "green", "blue")
        ]
    hashes, lengths = [], []
    for count in (1, 2, 3):
        inputs = ProcessorInputs(
            prompt=tokenizer.encode("<image>\n" * count + "Describe."),
            mm_data_items=processor.info.parse_mm_data({"image": images[:count]}),
            mm_uuid_items={"image": [f"image-{i}" for i in range(count)]}
            if use_uuid
            else None,
        )
        if use_embeds:
            hashes.append(processor._get_mm_hashes(inputs)["image"][0])
            continue
        result = processor.apply(inputs, TimingContext(enabled=False))
        hashes.append(result["mm_hashes"]["image"][0])
        lengths.append(result["mm_placeholders"]["image"][0].length)
    assert hashes[1] == hashes[2]
    if not use_embeds and model.endswith("-2"):
        assert lengths[0] == lengths[1] == lengths[2]
        assert lengths[0] > 273
        assert hashes[0] == hashes[1]
    elif use_embeds:
        assert hashes[0] == hashes[1]
        if use_uuid:
            assert hashes[0] == "image-0"
    else:
        assert lengths[0] > lengths[1] == lengths[2]
        assert hashes[0] != hashes[1]


@pytest.mark.parametrize("kwargs_on_init", [False, True])
def test_image_cache_identity_without_cropping(kwargs_on_init: bool):
    mm_processor_kwargs = {"crop_mode": False}
    config = ModelConfig(
        model="baidu/Unlimited-OCR",
        limit_mm_per_prompt={"image": 2},
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    images = [Image.new("RGB", (1280, 800), color) for color in ("red", "blue")]
    cache = MultiModalProcessorOnlyCache(config)
    results = []
    for count in (1, 2, 1):
        inputs = ProcessorInputs(
            prompt=processor.info.get_tokenizer().encode("<image>\n" * count),
            mm_data_items=processor.info.parse_mm_data({"image": images[:count]}),
            hf_processor_mm_kwargs={} if kwargs_on_init else mm_processor_kwargs,
            cache=cache,
        )
        results.append(processor.apply(inputs, TimingContext(enabled=False)))
    assert results[0]["mm_hashes"]["image"][0] == results[1]["mm_hashes"]["image"][0]
    assert (
        results[0]["mm_placeholders"]["image"][0].length
        == results[1]["mm_placeholders"]["image"][0].length
    )
    assert results[0]["prompt_token_ids"] == results[2]["prompt_token_ids"]


def test_v2_gundam_matches_official_wide_page_grid():
    """The checkpoint's 16-crop preset selects 5x3, rather than the v1 7x4 grid."""
    config = ModelConfig(model="baidu/Unlimited-OCR-2", max_model_len=8192)
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    images = [Image.new("RGB", (1920, 1080), color) for color in ("red", "blue")]
    inputs = ProcessorInputs(
        prompt=processor.info.get_tokenizer().encode(
            "<PAGE><image><PAGE><image>Multi page parsing."
        ),
        mm_data_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={"crop_mode": True},
    )
    result = processor.apply(inputs, TimingContext(enabled=False))
    assert [p.length for p in result["mm_placeholders"]["image"]] == [1803, 1803]
    assert len(result["prompt_token_ids"]) == 3613
