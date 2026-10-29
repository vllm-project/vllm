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


def test_v2_flashinfer_vision_falls_back_without_changing_dash_backend():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    config = EngineArgs(
        model="baidu/Unlimited-OCR-2",
        attention_config={"backend": "FLASHINFER"},
        mm_encoder_attn_backend="FLASHINFER",
    ).create_engine_config()
    assert config.model_config.multimodal_config.mm_encoder_attn_backend == (
        AttentionBackendEnum.FLASH_ATTN
    )
    assert config.model_config.hf_text_config.unlimited_ocr2_attention_backend == (
        "FLASHINFER"
    )


@pytest.mark.asyncio
async def test_v2_async_rendering_rejects_bad_layout_and_preserves_image_cache():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.exceptions import VLLMValidationError
    from vllm.renderers import renderer_from_config

    config = EngineArgs(
        model="baidu/Unlimited-OCR-2",
        max_model_len=4096,
        mm_processor_kwargs={"crop_mode": False},
    ).create_engine_config()
    renderer = renderer_from_config(config)
    image = Image.new("RGB", (1024, 1024), "white")
    try:
        for text, images in [("Parse.", []), ("<image>Parse.", [image])]:
            prompt = {"prompt_token_ids": renderer.tokenizer.encode(text)}
            if images:
                prompt["multi_modal_data"] = {"image": images}
            with pytest.raises(VLLMValidationError, match="<PAGE><image>"):
                await renderer.process_for_engine_async(prompt, 0)
        prompt = {
            "prompt_token_ids": renderer.tokenizer.encode("<PAGE><image>Parse."),
            "multi_modal_data": {"image": [image]},
        }
        first = await renderer.process_for_engine_async(prompt, 0)
        cached = await renderer.process_for_engine_async(prompt, 0)
        assert first["prompt_token_ids"] == cached["prompt_token_ids"]
        assert len(first["mm_placeholders"]["image"]) == 1
    finally:
        renderer.shutdown()


@pytest.mark.parametrize(
    "env, kwargs, message",
    [
        ("0", {}, "requires Model Runner V2"),
        ("1", {"quantization": "fp8"}, "does not support quantized weights"),
    ],
)
def test_v2_rejects_unsupported_runtime_before_loading(
    monkeypatch, env, kwargs, message
):
    from vllm.engine.arg_utils import EngineArgs

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", env)
    with pytest.raises(ValueError, match=message):
        EngineArgs(model="baidu/Unlimited-OCR-2", **kwargs).create_engine_config()


def test_v2_preprocessed_features_validate_page_layout_at_admission():
    """Pre-rendered scale-out inputs still need admission validation."""
    from copy import deepcopy

    from vllm.engine.arg_utils import EngineArgs
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor

    config = EngineArgs(
        model="baidu/Unlimited-OCR-2",
        max_model_len=4096,
        mm_processor_kwargs={"crop_mode": False},
    ).create_engine_config()
    processor = InputProcessor(config)
    renderer = processor.renderer
    try:
        rendered = renderer.process_for_engine(
            {
                "prompt_token_ids": renderer.tokenizer.encode("<PAGE><image>Parse."),
                "multi_modal_data": {
                    "image": [Image.new("RGB", (1024, 1024), "white")]
                },
            },
            0,
        )
        # InputProcessor receives already expanded image runs, as scale-out does.
        valid = processor.process_inputs(
            "valid", rendered, SamplingParams(), ("generate",)
        )
        assert valid.prompt_token_ids == rendered["prompt_token_ids"]
        page = config.model_config.hf_config.sparse_attention_config["page_token_id"]
        for ids in (
            [x for x in rendered["prompt_token_ids"] if x != page],
            [page] + rendered["prompt_token_ids"],
        ):
            invalid = deepcopy(rendered)
            invalid["prompt_token_ids"] = ids
            with pytest.raises(VLLMValidationError, match="<PAGE><image>"):
                processor.process_inputs(
                    "invalid", invalid, SamplingParams(), ("generate",)
                )
    finally:
        renderer.shutdown()
