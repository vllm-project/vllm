# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.ernie45_vl import (
    Ernie4_5_VLMoeForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.utils.torch_utils import set_default_torch_dtype

from ...registry import HF_EXAMPLE_MODELS
from ...utils import build_model_context, dummy_hf_overrides


@pytest.mark.parametrize(
    ("model_id", "max_model_len"),
    [
        ("zai-org/GLM-4.1V-9B-Thinking", 16384),
        ("AIDC-AI/Ovis2.5-2B", 32768),
        ("OpenGVLab/InternVL3-1B", 16384),
        ("internlm/Intern-S1", 16384),
        ("baidu/ERNIE-4.5-VL-28B-A3B-PT", 32768),
    ],
)
def test_video_budget_matches_profiled_video(model_id: str, max_model_len: int):
    ctx = build_model_context(
        model_id,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    dummy = processor.get_dummy_mm_inputs({"video": 1})
    num_embeds = sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["video"])

    assert budget["video"] == num_embeds


@pytest.mark.parametrize(
    ("model_id", "max_model_len"),
    [("baidu/ERNIE-4.5-VL-28B-A3B-PT", 16384)],
)
def test_video_budget_ignores_image_count(model_id: str, max_model_len: int):
    ctx = build_model_context(
        model_id,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    with_image = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    video_only = get_mm_max_toks_per_item(processor, {"video": 1}, None)

    assert with_image["video"] == video_only["video"]


ERNIE = "baidu/ERNIE-4.5-VL-28B-A3B-PT"


def _video_embeds(outputs) -> int:
    return sum(p.get_num_embeds() for p in outputs["mm_placeholders"]["video"])


@pytest.mark.parametrize(
    ("hint_width", "hint_height", "width", "height", "mm_processor_kwargs"),
    [
        (1288, 728, 1288, 728, None),
        (728, 1288, 728, 1288, None),
        (1288, None, 1288, 728, None),
        (None, 1288, 728, 1288, None),
        (1920, None, 1288, 728, None),
        (1920, 1080, 1288, 728, None),
        (896, None, 896, 448, {"video_min_pixels": 401408, "video_max_pixels": 401408}),
    ],
)
def test_ernie45_vl_dummy_video_follows_size_hint(
    hint_width: int | None,
    hint_height: int | None,
    width: int,
    height: int,
    mm_processor_kwargs: dict | None,
):
    hint = {"count": 1, "num_frames": 2, "width": hint_width, "height": hint_height}
    ctx = build_model_context(
        ERNIE,
        model_config_kwargs={"max_model_len": 32768},
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={
            "image": 1,
            "video": {k: v for k, v in hint.items() if v is not None},
        },
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    frames = np.zeros((2, height, width, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": 1.0,
        "total_num_frames": 2,
        "frames_indices": [0, 1],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    inputs = ProcessorInputs(
        prompt=processor.info.get_tokenizer().encode("<|video@placeholder|>"),
        mm_data_items=processor.info.parse_mm_data({"video": [(frames, metadata)]}),
    )
    real = processor.apply(inputs, TimingContext(enabled=False))
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    assert _video_embeds(dummy) == _video_embeds(real)


def test_ernie45_vl_video_budget_ignores_image_pixel_limit():
    ctx = build_model_context(
        ERNIE,
        model_config_kwargs={"max_model_len": 2048},
        mm_processor_kwargs={"max_pixels": 153664},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    assert budget["video"] == _video_embeds(dummy)


def _tiny_ernie45_vl(hf_config):
    hf_config = dummy_hf_overrides(
        hf_config, model_arch="Ernie4_5_VLMoeForConditionalGeneration"
    )
    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.depth = 2
    return hf_config


@pytest.mark.usefixtures("dist_init")
def test_ernie45_vl_encodes_videos_in_frame_pairs():
    model_info = HF_EXAMPLE_MODELS.find_hf_info(ERNIE)
    model_config = ModelConfig(
        ERNIE,
        trust_remote_code=model_info.trust_remote_code,
        revision=model_info.revision,
        dtype="float32",
        hf_overrides=_tiny_ernie45_vl,
        max_model_len=4096,
    )
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(torch.float32):
        model = Ernie4_5_VLMoeForConditionalGeneration(vllm_config=vllm_config)
    torch.manual_seed(0)
    for module in (model.vision_model, model.resampler_model):
        for param in module.parameters():
            torch.nn.init.normal_(param, std=0.02)
    process_weights_after_loading(model, model_config, torch.device("cpu"))

    grid_thw = torch.tensor([[4, 8, 12], [2, 4, 8]])
    pixel_values = torch.randn(int(grid_thw.prod(-1).sum()), 588)
    vision_calls = []
    hook = model.vision_model.register_forward_pre_hook(
        lambda _, args: vision_calls.append(len(args[0]))
    )
    with torch.inference_mode():
        embeds = model.embed_multimodal(
            pixel_values_videos=pixel_values, video_grid_thw=grid_thw
        )
        hook.remove()
        expected = model.resampler_model(
            model._vision_forward(pixel_values, grid_thw), grid_thw
        )

    assert vision_calls == [2 * 8 * 12, 2 * 8 * 12, 2 * 4 * 8]
    assert embeds is not None
    assert [len(e) for e in embeds] == [4 * 8 * 12 // 8, 2 * 4 * 8 // 8]
    torch.testing.assert_close(torch.cat(embeds), expected)
