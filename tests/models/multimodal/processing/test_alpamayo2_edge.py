# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.alpamayo2_edge import (
    Alpamayo2EdgeForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "nvidia/Alpamayo2-Edge"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture(scope="module")
def processor():
    ctx = build_model_context(MODEL_ID, limit_mm_per_prompt={"image": 2})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def _grid_and_tokens(processor, image: Image.Image, mm_kwargs: dict):
    processed = processor(
        IMAGE_PLACEHOLDER,
        mm_items=processor.info.parse_mm_data({"image": [image]}),
        hf_processor_mm_kwargs=mm_kwargs,
    )
    grid_thw = processed["mm_kwargs"].get_data()["image_grid_thw"]
    return grid_thw.tolist(), processed["mm_placeholders"]["image"][0].length


def test_images_use_the_checkpoint_pixel_budget(processor) -> None:
    # A 1920x1080 camera frame is resized to 576x320 (163,840 to 196,608
    # pixels), i.e. 20x36 patches and 180 image tokens.
    image = Image.new("RGB", (1920, 1080))
    assert _grid_and_tokens(processor, image, {}) == ([[1, 20, 36]], 180)


def test_pre_resized_frames_keep_their_size(processor) -> None:
    # The driving prompt resizes side cameras below the default budget; with
    # resizing disabled, a 416x224 frame keeps its 14x26 patches.
    image = Image.new("RGB", (416, 224))
    assert _grid_and_tokens(processor, image, {"do_resize": False}) == (
        [[1, 14, 26]],
        91,
    )


def test_video_input_is_rejected(processor) -> None:
    assert set(processor.info.supported_mm_limits) == {"image"}
    with pytest.raises(VLLMValidationError, match="At most 0 video"):
        processor.info.validate_num_items("video", 1)


def test_weights_mapper_splits_text_layers_and_drops_expert() -> None:
    mapper = Alpamayo2EdgeForConditionalGeneration.hf_to_vllm_mapper
    names = [
        "vlm.model.language_model.layers.3.input_layernorm.weight",
        "vlm.model.language_model.layers.3.self_attn.q_proj.weight",
        "vlm.model.language_model.layers.3.self_attn.o_proj.weight",
        "vlm.model.language_model.layers.3.post_attention_layernorm.weight",
        "vlm.model.language_model.layers.3.mlp.fc1.weight",
        "vlm.model.language_model.layers.3.mlp.fc2.weight",
        "vlm.model.language_model.norm.weight",
        "vlm.model.visual.encoder.layers.0.mlp.fc1.weight",
        "vlm.model.projector.linear_fc1.weight",
        "vlm.lm_head.weight",
        "expert.expert.layers.0.mlp.fc1.weight",
        "expert.action_out_proj.weight",
    ]
    assert mapper.apply_list(names) == [
        "language_model.model.layers.6.norm.weight",
        "language_model.model.layers.6.mixer.qkv_proj.weight",
        "language_model.model.layers.6.mixer.o_proj.weight",
        "language_model.model.layers.7.norm.weight",
        "language_model.model.layers.7.mixer.up_proj.weight",
        "language_model.model.layers.7.mixer.down_proj.weight",
        "language_model.model.norm_f.weight",
        "visual.encoder.encoder.layers.0.mlp.fc1.weight",
        "visual.projector.linear_fc1.weight",
        "language_model.lm_head.weight",
    ]
