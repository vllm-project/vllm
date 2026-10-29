# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.alpamayo2_super import (
    Alpamayo2SuperForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "nvidia/Alpamayo2-Super"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture(scope="module")
def processor():
    ctx = build_model_context(MODEL_ID, limit_mm_per_prompt={"image": 2})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def test_tokenizer_adds_trajectory_and_special_tokens(processor) -> None:
    tokenizer = processor.info.get_tokenizer()
    config = processor.info.get_hf_config()
    traj_ids = config.traj_ids

    assert tokenizer.convert_tokens_to_ids("<i0>") == traj_ids["history_id0"]
    future_id0 = f"<i{config.history_vocab_size}>"
    assert tokenizer.convert_tokens_to_ids(future_id0) == traj_ids["future_id0"]
    # Trajectory tokens written in a prompt tokenize to single ids.
    ids = tokenizer.encode(
        "<|traj_history_start|><i12><i999><|traj_history_end|>",
        add_special_tokens=False,
    )
    assert ids == [
        traj_ids["history_start"],
        traj_ids["history_id0"] + 12,
        traj_ids["history_id0"] + 999,
        traj_ids["history_end"],
    ]
    assert tokenizer.decode([traj_ids["future_start"]]) == "<|traj_future_start|>"


def test_images_use_the_config_pixel_budget(processor) -> None:
    # A 1920x1080 camera frame is resized to 576x320 (163,840 to 196,608
    # pixels), i.e. 20x36 patches and 180 image tokens.
    processed = processor(
        IMAGE_PLACEHOLDER,
        mm_items=processor.info.parse_mm_data(
            {"image": [Image.new("RGB", (1920, 1080))]}
        ),
        hf_processor_mm_kwargs={},
    )
    grid_thw = processed["mm_kwargs"].get_data()["image_grid_thw"]
    assert grid_thw.tolist() == [[1, 20, 36]]
    assert processed["mm_placeholders"]["image"][0].length == 180


def test_video_input_is_rejected(processor) -> None:
    assert set(processor.info.supported_mm_limits) == {"image"}
    with pytest.raises(VLLMValidationError, match="At most 0 video"):
        processor.info.validate_num_items("video", 1)


def test_weights_mapper_unwraps_vlm_and_drops_expert() -> None:
    mapper = Alpamayo2SuperForConditionalGeneration.hf_to_vllm_mapper
    names = [
        "vlm.model.language_model.layers.0.mlp.up_proj.weight",
        "vlm.model.visual.blocks.0.attn.qkv.weight",
        "vlm.lm_head.weight",
        "expert.expert.layers.0.mlp.up_proj.weight",
        "expert.action_out_proj.weight",
    ]
    assert mapper.apply_list(names) == [
        "language_model.model.layers.0.mlp.up_proj.weight",
        "visual.blocks.0.attn.qkv.weight",
        "language_model.lm_head.weight",
    ]
