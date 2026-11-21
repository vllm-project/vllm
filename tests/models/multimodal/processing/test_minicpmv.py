# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for MiniCPMV's multimodal preprocessing."""

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.model_executor.models.minicpmv import MiniCPMVProcessingInfo
from vllm.model_executor.models.minicpmv4_6 import MiniCPMV4_6ProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["openbmb/MiniCPM-V-4"])
def test_get_hf_processor_for_same_model_different_kwargs(model_id: str):
    """Calls with different kwargs must not reuse stale processor instances."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, MiniCPMVProcessingInfo)
    processor_1 = info.get_hf_processor(max_slice_nums=1)
    processor_2 = info.get_hf_processor(max_slice_nums=2)
    assert processor_1.image_processor.max_slice_nums == 1
    assert processor_2.image_processor.max_slice_nums == 2


@pytest.mark.parametrize(
    "model_ids",
    [
        ("openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-V-4"),
        ("openbmb/MiniCPM-Llama3-V-2_5", "openbmb/MiniCPM-o-2_6"),
    ],
)
def test_image_processor_for_different_models(model_ids):
    first_model_id, second_model_id = model_ids

    first_ctx = build_model_context(
        first_model_id,
        limit_mm_per_prompt={"image": 1},
    )
    first_processor = MULTIMODAL_REGISTRY.create_processor(first_ctx.model_config)
    assert isinstance(first_processor.info, MiniCPMVProcessingInfo)
    first_image_processor = first_processor.info.get_image_processor()

    second_ctx = build_model_context(
        second_model_id,
        limit_mm_per_prompt={"image": 1},
    )
    second_processor = MULTIMODAL_REGISTRY.create_processor(second_ctx.model_config)
    assert isinstance(second_processor.info, MiniCPMVProcessingInfo)
    second_image_processor = second_processor.info.get_image_processor()

    second_processor.info.get_sliced_grid(ImageSize(width=128, height=128))

    assert type(first_image_processor) is not type(second_image_processor)
    assert (
        type(first_image_processor).__module__
        != type(second_image_processor).__module__
    )


@pytest.mark.parametrize("model_id", ["openbmb/MiniCPM-V-4"])
def test_prompt_has_dif_BPE_boundaries_in_context(model_id: str):
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = ctx.get_tokenizer()

    messages = [
        {"role": "user", "content": "(<image>./</image>)\nWhat is in this image?"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image = np.zeros((768, 1024, 3), dtype=np.uint8)

    mm_items = processor.info.parse_mm_data({"image": [image]})
    processed = processor(
        prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )
    image_placeholders = processed["mm_placeholders"].get("image", [])
    assert len(image_placeholders) == 1
    assert image_placeholders[0].length > 0


@pytest.mark.parametrize(
    "model_id",
    [
        "openbmb/MiniCPM-Llama3-V-2_5",
        "openbmb/MiniCPM-V-4",
        "openbmb/MiniCPM-o-2_6",
    ],
)
@pytest.mark.parametrize("max_slice_nums", [1, 2])
def test_request_slice_limit_matches_image_features(model_id, max_slice_nums):
    """Request overrides must reserve one query block per processed image crop."""
    kwargs = {"max_slice_nums": max_slice_nums}
    image = np.zeros((896, 896, 3), dtype=np.uint8)
    results = []
    for kwargs_on_init in (True, False):
        ctx = build_model_context(
            model_id,
            mm_processor_kwargs=kwargs if kwargs_on_init else None,
            limit_mm_per_prompt={"image": 1},
        )
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
        result = processor(
            "(<image>./</image>)\nDescribe the image.",
            mm_items=processor.info.parse_mm_data({"image": image}),
            hf_processor_mm_kwargs={} if kwargs_on_init else kwargs,
        )
        tgt_sizes = result["mm_kwargs"].get_data()["tgt_sizes"]
        assert isinstance(tgt_sizes, torch.Tensor)
        assert isinstance(processor.info, MiniCPMVProcessingInfo)
        num_crops = tgt_sizes.numel() // 2
        num_queries = processor.info.get_image_processor().image_feature_size
        placeholder = result["mm_placeholders"]["image"][0]
        assert placeholder.get_num_embeds() == num_crops * num_queries
        results.append(result)

    assert results[0]["prompt_token_ids"] == results[1]["prompt_token_ids"]


def _video_item(num_frames: int, width: int, height: int) -> tuple:
    return (
        np.zeros((num_frames, height, width, 3), dtype=np.uint8),
        {
            "fps": 2.0,
            "duration": num_frames / 2.0,
            "total_num_frames": num_frames,
            "frames_indices": list(range(num_frames)),
            "video_backend": "opencv",
            "do_sample_frames": False,
        },
    )


def _num_video_embeds(processor, video: tuple) -> int:
    processed = processor(
        processor.info.video_pattern,
        mm_items=processor.info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={},
    )
    return sum(p.get_num_embeds() for p in processed["mm_placeholders"]["video"])


@pytest.mark.parametrize(
    ("mm_processor_kwargs", "image_size", "num_tokens"),
    [
        (None, (1265, 1112), 720),
        ({"max_slice_nums": 4}, (2901, 139), 362),
    ],
)
def test_minicpmv4_6_image_budget_covers_largest_image(
    mm_processor_kwargs: dict[str, object] | None,
    image_size: tuple[int, int],
    num_tokens: int,
):
    ctx = build_model_context(
        "openbmb/MiniCPM-V-4_6",
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, MiniCPMV4_6ProcessingInfo)
    max_tokens = info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len, mm_counts={"image": 1}
    )
    assert max_tokens is not None
    budget = max_tokens["image"]
    dummy = processor.get_dummy_mm_inputs({"image": 1})
    processed = processor(
        info.image_pattern,
        mm_items=info.parse_mm_data({"image": [Image.new("RGB", image_size)]}),
        hf_processor_mm_kwargs={},
    )

    assert budget == num_tokens
    assert dummy["mm_placeholders"]["image"][0].get_num_embeds() == budget
    assert processed["mm_placeholders"]["image"][0].get_num_embeds() == budget


def test_minicpmv4_6_video_budget_covers_largest_frames():
    ctx = build_model_context(
        "openbmb/MiniCPM-V-4_6",
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, MiniCPMV4_6ProcessingInfo)
    seq_len = ctx.model_config.max_model_len
    max_tokens = info.get_mm_max_tokens_per_item(
        seq_len=seq_len, mm_counts={"video": 1}
    )
    assert max_tokens is not None
    budget = max_tokens["video"]
    num_frames = info.get_num_frames_with_most_features(seq_len, 1)
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    assert budget == num_frames * 720
    assert sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["video"]) == budget
    assert _num_video_embeds(processor, _video_item(num_frames, 1265, 1112)) == budget


@pytest.mark.parametrize(
    ("modality", "mm_processor_kwargs"),
    [
        ("image", {"images_kwargs": {"downsample_mode": "4x"}}),
        ("video", {"videos_kwargs": {"downsample_mode": "4x"}}),
        ("video", {"videos_kwargs": {"max_slice_nums": 1}}),
    ],
)
def test_minicpmv4_6_budget_follows_configured_kwargs(
    modality: str,
    mm_processor_kwargs: dict[str, object],
):
    ctx = build_model_context(
        "openbmb/MiniCPM-V-4_6",
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, MiniCPMV4_6ProcessingInfo)
    seq_len = ctx.model_config.max_model_len
    max_tokens = info.get_mm_max_tokens_per_item(
        seq_len=seq_len, mm_counts={modality: 1}
    )
    assert max_tokens is not None
    budget = max_tokens[modality]
    dummy = processor.get_dummy_mm_inputs({modality: 1})
    if modality == "image":
        processed = processor(
            info.image_pattern,
            mm_items=info.parse_mm_data({"image": [Image.new("RGB", (1265, 1112))]}),
            hf_processor_mm_kwargs={},
        )
        num_embeds = processed["mm_placeholders"]["image"][0].get_num_embeds()
    else:
        num_frames = info.get_num_frames_with_most_features(seq_len, 1)
        num_embeds = _num_video_embeds(processor, _video_item(num_frames, 1265, 1112))

    dummy_embeds = sum(p.get_num_embeds() for p in dummy["mm_placeholders"][modality])

    assert dummy_embeds == budget
    assert budget >= num_embeds


@pytest.mark.parametrize("kwargs_on_init", [True, False])
@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [{"max_slice_nums": 1}, {"videos_kwargs": {"max_slice_nums": 1}}],
)
def test_minicpmv4_6_video_slice_limit_matches_hf(
    mm_processor_kwargs: dict[str, object],
    kwargs_on_init: bool,
):
    ctx = build_model_context(
        "openbmb/MiniCPM-V-4_6",
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, MiniCPMV4_6ProcessingInfo)
    video = _video_item(2, 1265, 1112)
    processed = processor(
        info.video_pattern,
        mm_items=info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={} if kwargs_on_init else mm_processor_kwargs,
    )
    video_token = "<|video_pad|>"
    reference = info.get_hf_processor()(
        text=[video_token],
        videos=[video[0]],
        videos_kwargs={"do_sample_frames": False, "max_slice_nums": 1},
        return_tensors="pt",
    )
    video_token_id = info.get_tokenizer().convert_tokens_to_ids(video_token)
    num_embeds = sum(p.get_num_embeds() for p in processed["mm_placeholders"]["video"])
    tgt_sizes = processed["mm_kwargs"].get_data()["video_tgt_sizes"]
    assert isinstance(tgt_sizes, torch.Tensor)

    assert num_embeds == int((reference["input_ids"] == video_token_id).sum())
    assert tgt_sizes.shape[1] == reference["target_sizes_videos"].shape[0]


@pytest.mark.parametrize(
    ("modality", "hint", "input_size"),
    [
        ("image", {"width": 1000, "height": 2000}, (1000, 2000)),
        ("image", {"width": 500}, (500, 4000)),
        ("image", {"height": 500}, (1120, 500)),
        ("video", {"width": 1000, "height": 2000}, (1000, 2000)),
        ("video", {"width": 500}, (500, 4000)),
    ],
)
def test_minicpmv4_6_dummy_follows_size_hint(
    modality: str,
    hint: dict[str, int],
    input_size: tuple[int, int],
):
    num_frames = 2
    options: dict[str, int] = {"count": 1, **hint}
    if modality == "video":
        options["num_frames"] = num_frames
    ctx = build_model_context(
        "openbmb/MiniCPM-V-4_6",
        limit_mm_per_prompt={"image": 0, "video": 0} | {modality: options},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, MiniCPMV4_6ProcessingInfo)
    if modality == "image":
        item: object = Image.new("RGB", input_size)
        prompt = info.image_pattern
    else:
        item = _video_item(num_frames, *input_size)
        prompt = info.video_pattern
    dummy = processor.get_dummy_mm_inputs({modality: 1})
    processed = processor(
        prompt,
        mm_items=info.parse_mm_data({modality: [item]}),
        hf_processor_mm_kwargs={},
    )

    assert sum(p.get_num_embeds() for p in dummy["mm_placeholders"][modality]) >= sum(
        p.get_num_embeds() for p in processed["mm_placeholders"][modality]
    )


@pytest.mark.parametrize(
    ("max_model_len", "previous_budget"),
    [(6400, 5832), (8192, 7776), (10000, 9720), (11520, 10368)],
)
def test_minicpmv4_6_video_budget_fills_short_context(
    max_model_len: int, previous_budget: int
):
    ctx = build_model_context(
        "openbmb/MiniCPM-V-4_6",
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, MiniCPMV4_6ProcessingInfo)
    budget = info.get_max_video_tokens(max_model_len, {"video": 1})
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    assert previous_budget <= budget <= max_model_len
    assert sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["video"]) == budget


def test_minicpmv4_6_video_dummy_follows_num_frames_hint():
    num_frames = 4
    ctx = build_model_context(
        "openbmb/MiniCPM-V-4_6",
        limit_mm_per_prompt={
            "image": 0,
            "video": {"count": 1, "num_frames": num_frames},
        },
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, MiniCPMV4_6ProcessingInfo)
    dummy = processor.get_dummy_mm_inputs({"video": 1})
    dummy_embeds = sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["video"])
    budget = info.get_max_video_tokens(ctx.model_config.max_model_len, {"video": 1})

    assert dummy_embeds >= _num_video_embeds(
        processor, _video_item(num_frames, 1265, 1112)
    )
    assert budget >= dummy_embeds
