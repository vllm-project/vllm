# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video placeholder accounting for GLM-5.3-Flash.

``Glm4vProcessingInfo._construct_video_placeholder`` emits one frame of
placeholders per timestamp, so the timestamps returned by
``_get_video_second_idx_glm46v`` decide how many placeholders the prompt gets
while ``video_grid_thw`` decides how many rows the vision tower produces. If
the two disagree they collide in ``_merge_multimodal_embeddings``, which raises
inside a worker and takes the engine down with it.

The grid checks below are arithmetic: no weights, no GPU.
``test_glm5_metadata_only_roundtrip`` loads the processor and compares
raw media with metadata-only prompts.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.glm5_next.video_processing_glm5_next import (
    Glm5NextVideoProcessor,
    smart_resize,
)
from transformers.video_utils import VideoMetadata

from vllm.distributed.ec_transfer.ec_connector.utils import (
    PlaceholderMetadataResolver,
    collect_ec_item_metadata,
)
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.platforms import current_platform
from vllm.renderers.chat_utils.item_tracker import _get_embeds_data

from ....multimodal.utils import random_image, random_video
from ...utils import build_model_context


def _video_with_metadata(frames: np.ndarray):
    num_frames = len(frames)
    return (
        frames,
        {
            "total_num_frames": num_frames,
            "fps": 2.0,
            "duration": num_frames / 2.0,
            "video_backend": "opencv",
            "frames_indices": list(range(num_frames)),
            "do_sample_frames": True,
        },
    )


@pytest.mark.parametrize("modality", ["image", "video"])
def test_glm5_metadata_only_roundtrip(modality):
    """EC metadata reproduces the raw prompt for image and video."""
    rng = np.random.RandomState(0)
    if modality == "image":
        mm_data = {"image": random_image(rng, 64, 65)}
        limit = {"image": 1, "video": 0}
        grid_key = "image_grid_thw"
    else:
        mm_data = {"video": _video_with_metadata(random_video(rng, 4, 5, 64, 65))}
        limit = {"image": 0, "video": 1}
        grid_key = "video_grid_thw"

    ctx = build_model_context(
        "zai-org/GLM-5.3-Flash",
        limit_mm_per_prompt=limit,
    )
    mm_config = ctx.model_config.multimodal_config
    mm_config.enable_mm_embeds = True
    mm_config.allow_missing_mm_embeddings = True
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
    )
    prompt = processor.dummy_inputs.get_dummy_text({modality: 1})
    raw = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )
    features = [
        MultiModalFeatureSpec(
            data=item,
            modality=modality,
            identifier=str(index),
            mm_position=position,
        )
        for index, (item, position) in enumerate(
            zip(raw["mm_kwargs"][modality], raw["mm_placeholders"][modality])
        )
    ]
    published = collect_ec_item_metadata(
        features, PlaceholderMetadataResolver(ctx.model_config)
    )
    metadata = json.loads(
        json.dumps([entry["metadata"] for entry in published.values()])
    )
    assert len(metadata) == 1
    assert grid_key in metadata[0]
    if modality == "video":
        assert metadata[0]["timestamps"]
    mm_data_remote = _get_embeds_data(modality, metadata, processor)
    remote = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({modality: mm_data_remote}),
        hf_processor_mm_kwargs={},
    )
    assert remote["prompt_token_ids"] == raw["prompt_token_ids"]
    assert remote["mm_placeholders"] == raw["mm_placeholders"]

    mm_config.allow_missing_mm_embeddings = False
    ordinary_processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
    )
    with pytest.raises(ValueError, match="should contain the fields"):
        ordinary_processor.info.parse_mm_data({modality: mm_data_remote})


@pytest.fixture(scope="module")
def processor():
    ctx = build_model_context(
        "zai-org/GLM-5.3-Flash",
        limit_mm_per_prompt={"video": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
    )


def _pixel_path_grid(
    video_processor: Glm5NextVideoProcessor,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[int, int, int]:
    """The ``video_grid_thw`` ``Glm5NextVideoProcessor._preprocess`` builds."""
    resized_height, resized_width = smart_resize(
        num_frames=num_frames,
        height=height,
        width=width,
        temporal_factor=video_processor.temporal_patch_size,
        factor=video_processor.patch_size
        * video_processor.merge_size
        * video_processor.patch_expand_factor,
        min_pixels=video_processor.min_image_tokens,
        max_pixels=video_processor.max_image_tokens,
    )
    padded_frames = num_frames + (-num_frames % video_processor.temporal_patch_size)
    return (
        padded_frames // video_processor.temporal_patch_size,
        resized_height // video_processor.patch_size,
        resized_width // video_processor.patch_size,
    )


@pytest.mark.parametrize(
    ("total_num_frames", "fps", "duration", "height", "width", "expected_grid"),
    [
        # 4 s at 8 fps: the GLM-4.6V sampler asks for 3x as many timestamps.
        (32, 8.0, 4.0, 480, 640, (4, 36, 46)),
        # 1080p, 20 s: same factor of 3 at a full-size canvas.
        (600, 30.0, 20.0, 1080, 1920, (20, 58, 102)),
        # 1080p, 60 s: the one duration window where the two samplers agree
        # anyway -- a regression guard, the count must not move.
        (1800, 30.0, 60.0, 1080, 1920, (60, 34, 58)),
        # Past 300 s the GLM-4.6V sampler asks for half as many instead.
        (9030, 30.0, 301.0, 720, 1280, (301, 14, 26)),
    ],
)
def test_video_placeholders_match_encoder_rows(
    processor,
    total_num_frames: int,
    fps: float,
    duration: float,
    height: int,
    width: int,
    expected_grid: tuple[int, int, int],
):
    info = processor.info
    video_processor = info.get_video_processor()

    frame_indices = video_processor.sample_frames(
        VideoMetadata(total_num_frames=total_num_frames, fps=fps, duration=duration)
    )
    grid_t, grid_h, grid_w = _pixel_path_grid(
        video_processor, len(frame_indices), height, width
    )
    assert (grid_t, grid_h, grid_w) == expected_grid

    timestamps = info._get_video_second_idx_glm46v(
        {
            "total_num_frames": total_num_frames,
            "fps": fps,
            "duration": duration,
            "do_sample_frames": True,
        },
        total_num_frames,
    )

    merge_length = video_processor.merge_size**2
    tokens_per_frame = grid_h * grid_w // merge_length
    encoder_rows = grid_t * grid_h * grid_w // merge_length

    assert len(timestamps) == grid_t
    assert len(timestamps) * tokens_per_frame == encoder_rows
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0
    assert timestamps[-1] <= duration


def test_video_placeholders_match_encoder_rows_when_presampled(processor):
    """The loader may pre-sample and hand the frames over as they are."""
    info = processor.info
    video_processor = info.get_video_processor()

    num_frames = 32
    grid_t, _, _ = _pixel_path_grid(video_processor, num_frames, 480, 640)

    timestamps = info._get_video_second_idx_glm46v(
        {
            "total_num_frames": 256,
            "fps": 8.0,
            "duration": 32.0,
            "do_sample_frames": False,
            "frames_indices": list(range(0, 256, 256 // num_frames)),
        },
        num_frames,
    )

    assert len(timestamps) == grid_t


def test_placeholder_rejects_timestamp_count_mismatch(processor):
    """A timestamp list that is not one entry per temporal patch is refused."""
    grid_thw = torch.tensor([4, 36, 46])
    with pytest.raises(ValueError, match="video_grid_thw"):
        processor.info._placeholder_from_timestamps([0, 1], grid_thw)


def test_video_timestamps_follow_supplied_frame_indices(processor):
    """Indices already chosen by the processor win over a fresh resample."""
    info = processor.info
    video_processor = info.get_video_processor()
    temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 1) or 1)
    fps = 8.0
    indices = [0, 8, 80, 88]
    metadata = {
        "total_num_frames": 256,
        "fps": fps,
        "duration": 32.0,
        "do_sample_frames": True,
        "frames_indices": indices,
    }

    timestamps = info._get_video_second_idx_glm46v(metadata, 256)
    resampled = info._get_video_second_idx_glm46v(
        {k: v for k, v in metadata.items() if k != "frames_indices"},
        256,
    )

    num_frames = len(indices)
    num_padded = num_frames + (-num_frames % temporal_patch_size)
    grid_t = max(num_padded // temporal_patch_size, 1)
    last = len(indices) - 1
    expected = [
        int(indices[min(i * temporal_patch_size, last)] / fps) for i in range(grid_t)
    ]
    assert timestamps == expected
    assert timestamps != resampled


def test_video_shorter_than_one_sampling_interval_is_rejected(processor):
    """A clip the sampler cannot pick a single frame from is a bad request."""
    with pytest.raises(ValueError, match="selected no frames"):
        processor.info._get_video_second_idx_glm46v(
            {
                "total_num_frames": 2,
                "fps": 8.0,
                "duration": 0.25,
                "do_sample_frames": True,
            },
            2,
        )


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    device = current_platform.device_type
    ctx = build_model_context(
        "zai-org/GLM-5.3-Flash",
        limit_mm_per_prompt={"image": 2},
    )
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [
        Image.new("RGB", (310, 470), color=(17, 89, 231)),
        Image.new("RGB", (480, 320), color=(201, 13, 127)),
    ]
    prompt = "<|begin_of_image|><|image|><|end_of_image|>" * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized_values = normalized_inputs["mm_kwargs"].get_data()["pixel_values"]

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw_values = raw_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert raw_values.dtype == torch.uint8

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.to(device), rtol=1e-5, atol=1e-6
    )


def _image_info(**kwargs):
    ctx = build_model_context(
        "zai-org/GLM-5.3-Flash",
        limit_mm_per_prompt={"image": 1},
        **kwargs,
    )
    return MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
    ).info


def test_image_encoder_cache_covers_full_token_budget():
    """The most-features probe must reach the processor's token ceiling.

    The inherited square probe refits to 89x89 = 7921 tokens under the
    max_image_tokens=8000 budget, so the encoder cache came up short and
    ordinary non-square images in 7922-8000 tokens were refused with
    HTTP 400 (#59539).
    """
    info = _image_info()
    assert info.get_max_image_tokens() == 8000

    # The shapes from the issue, with the token counts the processor
    # actually produces; every one must fit the cache.
    for width, height, expected_tokens in [
        (4032, 3024, 7931),  # phone photo, 4:3
        (3840, 2160, 7973),  # 4K frame, 16:9
        (3508, 2480, 7950),  # A4 at 300 dpi
        (2600, 2400, 7998),  # 13:12
        (3000, 3000, 7921),  # square worst case before the fix
    ]:
        num_tokens = info.get_num_image_tokens(image_width=width, image_height=height)
        assert num_tokens == expected_tokens
        assert num_tokens <= info.get_max_image_tokens()

    # The profiling dummy covers the real worst case, not the square one.
    size = info.get_image_size_with_most_features()
    assert size.width * size.height == 2240 * 2800


def test_image_encoder_cache_follows_max_pixels_override():
    info = _image_info(mm_processor_kwargs={"max_pixels": 1568 * 100})
    assert info.get_max_image_tokens() == 100
    size = info.get_image_size_with_most_features()
    assert (
        info.get_num_image_tokens(image_width=size.width, image_height=size.height)
        == 100
    )
