# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tests.transformers_utils.processors.test_hy_v4 import (
    hy_v4_processor_dir,  # noqa: F401
)
from vllm.models.hy_v4.nvidia.mm_preprocess import (
    HYV4VLDataParser,
    HYV4VLDummyInputsBuilder,
    HYV4VLMultiModalProcessor,
    HYV4VLProcessingInfo,
)
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.transformers_utils.processors.hy_v4 import HYV4VLProcessor, vision_token_count


def test_hy_v4_fields_split_by_media_item(request):
    hf_processor = HYV4VLProcessor.from_pretrained(
        request.getfixturevalue("hy_v4_processor_dir")
    )
    context = SimpleNamespace(
        tokenizer=hf_processor.tokenizer,
        get_hf_config=lambda _: hf_processor.config,
        get_hf_processor=lambda *args, **kwargs: hf_processor,
        get_merged_mm_kwargs=lambda *args, **kwargs: {},
    )
    info = HYV4VLProcessingInfo(context)
    processor = HYV4VLMultiModalProcessor(info, HYV4VLDummyInputsBuilder(info))
    images = [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 128))]
    videos = [
        (
            np.zeros((t, 64, 64, 3), dtype=np.uint8),
            {
                "fps": 2,
                "total_num_frames": t,
                "frames_indices": list(range(t)),
                "do_sample_frames": False,
            },
        )
        for t in (2, 3)
    ]
    data = hf_processor(images=images, videos=videos, return_tensors="pt")
    items = MultiModalKwargsItems.from_hf_inputs(
        data, processor._get_mm_fields_config(data, {})
    )
    assert len(items["image"]) == len(items["video"]) == 2
    for modality in ("image", "video"):
        pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
        for item in items[modality]:
            grid = item[f"{modality}_grid_thw"].data
            assert item[pixel_key].data.shape[0] == grid.prod().item()
            times = item["video_timestamps"].data if modality == "video" else None
            replacement = hf_processor.get_replacement(modality, grid, times)
            assert replacement.count(
                getattr(hf_processor, f"{modality}_token_id")
            ) == vision_token_count(grid, hf_processor.config.vision_config, modality)
    limits = info.get_mm_max_tokens_per_item(256, {"image": 2, "video": 2})
    for modality in items:
        assert all(
            vision_token_count(
                item[f"{modality}_grid_thw"].data,
                hf_processor.config.vision_config,
                modality,
            )
            <= limits[modality]
            for item in items[modality]
        )


@pytest.mark.parametrize(
    "mm_kwargs,image_pixels,video_pixels,max_frames",
    [
        ({}, 16384, 16384, 8),
        ({"max_pixels": 8192, "max_frames": 2}, 8192, 8192, 2),
        (
            {
                "images_kwargs": {"max_pixels": 4096},
                "videos_kwargs": {"max_pixels": 8192, "max_frames": 2},
            },
            4096,
            8192,
            2,
        ),
    ],
)
def test_hy_v4_profiling_uses_context_mm_kwargs(
    request, monkeypatch, mm_kwargs, image_pixels, video_pixels, max_frames
):
    from vllm.config import ModelConfig
    from vllm.multimodal.processing.context import InputProcessingContext

    config = ModelConfig(
        str(request.getfixturevalue("hy_v4_processor_dir")),
        trust_remote_code=False,
        max_model_len=256,
        mm_processor_kwargs=mm_kwargs,
    )
    info = HYV4VLProcessingInfo(InputProcessingContext(config, tokenizer=None))
    hf_processor = info.get_hf_processor()
    for component in (hf_processor.image_processor, hf_processor.video_processor):
        monkeypatch.setattr(
            component,
            "size",
            {key: component.size[key] for key in ("shortest_edge", "longest_edge")},
        )

    size = info.get_image_size_with_most_features()
    assert size.height * size.width == image_pixels
    limits = info.get_mm_max_tokens_per_item(256, {"image": 1, "video": 1})
    vision_config = config.hf_config.vision_config
    patch = vision_config.patch_size
    factor = patch * vision_config.spatial_merge_size
    assert limits["image"] == vision_token_count(
        (1, size.height // patch, size.width // patch), vision_config, "image"
    )
    assert limits["video"] == 2 * (video_pixels // factor**2 + max_frames)
    frames, height, width = info.get_dummy_video_shape(256, {"video": 1})
    assert 1 <= frames <= max_frames
    assert frames * height * width <= video_pixels


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("batched", [False, True])
def test_hy_v4_rejects_precomputed_embeddings(modality, batched):
    parser = HYV4VLDataParser()
    embeddings = torch.zeros(1, 6, 32) if batched else [torch.zeros(6, 32)]
    with pytest.raises(ValueError, match="precomputed"):
        parser.parse_mm_data({modality: embeddings})


@pytest.mark.usefixtures("hy_v4_processor_dir")
@pytest.mark.parametrize(
    "hy_v4_processor_dir", ["named", "custom", "unnamed"], indirect=True
)
def test_hy_v4_registered_processing_cache_and_mixed_inputs(request):
    from vllm.config import ModelConfig
    from vllm.models.hy_v4.nvidia.vl_model import HYV4VLForConditionalGeneration
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    path = str(request.getfixturevalue("hy_v4_processor_dir"))
    config = ModelConfig(path, trust_remote_code=False, max_model_len=256)
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    tokenizer = processor.info.get_tokenizer()
    hf_processor = processor.info.get_hf_processor()
    prompt = processor.dummy_inputs.get_dummy_text({"image": 2, "video": 1})
    raw_image = hf_processor.get_placeholder("image")
    raw_video = hf_processor.get_placeholder("video")
    assert prompt == raw_image * 2 + raw_video
    media = {
        "image": [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 96))],
        "video": [
            (
                np.zeros((2, 64, 64, 3), dtype=np.uint8),
                {
                    "fps": 2,
                    "total_num_frames": 2,
                    "frames_indices": [0, 1],
                    "do_sample_frames": False,
                },
            )
        ],
    }
    items = processor.info.parse_mm_data(media)
    inputs = ProcessorInputs(tokenizer.encode(prompt, add_special_tokens=False), items)
    uncached = processor.apply(inputs, TimingContext(enabled=False))
    inputs.cache = MultiModalProcessorOnlyCache(config)
    first = processor.apply(inputs, TimingContext(enabled=False))
    second = processor.apply(inputs, TimingContext(enabled=False))
    for actual in (first, second):
        assert actual["prompt_token_ids"] == uncached["prompt_token_ids"]
        assert actual["mm_hashes"] == uncached["mm_hashes"]
        for modality, ranges in actual["mm_placeholders"].items():
            assert [r.get_num_embeds() for r in ranges] == [
                r.get_num_embeds() for r in uncached["mm_placeholders"][modality]
            ]
    assert len(first["mm_placeholders"]["image"]) == 2
    assert len(first["mm_placeholders"]["video"]) == 1

    image_marker = HYV4VLForConditionalGeneration.get_placeholder_str("image", 0)
    video_marker = HYV4VLForConditionalGeneration.get_placeholder_str("video", 0)
    for alternate_prompt in (
        image_marker * 2 + video_marker,
        image_marker + raw_image + raw_video,
        raw_image + image_marker + video_marker,
    ):
        alternate = processor.apply(
            ProcessorInputs(
                tokenizer.encode(alternate_prompt, add_special_tokens=False),
                items,
                cache=inputs.cache,
            ),
            TimingContext(enabled=False),
        )
        assert alternate["prompt_token_ids"] == uncached["prompt_token_ids"]
        assert alternate["mm_hashes"] == uncached["mm_hashes"]


def test_hy_v4_partial_video_cache_preserves_per_item_timestamps(request):
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    path = str(request.getfixturevalue("hy_v4_processor_dir"))
    config = ModelConfig(path, trust_remote_code=False, max_model_len=256)
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    cache = MultiModalProcessorOnlyCache(config)
    tokens = processor.info.get_tokenizer().encode(
        processor.dummy_inputs.get_dummy_text({"video": 2}),
        add_special_tokens=False,
    )

    def inputs(values, use_cache):
        media = [
            (
                np.full((2, 64, 64, 3), value, dtype=np.uint8),
                {
                    "fps": 2,
                    "total_num_frames": 2,
                    "frames_indices": [0, 1],
                    "do_sample_frames": False,
                },
            )
            for value in values
        ]
        return ProcessorInputs(
            tokens,
            processor.info.parse_mm_data({"video": media}),
            hf_processor_mm_kwargs={"second_per_grid_ts": [1.0, 2.0]},
            cache=cache if use_cache else None,
        )

    processor.apply(inputs([0, 1], True), TimingContext(enabled=False))
    partial = processor.apply(inputs([0, 2], True), TimingContext(enabled=False))
    expected = processor.apply(inputs([0, 2], False), TimingContext(enabled=False))
    assert partial["prompt_token_ids"] == expected["prompt_token_ids"]
    assert partial["mm_hashes"] == expected["mm_hashes"]
    video_kwargs = partial["mm_kwargs"]["video"][1]
    assert video_kwargs is not None
    timestamps = video_kwargs["video_timestamps"].data
    assert isinstance(timestamps, torch.Tensor)
    assert timestamps.tolist() == [0.0, 2.0]
