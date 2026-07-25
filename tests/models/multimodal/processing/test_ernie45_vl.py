# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

from vllm.model_executor.models.ernie45_vl import Ernie4_5VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

from ...utils import build_model_context

MODEL_ID = "baidu/ERNIE-4.5-VL-28B-A3B-PT"


def _make_video(num_frames: int) -> np.ndarray:
    frame, row, column = np.indices((num_frames, 56, 56))
    return np.stack(
        (
            (17 * frame + 3 * column) % 256,
            (29 * frame + 5 * row) % 256,
            (11 * frame + row + column) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _metadata(
    frame_indices: list[int],
    total_num_frames: int,
    do_sample_frames: bool | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "total_num_frames": total_num_frames,
        "fps": 7.5,
        "duration": total_num_frames / 7.5,
        "video_backend": "opencv",
        "frames_indices": frame_indices,
    }
    if do_sample_frames is not None:
        metadata["do_sample_frames"] = do_sample_frames
    return metadata


def _clone_video(video: np.ndarray | torch.Tensor):
    return video.copy() if isinstance(video, np.ndarray) else video.clone()


def _video_hash(prompt: str, mm_items) -> str:
    inputs = ProcessorInputs(prompt=prompt, mm_data_items=mm_items)
    return inputs.get_mm_hashes(MODEL_ID)["video"][0]


@pytest.fixture(scope="module")
def ernie_context():
    return build_model_context(
        MODEL_ID,
        limit_mm_per_prompt={"image": 2, "video": 2},
        mm_processor_cache_gb=1,
    )


@pytest.mark.parametrize(
    ("modality", "legacy_marker"),
    [
        ("image", "<|image@placeholder|>"),
        ("video", "<|video@placeholder|>"),
    ],
)
def test_native_and_legacy_markers_on_cache_miss_and_hit(
    ernie_context,
    modality: str,
    legacy_marker: str,
):
    cache = MultiModalProcessorOnlyCache(ernie_context.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ernie_context.model_config,
        tokenizer=ernie_context.tokenizer,
        cache=cache,
    )
    hf_processor = processor.info.get_hf_processor()
    native_marker = getattr(hf_processor, f"{modality}_token")
    native_prompt = f"{native_marker} Describe this input."
    legacy_prompt = native_prompt.replace(native_marker, legacy_marker)

    if modality == "image":
        mm_data = {"image": Image.new("RGB", (56, 56), "white")}
    else:
        video = _make_video(2)
        mm_data = {"video": (video, _metadata([0, 1], 2, False))}
    mm_items = processor.info.parse_mm_data(mm_data)

    results = [
        processor(legacy_prompt, mm_items=mm_items),
        processor(native_prompt, mm_items=mm_items),
        processor(native_prompt, mm_items=mm_items),
    ]
    expected = results[0]
    for result in results[1:]:
        assert result["prompt_token_ids"] == expected["prompt_token_ids"]
        assert result["mm_placeholders"] == expected["mm_placeholders"]

    (placeholder,) = expected["mm_placeholders"][modality]
    placeholder_ids = expected["prompt_token_ids"][
        placeholder.offset : placeholder.offset + placeholder.length
    ]
    token_id = getattr(processor.info.get_hf_config(), f"{modality}_token_id")
    assert placeholder_ids == [token_id] * placeholder.length
    assert len(cache._cache) == 1


@pytest.mark.parametrize("prompt_type", ["text", "tokens"])
def test_mixed_native_and_legacy_image_markers_preserve_item_order(
    ernie_context,
    prompt_type: str,
):
    cache = MultiModalProcessorOnlyCache(ernie_context.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ernie_context.model_config,
        tokenizer=ernie_context.tokenizer,
        cache=cache,
    )
    native_marker = processor.info.get_hf_processor().image_token
    legacy_marker = "<|image@placeholder|>"
    mm_items = processor.info.parse_mm_data(
        {
            "image": [
                Image.new("RGB", (56, 56), "white"),
                Image.new("RGB", (112, 56), "black"),
            ]
        }
    )
    prompt = f"{native_marker} A {legacy_marker}"

    def run_processor():
        if prompt_type == "text":
            return processor(prompt, mm_items=mm_items)
        prompt_tokens = ernie_context.tokenizer.encode(prompt, add_special_tokens=False)
        return processor.apply(
            ProcessorInputs(prompt_tokens, mm_items),
            TimingContext(enabled=False),
        )

    cache_miss = run_processor()
    cache_hit = run_processor()
    assert cache_hit["prompt_token_ids"] == cache_miss["prompt_token_ids"]
    assert cache_hit["mm_placeholders"] == cache_miss["mm_placeholders"]
    assert len(cache._cache) == 2

    placeholders = cache_miss["mm_placeholders"]["image"]
    image_grids = cache_miss["mm_kwargs"].get_data()["image_grid_thw"]
    merge_length = processor.info.get_hf_config().spatial_conv_size ** 2
    assert len(placeholders) == 2
    assert placeholders[0].offset < placeholders[1].offset
    assert [placeholder.length for placeholder in placeholders] == [
        int(grid.prod()) // merge_length for grid in image_grids
    ]


def test_native_processor_size_layout_ignores_trust_permission(
    ernie_context,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(ernie_context.model_config, "trust_remote_code", True)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ernie_context.model_config,
        tokenizer=ernie_context.tokenizer,
    )

    assert processor.info.get_max_image_tokens() > 0


@pytest.mark.parametrize(
    ("configured_max_pixels", "runtime_max_pixels", "expected_grid"),
    [
        (None, 12544, 8),
        (12544, None, 8),
        (12544, 50176, 16),
    ],
)
def test_native_pixel_aliases_match_processor_size(
    ernie_context,
    monkeypatch: pytest.MonkeyPatch,
    configured_max_pixels: int | None,
    runtime_max_pixels: int | None,
    expected_grid: int,
):
    runtime_kwargs = {}
    if configured_max_pixels is not None:
        mm_config = ernie_context.model_config.get_multimodal_config()
        monkeypatch.setattr(
            mm_config,
            "mm_processor_kwargs",
            {"min_pixels": 3136, "max_pixels": configured_max_pixels},
        )
    if runtime_max_pixels is not None:
        runtime_kwargs = {
            "min_pixels": 3136,
            "max_pixels": runtime_max_pixels,
        }

    processor = MULTIMODAL_REGISTRY.create_processor(
        ernie_context.model_config,
        tokenizer=ernie_context.tokenizer,
    )
    mm_items = processor.info.parse_mm_data(
        {"image": Image.new("RGB", (500, 500), "white")}
    )
    result = processor(
        "<|image@placeholder|>",
        mm_items=mm_items,
        hf_processor_mm_kwargs=runtime_kwargs,
    )

    image_grid = result["mm_kwargs"].get_data()["image_grid_thw"]
    assert image_grid.tolist() == [[1, expected_grid, expected_grid]]
    expected_tokens = processor.info.get_num_image_tokens(
        image_width=500,
        image_height=500,
        image_processor=processor.info.get_image_processor(),
        mm_kwargs=runtime_kwargs,
    )
    assert expected_tokens == int(image_grid.prod()) // (
        processor.info.get_hf_config().spatial_conv_size ** 2
    )


def test_native_video_dummy_matches_profiled_tokens(ernie_context):
    processor = MULTIMODAL_REGISTRY.create_processor(
        ernie_context.model_config,
        tokenizer=ernie_context.tokenizer,
    )
    seq_len = 8192
    mm_counts = {"image": 0, "video": 1}
    dummy_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
        seq_len, mm_counts, {}
    )
    video, _ = dummy_inputs.mm_data_items["video"][0]
    target_size = processor.info.get_video_size_with_most_features()

    assert video.shape[1:3] == (target_size.width, target_size.height)

    result = processor(
        dummy_inputs.prompt,
        mm_items=dummy_inputs.mm_data_items,
    )
    (placeholder,) = result["mm_placeholders"]["video"]
    assert placeholder.length == processor.info.get_max_video_tokens(seq_len, mm_counts)


@pytest.mark.parametrize(
    ("video_type", "do_sample_frames"),
    [("numpy", False), ("torch", True)],
)
def test_native_video_matches_hf_and_preserves_cache_inputs(
    ernie_context,
    video_type: str,
    do_sample_frames: bool,
):
    cache = MultiModalProcessorOnlyCache(ernie_context.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ernie_context.model_config,
        tokenizer=ernie_context.tokenizer,
        cache=cache,
    )
    hf_processor = processor.info.get_hf_processor()
    prompt = f"{hf_processor.video_token} Describe this input."

    num_frames = 24 if do_sample_frames else 16
    total_num_frames = num_frames if do_sample_frames else 32
    frame_indices = (
        list(range(num_frames))
        if do_sample_frames
        else np.linspace(0, total_num_frames - 1, num_frames, dtype=int).tolist()
    )
    video_array = np.transpose(_make_video(num_frames), (0, 3, 1, 2)).copy()
    video = (
        torch.from_numpy(video_array.copy()) if video_type == "torch" else video_array
    )
    video_before = _clone_video(video)
    metadata = _metadata(frame_indices, total_num_frames, do_sample_frames)
    metadata_before = deepcopy(metadata)
    frame_indices_before = metadata["frames_indices"]
    mm_items = processor.info.parse_mm_data({"video": (video, metadata)})
    hash_before = _video_hash(prompt, mm_items)
    processor_kwargs = {"num_frames": 16} if do_sample_frames else {}

    hf_metadata = VideoMetadata(
        **{key: value for key, value in metadata.items() if key != "do_sample_frames"}
    )
    direct_hf = hf_processor(
        text=[prompt],
        videos=[_clone_video(video)],
        video_metadata=[hf_metadata],
        do_sample_frames=do_sample_frames,
        return_tensors="pt",
        **processor_kwargs,
    )

    def run_processor():
        return processor(
            prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=processor_kwargs,
        )

    cache_miss = run_processor()
    hash_after_miss = _video_hash(prompt, mm_items)
    cache_hit = run_processor()

    if isinstance(video, np.ndarray):
        assert np.array_equal(video, video_before)
    else:
        assert torch.equal(video, video_before)
    assert metadata == metadata_before
    assert metadata["frames_indices"] is frame_indices_before
    assert hash_before == hash_after_miss == _video_hash(prompt, mm_items)
    assert len(cache._cache) == 1

    expected_ids = direct_hf["input_ids"][0].tolist()
    expected_grid = direct_hf["video_grid_thw"]
    expected_pixels = direct_hf["pixel_values_videos"].to(
        ernie_context.model_config.dtype
    )
    for result in (cache_miss, cache_hit):
        assert result["prompt_token_ids"] == expected_ids
        mm_kwargs = result["mm_kwargs"].get_data()
        assert torch.equal(mm_kwargs["video_grid_thw"], expected_grid)
        assert torch.equal(mm_kwargs["pixel_values_videos"], expected_pixels)


def test_native_video_sampling_policy_resolution():
    processor = object.__new__(Ernie4_5VLMultiModalProcessor)
    configured_kwargs = {"do_sample_frames": False}
    processor.info = SimpleNamespace(
        ctx=SimpleNamespace(
            get_merged_mm_kwargs=lambda kwargs: configured_kwargs | dict(kwargs)
        )
    )
    hf_processor = SimpleNamespace(
        video_processor=SimpleNamespace(do_sample_frames=True)
    )
    video = _make_video(2)

    _, kwargs = processor._prepare_native_video_inputs(
        hf_processor,
        {"videos": [(video, _metadata([0, 1], 2))]},
        {},
    )
    assert kwargs["do_sample_frames"] is False

    configured_kwargs.clear()
    with pytest.raises(ValueError, match="same do_sample_frames policy"):
        processor._prepare_native_video_inputs(
            hf_processor,
            {
                "videos": [
                    (video, _metadata([0, 1], 2, False)),
                    (video, _metadata([0, 1], 2, True)),
                ]
            },
            {},
        )
