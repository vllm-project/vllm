# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor

from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.models.kimi_k3.common.mm_preprocess import (
    KIMI_K3_VIDEO_PLACEHOLDER,
    encode_kimi_k3_videos,
    format_kimi_k3_timestamp,
    plan_kimi_k3_video,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)

from ...utils import build_model_context

_MODEL_ID = "moonshotai/Kimi-K3"


@pytest.fixture(scope="module")
def image_processors():
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    k25_fused = getattr(processor.info, "image_processor")  # noqa: B009
    assert isinstance(k25_fused, KimiK25FusedVisionProcessor)
    k3_hf = AutoImageProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    return k25_fused, k3_hf


def _noise_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("width", "height"),
    [(310, 470), (7300, 30), (13, 9)],
)
def test_fused_image_processor_matches_hf(image_processors, width: int, height: int):
    fused, hf = image_processors
    image = _noise_image(width, height)
    expected = hf.preprocess([{"type": "image", "image": image}], return_tensors="pt")
    actual = fused.preprocess([{"type": "image", "image": image}], return_tensors="pt")

    torch.testing.assert_close(actual["grid_thws"], expected["grid_thws"])
    torch.testing.assert_close(actual["pixel_values"], expected["pixel_values"])
    assert fused.media_tokens_calculator(
        {"type": "image", "image": image}
    ) == hf.media_tokens_calculator({"type": "image", "image": image})


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    device = current_platform.device_type
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 2})
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [
        Image.new("RGB", (310, 470), color=(17, 89, 231)),
        Image.new("RGB", (480, 320), color=(201, 13, 127)),
    ]
    prompt = processor.info.get_hf_config().image_placeholder * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized_values = normalized_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(normalized_values, torch.Tensor)

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw_values = raw_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(raw_values, torch.Tensor)
    assert raw_values.dtype == torch.uint8

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.flatten(1).to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.flatten(1).to(device), rtol=1.6e-2, atol=1e-5
    )


# The video keys of the checkpoint's `media_proc_cfg`.
_MEDIA_PROC_CFG = {
    "in_patch_limit": 65536,
    "in_patch_limit_each_frame": 16384,
    "in_patch_limit_video": 655360,
    "sample_fps": 8.0,
    "max_num_frames_each_video": None,
    "temporal_merge_kernel_size": 4,
}


def _video(
    num_frames: int, width: int, height: int, fps: float
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(num_frames)
    frames = rng.integers(0, 256, (num_frames, height, width, 3), dtype=np.uint8)
    return frames, {
        "fps": fps,
        "duration": num_frames / fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
    }


@pytest.mark.parametrize(
    ("seconds", "mode", "expected"),
    [
        (0.0, "hh:mm:ss.fff", "00:00:00.000"),
        (0.5, "hh:mm:ss.fff", "00:00:00.500"),
        (3725.25, "hh:mm:ss.fff", "01:02:05.250"),
        (125.5, "mm:ss.fff", "02:05.500"),
        (125.5, "mm:ss", "02:05"),
    ],
)
def test_format_kimi_k3_timestamp(seconds: float, mode: str, expected: str):
    assert format_kimi_k3_timestamp(seconds, mode) == expected


def test_plan_kimi_k3_video_samples_at_checkpoint_fps():
    # 3 s at 30 fps -> 24 frames at 8 fps -> 6 chunks of 4 frames.
    plan = plan_kimi_k3_video(
        90, {"fps": 30.0, "total_num_frames": 90}, _MEDIA_PROC_CFG
    )
    expected = np.linspace(0, 89, 24).round().astype(int).tolist()
    assert list(plan.frame_indices) == expected
    assert plan.frames_per_chunk == 4
    assert plan.num_chunks == 6
    assert plan.chunk_starts == tuple(expected[i] / 30.0 for i in range(0, 24, 4))
    # 655360 / 24 frames is above the per-frame limit, so the latter binds.
    assert plan.in_patch_limit_each_frame == 16384


def test_plan_kimi_k3_video_spreads_video_budget():
    # 30 s at 8 fps -> 240 frames share the 655360-patch video budget.
    plan = plan_kimi_k3_video(240, {"fps": 8.0}, _MEDIA_PROC_CFG)
    assert len(plan.frame_indices) == 240
    assert plan.num_chunks == 60
    assert plan.in_patch_limit_each_frame == round(655360 / 240)


def test_plan_kimi_k3_video_keeps_loader_timestamps():
    # The loader already sampled 2 fps out of a 30 fps source: keep every
    # decoded frame and take chunk start times from the source indices.
    indices = list(range(0, 300, 15))
    plan = plan_kimi_k3_video(
        len(indices),
        {"fps": 30.0, "total_num_frames": 300, "frames_indices": indices},
        _MEDIA_PROC_CFG,
    )
    assert list(plan.frame_indices) == list(range(20))
    assert plan.chunk_starts == (0.0, 2.0, 4.0, 6.0, 8.0)


@pytest.mark.parametrize(
    "metadata", [None, {"frames_indices": [0, 30, 60, 90, 120, 150]}]
)
def test_plan_kimi_k3_video_without_fps(metadata):
    # Unknown source rate: the decoded frames count as already sampled, and
    # source frame indices cannot be turned into times.
    plan = plan_kimi_k3_video(6, metadata, _MEDIA_PROC_CFG)
    assert list(plan.frame_indices) == list(range(6))
    assert plan.chunk_starts == (0.0, 0.5)


@pytest.fixture(scope="module")
def video_processor():
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1, "video": 2})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


@pytest.mark.parametrize("num_frames", [24, 26])
def test_video_prompt_expansion(video_processor, num_frames: int):
    processor = video_processor
    tokenizer = processor.info.get_tokenizer()
    media_token_id = processor.info.media_token_id
    # 140x112 frames resize to a 10x8 patch grid: 20 tokens per chunk.
    video = _video(num_frames, 140, 112, fps=8.0)
    mm_items = processor.info.parse_mm_data({"video": [video]})
    outputs = processor(f"a{KIMI_K3_VIDEO_PLACEHOLDER}b", mm_items=mm_items)

    num_chunks = -(-num_frames // 4)
    data = outputs["mm_kwargs"].get_data()
    grids = data["video_grid_thws"].tolist()
    assert len(grids) == num_chunks
    assert grids[:-1] == [[4, 8, 10]] * (num_chunks - 1)
    assert grids[-1] == [num_frames - 4 * (num_chunks - 1), 8, 10]
    assert data["video_num_chunks"].tolist() == [num_chunks]

    token_ids = outputs["prompt_token_ids"]
    assert token_ids.count(media_token_id) == 20 * num_chunks
    blocks = "".join(
        f"{format_kimi_k3_timestamp(i * 0.5, 'hh:mm:ss.fff')}"
        "<|media_begin|>video<|media_content|>" + "<|media_pad|>" * 20 + "<|media_end|>"
        for i in range(num_chunks)
    )
    assert tokenizer.decode(token_ids) == f"a{blocks}b"

    [placeholder] = outputs["mm_placeholders"]["video"]
    assert placeholder.get_num_embeds() == 20 * num_chunks


def test_video_as_pil_frames(video_processor):
    processor = video_processor
    frames = [_noise_image(56, 84) for _ in range(4)]
    mm_items = processor.info.parse_mm_data({"video": [(frames, {"fps": 8.0})]})
    outputs = processor(KIMI_K3_VIDEO_PLACEHOLDER, mm_items=mm_items)
    assert outputs["mm_kwargs"].get_data()["video_grid_thws"].tolist() == [[4, 6, 4]]


def test_image_and_video_in_one_prompt(video_processor):
    processor = video_processor
    image_placeholder = processor.info.get_hf_config().image_placeholder
    mm_items = processor.info.parse_mm_data(
        {
            "image": [_noise_image(56, 56)],
            "video": [_video(8, 56, 56, fps=8.0), _video(4, 84, 56, fps=8.0)],
        }
    )
    outputs = processor(
        f"{KIMI_K3_VIDEO_PLACEHOLDER}{image_placeholder}{KIMI_K3_VIDEO_PLACEHOLDER}",
        mm_items=mm_items,
    )
    placeholders = outputs["mm_placeholders"]
    assert [p.get_num_embeds() for p in placeholders["video"]] == [2 * 4, 6]
    assert [p.get_num_embeds() for p in placeholders["image"]] == [4]
    assert (
        placeholders["video"][0].offset
        < placeholders["image"][0].offset
        < placeholders["video"][1].offset
    )


def test_video_fits_profiled_encoder_cache(video_processor):
    info = video_processor.info
    # 480 decoded 768x576 frames at 8 fps: frames padded to whole merge
    # windows need more tokens than the profiling video, so the per-frame
    # patch limit is lowered until the video fits.
    frame = np.zeros((576, 768, 3), dtype=np.uint8)
    item = ([frame] * 480, {"fps": 8.0, "total_num_frames": 480})
    unshrunk = plan_kimi_k3_video(480, item[1], info.media_proc_cfg)
    assert info.video_tokens(768, 576, unshrunk) > info.max_video_tokens

    plan, chunks = info.get_video_chunks(item)
    assert len(chunks) == plan.num_chunks == 120
    assert plan.in_patch_limit_each_frame < unshrunk.in_patch_limit_each_frame
    assert info.video_tokens(768, 576, plan) <= info.max_video_tokens
    assert all(
        c["in_patch_limit_each_frame"] == plan.in_patch_limit_each_frame for c in chunks
    )


def test_video_dummy_fills_profiled_encoder_cache(video_processor):
    info = video_processor.info
    num_frames, width, height = info.dummy_video_shape()
    plan = plan_kimi_k3_video(
        num_frames,
        {"fps": info.media_proc_cfg["sample_fps"], "total_num_frames": num_frames},
        info.media_proc_cfg,
    )
    assert info.video_tokens(width, height, plan) == info.max_video_tokens
    # A shorter video at the largest frame size has fewer tokens.
    short = plan_kimi_k3_video(num_frames // 2, {"fps": 8.0}, info.media_proc_cfg)
    assert info.video_tokens(width, height, short) < info.max_video_tokens


@pytest.mark.parametrize("device_normalize", [True, False])
def test_video_pixels_follow_mm_device_do_normalize(device_normalize: bool):
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"video": 1})
    ctx.model_config.multimodal_config.mm_device_do_normalize = device_normalize
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    mm_items = processor.info.parse_mm_data({"video": [_video(4, 56, 56, 8.0)]})
    outputs = processor(KIMI_K3_VIDEO_PLACEHOLDER, mm_items=mm_items)
    values = outputs["mm_kwargs"].get_data()["video_pixel_values"]
    assert isinstance(values, torch.Tensor)
    # The tower normalizes on device; normalizing here too would do it twice.
    assert (values.dtype == torch.uint8) == device_normalize


def test_encode_kimi_k3_videos_groups_chunks():
    # Two videos: 2 + 3 chunks of 4x4 patches, then one 8x8 chunk.
    grids = torch.tensor(
        [[4, 4, 4], [4, 4, 4], [4, 4, 4], [2, 4, 4], [4, 8, 8]], dtype=torch.long
    )
    num_chunks = torch.tensor([2, 3])
    num_patches = int(grids.prod(-1).sum())
    pixel_values = torch.arange(num_patches, dtype=torch.float32)[:, None]
    calls = []

    def encode(pixels: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        calls.append(int(grid_thws.prod(-1).sum()))
        out, start = [], 0
        for t, h, w in grid_thws.tolist():
            n = t * h * w
            # One row per merge window, tagged with the chunk's first patch.
            out.append(pixels[start].expand(h * w // 4, 3).clone())
            start += n
        return out

    videos = encode_kimi_k3_videos(encode, pixel_values, grids, num_chunks, (2, 2), 128)
    # Groups never exceed 128 patches, except a single chunk that is larger.
    assert calls == [128, 96, 256]
    assert [v.shape for v in videos] == [(8, 3), (24, 3)]
    starts = [0, 64, 128, 192, 224]
    torch.testing.assert_close(
        torch.cat(videos)[:, 0],
        torch.tensor(
            [s for s, n in zip(starts, [4, 4, 4, 4, 16]) for _ in range(n)]
        ).float(),
    )
