# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from PIL import Image

from tests.utils import RemoteLaunchRenderServer
from vllm import envs
from vllm.config.multimodal import MultiModalDummyOptions, VideoDummyOptions
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.molmo2 import (
    Molmo2DummyInputsBuilder,
    Molmo2VisionBackbone,
    build_flat_image_bool_length,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import encode_image_url

from ...utils import build_model_context

pytestmark = pytest.mark.skip_global_cleanup


def test_build_flat_image_bool_length_matches_molmoweb_processor_tokens():
    hf_config = SimpleNamespace(
        image_patch_id=151938,
        low_res_image_start_token_id=151940,
        image_start_token_id=151936,
        image_col_id=151939,
        image_end_token_id=151937,
    )
    image_grids = torch.tensor([[14, 14, 14, 23]], dtype=torch.long)

    image_tokens, num_image_tokens = build_flat_image_bool_length(
        image_grids,
        hf_config,
        image_use_col_tokens=True,
        use_single_crop_col_tokens=None,
        use_single_crop_start_token=False,
    )

    assert num_image_tokens.tolist() == [550]
    assert len(image_tokens) == 550
    assert image_tokens[0].item() == hf_config.image_start_token_id
    assert (image_tokens == hf_config.image_col_id).sum().item() == 28


def test_build_flat_image_bool_length_respects_disabled_col_tokens():
    hf_config = SimpleNamespace(
        image_patch_id=151938,
        low_res_image_start_token_id=151940,
        image_start_token_id=151936,
        image_col_id=151939,
        image_end_token_id=151937,
    )
    image_grids = torch.tensor([[2, 3, 5, 7]], dtype=torch.long)

    image_tokens, num_image_tokens = build_flat_image_bool_length(
        image_grids,
        hf_config,
        image_use_col_tokens=False,
        use_single_crop_col_tokens=False,
        use_single_crop_start_token=True,
    )

    assert num_image_tokens.tolist() == [45]
    assert len(image_tokens) == 45
    assert image_tokens[0].item() == hf_config.low_res_image_start_token_id
    assert (image_tokens == hf_config.image_col_id).sum().item() == 0


@pytest.mark.parametrize(
    ("num_frames_override", "expected_frames"),
    [(1, 2), (2, 2), (3, 3)],
)
def test_dummy_video_num_frames_override_honors_min_of_two(
    num_frames_override, expected_frames
):
    """A `num_frames` override below 2 must be ignored (the model needs at
    least 2 frames), matching the "cannot be less than 2, will be ignored"
    warning."""
    builder = object.__new__(Molmo2DummyInputsBuilder)
    builder.info = SimpleNamespace(
        get_hf_processor=lambda: SimpleNamespace(
            video_processor=SimpleNamespace(size={"width": 64, "height": 64})
        ),
        get_num_frames_with_most_features=lambda seq_len, num_videos: 16,
        get_image_size_with_most_features=lambda: (64, 64),
    )

    data = builder.get_dummy_mm_data(
        seq_len=128,
        mm_counts={"image": 0, "video": 1},
        mm_options=MultiModalDummyOptions(
            video=VideoDummyOptions(num_frames=num_frames_override)
        ),
    )

    videos = data["video"]
    assert isinstance(videos, list)
    video, _metadata = videos[0]
    assert video.shape[0] == expected_frames


@pytest.mark.parametrize("masked_average", [False, True])
def test_vision_pooling_means_preserve_batch_indices_and_drop_invalid_groups(
    masked_average,
):
    """Isolate pooling preparation from the vision encoder and learned layers."""
    image_features = (
        torch.tensor([2, 4, 6, 8, 10, 20, 30, 40]).float().reshape(2, 2, 2, 1)
    )
    token_pooling = torch.tensor(
        [
            [[0, 3, 3, -1], [-1, -1, -1, -1]],
            [[0, 2, 2, -1], [1, 3, 2, 0]],
        ],
        dtype=torch.int32,
    )
    expected_query = torch.tensor(
        [6 if masked_average else 4.5, 0, 70 / 3 if masked_average else 17.5, 25]
    ).reshape(4, 1, 1)

    def pool(query, to_pool, *, attn_mask):
        expected_patches = (
            torch.tensor(
                [[2, 8, 8, 0], [0, 0, 0, 0], [10, 30, 30, 0], [20, 40, 30, 10]]
            )
            .float()
            .unsqueeze(-1)
        )
        torch.testing.assert_close(to_pool, expected_patches)
        torch.testing.assert_close(query, expected_query)
        if masked_average:
            torch.testing.assert_close(
                attn_mask, (token_pooling >= 0).reshape(4, 1, 1, 4)
            )
        else:
            assert attn_mask is None
        return query

    backbone = SimpleNamespace(
        device=image_features.device,
        dtype=image_features.dtype,
        adapter_config=SimpleNamespace(pooling_attention_mask=masked_average),
        encode_image=lambda images: images,
        image_pooling_2d=pool,
        image_projector=torch.nn.Identity(),
    )
    actual = Molmo2VisionBackbone.forward(backbone, image_features, token_pooling)
    torch.testing.assert_close(actual, expected_query[[0, 2, 3], 0])


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("max_crops", [1, 2])
@pytest.mark.parametrize("prompt_as_tokens", [False, True])
def test_crop_kwargs_match_processed_image_tokens(max_crops, prompt_as_tokens):
    """Crop counts and placeholders follow the actual processor output."""
    images = [Image.new("RGB", (1600, 1200)), Image.new("RGB", (400, 300))]
    kwargs = {"max_crops": max_crops}
    results = []
    for on_engine in (False, True):
        ctx = build_model_context(
            "allenai/Molmo2-8B",
            mm_processor_kwargs=kwargs if on_engine else None,
            limit_mm_per_prompt={"image": 2},
            mm_processor_cache_gb=0.01,
        )
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
        prompt = processor.dummy_inputs.get_dummy_text({"image": 2})
        if prompt_as_tokens:
            prompt = ctx.get_tokenizer().encode(prompt)
        for _ in range(2):
            result = processor(
                prompt,
                mm_items=processor.info.parse_mm_data({"image": images}),
                hf_processor_mm_kwargs={} if on_engine else kwargs,
            )
            for placeholder, item in zip(
                result["mm_placeholders"]["image"], result["mm_kwargs"]["image"]
            ):
                assert item is not None
                image_tokens = item["image_tokens"].data
                pixels = item["pixel_values"].data
                assert isinstance(image_tokens, torch.Tensor)
                assert isinstance(pixels, torch.Tensor)
                expected = image_tokens.tolist()
                assert placeholder.length == len(expected)
                start = placeholder.offset
                assert (
                    result["prompt_token_ids"][start : start + len(expected)]
                    == expected
                )
                assert pixels.shape[0] <= max_crops + 1
        results.append(result["prompt_token_ids"])
    assert results[0] == results[1]


@pytest.mark.usefixtures("default_vllm_config")
def test_molmo2_rejects_cumulative_image_crops_before_processing(monkeypatch):
    ctx = build_model_context(
        "allenai/Molmo2-8B",
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=0,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [Image.new("RGB", (400, 400), color) for color in ("red", "blue")]
    kwargs = {"max_crops": 4}

    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", 25_000_000)
    prompt = processor.dummy_inputs.get_dummy_text({"image": 2})
    with pytest.raises(VLLMValidationError, match="Image preprocessing exceeds"):
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data({"image": images}),
            hf_processor_mm_kwargs=kwargs,
        )

    one_image = processor(
        processor.dummy_inputs.get_dummy_text({"image": 1}),
        mm_items=processor.info.parse_mm_data({"image": images[:1]}),
        hf_processor_mm_kwargs=kwargs,
    )
    assert len(one_image["mm_kwargs"]["image"]) == 1


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("layout", ["chw", "rgba"])
def test_molmo2_image_budget_preserves_array_processing(monkeypatch, layout):
    ctx = build_model_context(
        "allenai/Molmo2-8B",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    image = (
        torch.from_numpy(np.array(Image.new("RGB", (64, 80)))).permute(2, 0, 1)
        if layout == "chw"
        else np.array(Image.new("RGBA", (64, 80)))
    )
    prompt = processor.dummy_inputs.get_dummy_text({"image": 1})
    items = processor.info.parse_mm_data({"image": [image]})

    def process_image(limit: int):
        monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", limit)
        return processor(prompt, mm_items=items)

    budget = 100_000_000
    try:
        expected = process_image(0)
    except ValueError as exc:
        assert isinstance(exc.__cause__, ValueError)
        with pytest.raises(ValueError, match="Failed to apply") as exc_info:
            process_image(budget)
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == str(exc.__cause__)
    else:
        actual = process_image(budget)
        assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
        (expected_image,) = expected["mm_kwargs"]["image"]
        (actual_image,) = actual["mm_kwargs"]["image"]
        assert expected_image is not None and actual_image is not None
        expected_pixels = expected_image["pixel_values"].data
        actual_pixels = actual_image["pixel_values"].data
        assert isinstance(expected_pixels, torch.Tensor)
        assert isinstance(actual_pixels, torch.Tensor)
        torch.testing.assert_close(actual_pixels, expected_pixels, rtol=0, atol=0)


def test_molmo2_render_validates_image_preprocessing():
    model = "allenai/Molmo2-8B"
    content = [
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(Image.new("RGB", (400, 400), color))},
        }
        for color in ("red", "blue")
    ]
    with RemoteLaunchRenderServer(
        model,
        [
            "--host",
            "127.0.0.1",
            "--dtype",
            "float32",
            "--max-model-len",
            "4096",
            "--mm-processor-cache-gb",
            "0",
            "--trust-remote-code",
            "--trust-request-mm-kwargs",
        ],
        env_dict={"VLLM_MAX_MEDIA_PREPROCESSING_BYTES": "25000000"},
    ) as server:
        url = server.url_for("v1/chat/completions/render")
        malformed = httpx.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": content[:1]}],
                "mm_processor_kwargs": {"images_kwargs": {"size": {"height": 378}}},
            },
            timeout=120,
        )
        assert malformed.status_code == 400, malformed.text[:512]
        assert "size must contain 'height' and 'width' keys" in malformed.text

        for images, status in ((content, 400), (content[:1], 200)):
            response = httpx.post(
                url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": images}],
                },
                timeout=120,
            )
            assert response.status_code == status, response.text[:512]
            if status == 400:
                assert "Image preprocessing exceeds" in response.text
