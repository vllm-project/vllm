# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.multimodal.parse import parse_mm_uuids
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import tokenizer_args_from_config

cherry_pil_image = ImageAsset("cherry_blossom").pil_image
stop_pil_image = ImageAsset("stop_sign").pil_image
baby_reading_np_ndarrays = VideoAsset("baby_reading").np_ndarrays


def _build_renderer(
    *, mm_cache_gb: float = 4.0, enable_prefix_caching: bool = True
) -> HfRenderer:
    model_config = ModelConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=128,
        mm_processor_cache_gb=mm_cache_gb,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(enable_prefix_caching=enable_prefix_caching),
    )

    _, tokenizer_name, _, kwargs = tokenizer_args_from_config(model_config)

    return HfRenderer.from_config(
        vllm_config,
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )


def test_multi_modal_uuids_length_mismatch_raises():
    renderer = _build_renderer()

    mm_data = {"image": [cherry_pil_image, stop_pil_image]}

    # Mismatch: 2 items but only 1 uuid provided
    mm_uuids = {"image": ["hash_cherry"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="must have same length as"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-1")


def test_multi_modal_uuids_missing_modality_raises():
    renderer = _build_renderer()

    mm_data = {
        "image": [cherry_pil_image],
        "video": None,
    }

    # Only image uuids provided; video missing should raise
    mm_uuids = {"image": ["hash_cherry"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="is empty but .* is missing"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-2")


@pytest.mark.parametrize(
    "mm_cache_gb, enable_prefix_caching",
    [
        (4.0, True),  # default behavior
        (4.0, False),  # prefix caching disabled
        (0.0, True),  # processor cache disabled
    ],
)
def test_multi_modal_uuids_accepts_none_and_passes_through(
    mm_cache_gb: float, enable_prefix_caching: bool
):
    renderer = _build_renderer(
        mm_cache_gb=mm_cache_gb,
        enable_prefix_caching=enable_prefix_caching,
    )

    mm_data = {
        "image": [cherry_pil_image, stop_pil_image],
        "video": baby_reading_np_ndarrays,
    }

    # Use a consistent two-image scenario across all configurations
    mm_uuids = {"image": [None, "hash_stop"], "video": None}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, "req-3"
    )

    assert processed_mm_uuids == mm_uuids


@pytest.mark.parametrize(
    "mm_cache_gb, enable_prefix_caching",
    [
        (4.0, True),  # default behavior
        (4.0, False),  # prefix caching disabled
        (0.0, True),  # processor cache disabled
    ],
)
def test_multi_modal_uuids_accepts_empty(
    mm_cache_gb: float, enable_prefix_caching: bool
):
    renderer = _build_renderer(
        mm_cache_gb=mm_cache_gb,
        enable_prefix_caching=enable_prefix_caching,
    )

    # While None means cached multi-modal input requiring UUIDs
    # an empty list means no multi-modal input
    mm_data = {"image": [], "video": []}  # type: ignore[var-annotated]
    mm_uuids = {"image": [], "video": None}  # type: ignore[var-annotated]

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, "req-4"
    )

    assert processed_mm_uuids == mm_uuids


def test_multi_modal_uuids_ignored_when_caching_disabled():
    # When both processor cache is 0 and prefix caching disabled, the
    # processor builds overrides from request id instead of using user UUIDs.
    renderer = _build_renderer(mm_cache_gb=0.0, enable_prefix_caching=False)

    request_id = "req-42"
    mm_data = {
        "image": [cherry_pil_image, stop_pil_image],
        "video": baby_reading_np_ndarrays,
    }
    mm_uuids = {"image": ["hash_cherry", "hash_stop"], "video": ["hash_video"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, request_id
    )

    # Expect request-id-based overrides are passed through
    assert set(mm_uuids.keys()) == {"image", "video"}
    assert len(mm_uuids["image"]) == 2
    assert len(mm_uuids["video"]) == 1
    assert processed_mm_uuids["image"][0].startswith(
        f"{request_id}-image-"
    ) and processed_mm_uuids["image"][0].endswith("-0")
    assert processed_mm_uuids["image"][1].startswith(
        f"{request_id}-image-"
    ) and processed_mm_uuids["image"][1].endswith("-1")
    assert processed_mm_uuids["video"][0].startswith(
        f"{request_id}-video-"
    ) and processed_mm_uuids["video"][0].endswith("-0")
