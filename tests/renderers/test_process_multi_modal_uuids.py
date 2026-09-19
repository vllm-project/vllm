# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

import pytest

from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.entrypoints.serve import create_error_response
from vllm.multimodal.parse import parse_mm_uuids
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config

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

    return HfRenderer(
        vllm_config,
        cached_tokenizer_from_config(model_config),
    )


def _build_text_only_renderer() -> HfRenderer:
    model_config = ModelConfig(model="openai-community/gpt2", max_model_len=128)

    return HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )


def test_text_only_model_mm_data_maps_to_bad_request():
    """Sending multimodal data to a text-only model is a client mistake, so it
    must surface as a ValueError and reach the client as HTTP 400, not 500."""
    renderer = _build_text_only_renderer()

    with pytest.raises(ValueError, match="text-only") as exc_info:
        renderer._process_multimodal(
            prompt=[1],
            mm_data={"image": [cherry_pil_image]},
            mm_uuids=None,
            mm_processor_kwargs=None,
        )

    error_response = create_error_response(exc_info.value)
    assert error_response.error.code == HTTPStatus.BAD_REQUEST


def test_multi_modal_uuids_length_mismatch_raises():
    renderer = _build_renderer()

    mm_data = {"image": [cherry_pil_image, stop_pil_image]}

    # Mismatch: 2 items but only 0 uuids provided
    mm_uuids = {"image": []}  # type: ignore[var-annotated]

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="must have same length as"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-1a")

    # Mismatch: 2 items but only 1 uuid provided
    mm_uuids = {"image": ["hash_cherry"]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    with pytest.raises(ValueError, match="must have same length as"):
        renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-1b")


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
    mm_data = {"image": [], "video": [], "audio": None}  # type: ignore[var-annotated]
    mm_uuids = {"image": [], "video": None, "audio": []}  # type: ignore[var-annotated]

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, "req-4"
    )

    assert processed_mm_uuids == mm_uuids


@pytest.mark.parametrize(
    "mm_uuids, expected",
    [
        (
            {"image": ["hash_cherry", "hash_stop"], "video": ["hash_video"]},
            {"image": ["hash_cherry", "hash_stop"], "video": ["hash_video"]},
        ),
        (
            {"image": [None, "hash_stop"], "video": None},
            {"image": ["req-42-image-0", "hash_stop"], "video": ["req-42-video-0"]},
        ),
        (
            {"image": ["", None]},
            {"image": ["", "req-42-image-1"], "video": ["req-42-video-0"]},
        ),
        (
            {},
            {
                "image": ["req-42-image-0", "req-42-image-1"],
                "video": ["req-42-video-0"],
            },
        ),
    ],
)
def test_multi_modal_uuids_preserved_when_caching_disabled(mm_uuids, expected):
    """Only missing UUIDs get request-local IDs when both caches are disabled."""
    renderer = _build_renderer(mm_cache_gb=0.0, enable_prefix_caching=False)

    request_id = "req-42"
    mm_data = {
        "image": [cherry_pil_image, stop_pil_image],
        "video": baby_reading_np_ndarrays,
    }

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids(mm_uuids)

    processed_mm_uuids = renderer._process_mm_uuids(
        mm_data, mm_data_items, mm_uuid_items, request_id
    )

    assert processed_mm_uuids == expected


def test_validate_mm_uuids_does_not_decode_lazy_media():
    """UUID validation only checks None-ness, so it must not unwrap lazy
    items (unwrapping would decode every item on the single _mm_executor
    worker, defeating cache-hit-skips-decode)."""
    from vllm.multimodal.media import LazyMedia

    renderer = _build_renderer()

    decoder_calls = 0

    def decode():
        nonlocal decoder_calls
        decoder_calls += 1
        return baby_reading_np_ndarrays

    mm_data = {"video": [LazyMedia(decode, b"video-bytes")]}

    mm_processor = renderer.get_mm_processor()
    mm_data_items = mm_processor.info.parse_mm_data(mm_data)
    mm_uuid_items = parse_mm_uuids({"video": [None]})

    renderer._process_mm_uuids(mm_data, mm_data_items, mm_uuid_items, "req-lazy")

    assert decoder_calls == 0


@pytest.mark.asyncio
async def test_process_multimodal_async_does_not_block_mm_worker_on_decode():
    """Two-phase `_process_multimodal_async` must free the single
    `_mm_executor` worker while a lazy decode is in flight, so a second
    request's phase 1 can interleave (cross-request decode overlap)."""
    import asyncio
    import threading

    from vllm.multimodal.media import LazyMedia

    renderer = _build_renderer()
    processor = renderer.get_mm_processor()
    assert processor.supports_two_phase_apply

    tokenizer = renderer.tokenizer
    prompt = tokenizer.encode("<|vision_start|><|image_pad|><|vision_end|>")

    decode_entered = threading.Event()
    release_decode = threading.Event()

    def gated_decode():
        decode_entered.set()
        release_decode.wait(timeout=60)
        return stop_pil_image

    try:
        task_a = asyncio.create_task(
            renderer._process_multimodal_async(
                prompt, {"image": [LazyMedia(gated_decode, b"a-bytes")]}, None, None
            )
        )
        await asyncio.to_thread(decode_entered.wait, 60)

        # B runs to completion while A's decode is still blocked; with the
        # old single-call blocking apply, B would queue behind A on the
        # single mm worker.
        result_b = await asyncio.wait_for(
            renderer._process_multimodal_async(
                prompt, {"image": [cherry_pil_image]}, None, None
            ),
            timeout=60,
        )
        assert not release_decode.is_set()
        assert result_b["mm_kwargs"]["image"][0] is not None

        release_decode.set()
        result_a = await asyncio.wait_for(task_a, timeout=60)
        assert result_a["mm_kwargs"]["image"][0] is not None
    finally:
        release_decode.set()
