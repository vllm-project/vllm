# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.multimodal import MultiModalUUIDDict
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor

cherry_pil_image = ImageAsset("cherry_blossom").pil_image
stop_pil_image = ImageAsset("stop_sign").pil_image
baby_reading_np_ndarrays = VideoAsset("baby_reading").np_ndarrays


def _build_input_processor(
    *, mm_cache_gb: float = 4.0, enable_prefix_caching: bool = True
) -> InputProcessor:
    model_config = ModelConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        skip_tokenizer_init=True,
        max_model_len=128,
        mm_processor_cache_gb=mm_cache_gb,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(enable_prefix_caching=enable_prefix_caching),
    )

    return InputProcessor(vllm_config)


def test_multi_modal_uuids_length_mismatch_raises():
    input_processor = _build_input_processor()

    prompt = {
        "prompt": "USER: <image>\nDescribe\nASSISTANT:",
        "multi_modal_data": {"image": [cherry_pil_image, stop_pil_image]},
        # Mismatch: 2 items but only 1 uuid provided
        "multi_modal_uuids": {"image": ["hash_cherry"]},
    }

    with pytest.raises(ValueError, match="must have same length as"):
        input_processor.process_inputs(
            request_id="req-1",
            prompt=prompt,  # type: ignore[arg-type]
            params=SamplingParams(),
        )


def test_multi_modal_uuids_missing_modality_raises():
    input_processor = _build_input_processor()

    prompt = {
        "prompt": "USER: <image><video>\nDescribe\nASSISTANT:",
        # Two modalities provided in data
        "multi_modal_data": {
            "image": [cherry_pil_image],
            "video": None,
        },
        # Only image uuids provided; video missing should raise
        "multi_modal_uuids": {"image": ["hash_cherry"]},
    }

    with pytest.raises(ValueError, match="is empty but .* is missing"):
        input_processor.process_inputs(
            request_id="req-2",
            prompt=prompt,  # type: ignore[arg-type]
            params=SamplingParams(),
        )


@pytest.mark.parametrize(
    "mm_cache_gb, enable_prefix_caching",
    [
        (4.0, True),  # default behavior
        (4.0, False),  # prefix caching disabled
        (0.0, True),  # processor cache disabled
    ],
)
def test_multi_modal_uuids_accepts_none_and_passes_through(
    monkeypatch, mm_cache_gb: float, enable_prefix_caching: bool
):
    input_processor = _build_input_processor(
        mm_cache_gb=mm_cache_gb,
        enable_prefix_caching=enable_prefix_caching,
    )

    # Capture the overrides passed to InputPreprocessor.preprocess
    captured: dict[str, object] = {}

    def fake_preprocess(
        prompt, *, tokenization_kwargs=None, lora_request=None, mm_uuids=None
    ):
        captured["mm_uuids"] = mm_uuids
        # Minimal processed inputs for decoder-only flow
        return {"type": "token", "prompt_token_ids": [1]}

    # Monkeypatch only the bound preprocess method on this instance
    monkeypatch.setattr(
        input_processor.input_preprocessor, "preprocess", fake_preprocess, raising=True
    )

    # Use a consistent two-image scenario across all configurations
    mm_uuids = {"image": [None, "hash_stop"], "video": None}
    prompt = {
        "prompt": "USER: <image><image>\nTwo images\nASSISTANT:",
        "multi_modal_data": {
            "image": [cherry_pil_image, stop_pil_image],
            "video": baby_reading_np_ndarrays,
        },
        "multi_modal_uuids": mm_uuids,
    }

    input_processor.process_inputs(
        request_id="req-3",
        prompt=prompt,  # type: ignore[arg-type]
        params=SamplingParams(),
    )

    assert captured["mm_uuids"] == mm_uuids


def test_multi_modal_uuids_ignored_when_caching_disabled(monkeypatch):
    # When both processor cache is 0 and prefix caching disabled, the
    # processor builds overrides from request id instead of using user UUIDs.
    input_processor = _build_input_processor(
        mm_cache_gb=0.0, enable_prefix_caching=False
    )

    captured: dict[str, MultiModalUUIDDict] = {}

    def fake_preprocess(
        prompt, *, tokenization_kwargs=None, lora_request=None, mm_uuids=None
    ):
        captured["mm_uuids"] = mm_uuids
        return {"type": "token", "prompt_token_ids": [1]}

    monkeypatch.setattr(
        input_processor.input_preprocessor, "preprocess", fake_preprocess, raising=True
    )

    request_id = "req-42"
    mm_uuids = {"image": ["hash_cherry", "hash_stop"], "video": ["hash_video"]}
    prompt = {
        "prompt": "USER: <image><image><video>\nDescribe\nASSISTANT:",
        "multi_modal_data": {
            "image": [cherry_pil_image, stop_pil_image],
            "video": [baby_reading_np_ndarrays],
        },
        "multi_modal_uuids": mm_uuids,
    }

    input_processor.process_inputs(
        request_id=request_id,
        prompt=prompt,  # type: ignore[arg-type]
        params=SamplingParams(),
    )

    # Expect request-id-based overrides are passed through
    assert set(mm_uuids.keys()) == {"image", "video"}
    assert len(mm_uuids["image"]) == 2
    assert len(mm_uuids["video"]) == 1
    assert captured["mm_uuids"]["image"][0].startswith(
        f"{request_id}-image-"
    ) and captured["mm_uuids"]["image"][0].endswith("-0")
    assert captured["mm_uuids"]["image"][1].startswith(
        f"{request_id}-image-"
    ) and captured["mm_uuids"]["image"][1].endswith("-1")
    assert captured["mm_uuids"]["video"][0].startswith(
        f"{request_id}-video-"
    ) and captured["mm_uuids"]["video"][0].endswith("-0")
