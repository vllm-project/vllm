# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm.model_executor.models.preprocessing.kimi_k25_preprocessing import (
    KimiK25PreprocessConfig,
    KimiK25Preprocessor,
)
from vllm.multimodal.inputs import VisionChunkImage
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k25 import KimiK25Processor

MODEL_ID = "moonshotai/Kimi-K2.5"
MEDIA_PAD = "<|media_pad|>"


@pytest.fixture(scope="module")
def kimi_k25_hf_processor():
    pytest.importorskip("transformers")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)


@pytest.fixture(scope="module")
def kimi_k25_vllm_preprocessor(kimi_k25_hf_processor):
    tokenizer = kimi_k25_hf_processor.tokenizer
    image_processor = kimi_k25_hf_processor.image_processor
    media_token_id = tokenizer.convert_tokens_to_ids(MEDIA_PAD)
    config = KimiK25PreprocessConfig(
        media_token_id=media_token_id,
        video_placeholder=kimi_k25_hf_processor.video_placeholder,
    )
    return KimiK25Preprocessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        config=config,
    )


def _dummy_preprocessor(config: KimiK25PreprocessConfig) -> KimiK25Preprocessor:
    return KimiK25Preprocessor(
        tokenizer=SimpleNamespace(),  # type: ignore[arg-type]
        image_processor=SimpleNamespace(media_tokens_calculator=lambda _: 0),
        config=config,
    )


def test_expand_media_tokens():
    config = KimiK25PreprocessConfig(media_token_id=99)
    preprocessor = _dummy_preprocessor(config)
    expanded = preprocessor.expand_media_tokens([1, 99, 2, 99, 3], [4, 5])
    assert expanded == [1, 99, 99, 99, 99, 2, 99, 99, 99, 99, 99, 3]


def test_update_raw_text():
    config = KimiK25PreprocessConfig(
        media_token_id=1,
        video_placeholder="<|video|>",
    )
    preprocessor = _dummy_preprocessor(config)
    text = "a<|video|>b<|video|>c"
    updated = preprocessor.update_raw_text(text, ["V1", "V2"])
    assert updated == "aV1bV2c"


def test_wrapper_matches_preprocessor(kimi_k25_vllm_preprocessor):
    image = Image.new("RGB", (224, 224), color=(128, 64, 32))
    vision_chunk = VisionChunkImage(type="image", image=image, uuid=None)
    prompt = (
        "<|media_begin|>image<|media_content|>"
        f"{MEDIA_PAD}<|media_end|>Describe."
    )

    wrapper = KimiK25Processor(kimi_k25_vllm_preprocessor)
    wrapped = wrapper(
        text=prompt,
        vision_chunks=[vision_chunk],
        return_tensors="pt",
    )
    direct = kimi_k25_vllm_preprocessor.preprocess(
        prompt,
        vision_chunks=[vision_chunk],
        return_tensors="pt",
    )

    assert wrapped["input_ids"][0].tolist() == direct.input_ids
    assert torch.equal(wrapped["pixel_values"], direct.pixel_values)
    assert torch.equal(wrapped["grid_thws"], direct.grid_thws)


def test_preprocess_from_medias_matches_hf(kimi_k25_hf_processor, kimi_k25_vllm_preprocessor):
    image = Image.new("RGB", (320, 240), color=(10, 20, 30))
    medias = [{"type": "image", "image": image}]
    text = (
        "<|media_begin|>image<|media_content|>"
        f"{MEDIA_PAD}<|media_end|>Hello"
    )

    hf_out = kimi_k25_hf_processor(text=text, medias=medias, return_tensors="pt")
    vllm_out = kimi_k25_vllm_preprocessor.preprocess_from_medias(
        text=text,
        medias=medias,
        return_tensors="pt",
    )

    assert hf_out["input_ids"][0].tolist() == vllm_out.input_ids
    assert torch.equal(hf_out["pixel_values"], vllm_out.pixel_values)
    assert torch.equal(hf_out["grid_thws"], vllm_out.grid_thws)


def test_vllm_image_processor_loads(kimi_k25_vllm_preprocessor):
    """Sanity check that cached_get_image_processor is usable with the preprocessor."""
    image_processor = cached_get_image_processor(MODEL_ID, trust_remote_code=True)
    assert hasattr(image_processor, "media_tokens_calculator")
    assert callable(kimi_k25_vllm_preprocessor.media_tokens_calculator)
