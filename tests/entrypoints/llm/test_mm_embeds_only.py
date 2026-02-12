# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "llava-hf/llava-1.5-7b-hf"
PROMPT = "USER: <image>\nDescribe this image briefly.\nASSISTANT:"
TEXT_ONLY_PROMPT = "USER: What is 2 + 2?\nASSISTANT:"


@pytest.fixture(scope="module")
def llm():
    """LLM with enable_mm_embeds=True and all modality limits zeroed out."""
    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 0},
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
def test_generate_with_embedding(llm: LLM):
    """Pre-computed embedding produces tokens without hanging."""
    embedding = ImageAsset("stop_sign").image_embeds
    outputs = llm.generate(
        {"prompt": PROMPT, "multi_modal_data": {"image": embedding}},
        sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0


@pytest.mark.skip_global_cleanup
def test_raw_image_rejected(llm: LLM):
    """Raw image input is still rejected when limit=0."""
    raw_image = ImageAsset("stop_sign").pil_image
    with pytest.raises(ValueError, match=r"At most 0 image\(s\)"):
        llm.generate(
            {"prompt": PROMPT, "multi_modal_data": {"image": raw_image}},
            sampling_params=SamplingParams(max_tokens=16),
        )


@pytest.mark.skip_global_cleanup
def test_text_only_prompt(llm: LLM):
    """Text-only prompts still work under this config."""
    outputs = llm.generate(
        TEXT_ONLY_PROMPT,
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
    )
    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0
