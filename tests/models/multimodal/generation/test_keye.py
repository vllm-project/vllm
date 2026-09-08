# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import NamedTuple

import pytest
from PIL.Image import Image
from transformers import AutoProcessor

from vllm import LLM, EngineArgs, SamplingParams
from vllm.multimodal.utils import encode_image_url

MODEL_NAME = "Kwai-Keye/Keye-VL-8B-Preview"

QUESTION = "What is the content of each image?"


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: list[int] | None = None
    chat_template: str | None = None
    sampling_params: SamplingParams | None = None


@pytest.mark.parametrize("question", [QUESTION])
def test_keye_vl(image_assets, question: str):
    images = [asset.pil_image for asset in image_assets]
    image_urls = [encode_image_url(image) for image in images]

    placeholders = [{"type": "image", "image": url} for url in image_urls]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": question},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
        seed=42,
    )

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=256, stop_token_ids=None
    )

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        },
        sampling_params=sampling_params,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        assert len(generated_text) > 10, (
            f"Generated text is too short: {generated_text}"
        )
        print("-" * 50)
