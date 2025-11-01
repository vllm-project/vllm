# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict
from typing import NamedTuple

import pytest
from PIL.Image import Image
from transformers import AutoProcessor

from vllm import LLM, EngineArgs, SamplingParams
from vllm.multimodal.utils import encode_image_base64

MODEL_NAME = "Kwai-Keye/Keye-VL-8B-Preview"

QUESTION = "What is the content of each image?"


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: list[Image]
    stop_token_ids: list[int] | None = None
    chat_template: str | None = None
    sampling_params: SamplingParams | None = None


@pytest.mark.core_model
@pytest.mark.parametrize("question", [QUESTION])
def test_keye_vl(
    image_assets,
    question: str,
):
    images = [asset.pil_image for asset in image_assets]

    image_urls = [
        f"data:image/jpeg;base64,{encode_image_base64(image)}" for image in images
    ]

    engine_args = EngineArgs(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_model_len=8192,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
    )

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

    engine_args = asdict(engine_args) | {"seed": 42}
    llm = LLM(**engine_args)

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
