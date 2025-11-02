# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict

import pytest
from transformers import AutoProcessor

from vllm import LLM, EngineArgs, SamplingParams
from vllm.attention.backends.registry import _MHA_Backend
from vllm.multimodal.utils import encode_image_base64
from vllm.platforms import current_platform

from ....utils import large_gpu_test

MODEL_NAME = "baidu/ERNIE-4.5-VL-28B-A3B-PT"

QUESTION = "What is the content of each image?"


@large_gpu_test(min_gb=80)
@pytest.mark.parametrize("question", [QUESTION])
@pytest.mark.parametrize(
    "mm_encoder_attn_backend",
    [None] + current_platform.get_supported_vit_attn_backends(),
)
def test_ernie45_vl_vit_attn_backend_functionality(
    image_assets,
    question: str,
    mm_encoder_attn_backend: _MHA_Backend | None,
):
    images = [asset.pil_image for asset in image_assets]

    image_urls = [
        f"data:image/jpeg;base64,{encode_image_base64(image)}" for image in images
    ]

    engine_args = EngineArgs(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_model_len=16384,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_encoder_attn_backend=mm_encoder_attn_backend,
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
