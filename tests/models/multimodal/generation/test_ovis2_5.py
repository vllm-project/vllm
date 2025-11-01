# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict

import pytest

from vllm import LLM, EngineArgs, SamplingParams
from vllm.attention.backends.registry import _MHA_Backend
from vllm.multimodal.utils import encode_image_base64
from vllm.platforms import current_platform

# This model uses the ViT Class from
# vllm/model_executor/models/siglip2navit.py
MODEL_NAME = "AIDC-AI/Ovis2.5-2B"

QUESTION = "What is the content of each image?"


@pytest.mark.parametrize("question", [QUESTION])
@pytest.mark.parametrize(
    "mm_encoder_attn_backend",
    [None] + current_platform.get_supported_vit_attn_backends(),
)
def test_ovis2_5_vit_attn_backend_functionality(
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
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": len(image_urls)},
        mm_encoder_attn_backend=mm_encoder_attn_backend,
    )

    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    prompt = (
        f"<|im_start|>user\n\n{placeholders}\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
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
