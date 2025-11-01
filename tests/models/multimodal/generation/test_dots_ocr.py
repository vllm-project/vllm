# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import asdict

import pytest

from vllm import LLM, EngineArgs, SamplingParams
from vllm.attention.backends.registry import _MHA_Backend
from vllm.multimodal.utils import encode_image_base64
from vllm.platforms import current_platform

MODEL_NAME = "rednote-hilab/dots.ocr"

# Exact prompt from dots.ocr
# https://github.com/rednote-hilab/dots.ocr/blob/d72d1d8c5bdd0362eb264f714cdbd1e5daa7cdff/dots_ocr/utils/prompts.py#L3
# ruff: noqa: E501
PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


@pytest.mark.core_model
@pytest.mark.parametrize("prompt", [PROMPT])
@pytest.mark.parametrize(
    "mm_encoder_attn_backend",
    [None] + current_platform.get_supported_vit_attn_backends(),
)
def test_dots_ocr_vit_attn_backend_functionality(
    image_assets,
    prompt: str,
    mm_encoder_attn_backend: _MHA_Backend | None,
):
    # images = [asset.pil_image for asset in image_assets]
    # Use the stop_sign image which has clear text
    stop_sign_image = [
        asset.pil_image for asset in image_assets if asset.name == "stop_sign"
    ][0]

    image_urls = [f"data:image/jpeg;base64,{encode_image_base64(stop_sign_image)}"]

    engine_args = EngineArgs(
        model=MODEL_NAME,
        trust_remote_code=True,
        max_model_len=32768,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
        mm_encoder_attn_backend=mm_encoder_attn_backend,
    )

    # From the demo example of dots.ocr
    # https://github.com/rednote-hilab/dots.ocr/blob/d72d1d8c5bdd0362eb264f714cdbd1e5daa7cdff/dots_ocr/model/inference.py#L22

    placeholders = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    ]
    messages = [
        {
            "role": "user",
            "content": [
                *placeholders,
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"},
            ],
        },
    ]

    engine_args = asdict(engine_args) | {"seed": 42}
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=16384,
        stop_token_ids=None,
        top_p=0.9,
    )

    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
    )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        assert len(generated_text) > 10, (
            f"Generated text is too short: {generated_text}"
        )
        assert "stop" in generated_text.lower(), (
            f"Generated text does not contain 'stop': {generated_text}"
        )
        print("-" * 50)
