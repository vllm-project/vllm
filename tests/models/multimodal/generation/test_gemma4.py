# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from vllm import SamplingParams

MODEL_ID = "google/gemma-4-E2B-it"


@pytest.mark.core_model
def test_unresized_image_engine_starts_and_generates(vllm_runner) -> None:
    """With resizing disabled, startup profiling must build a patch-aligned
    dummy, and a patch-aligned image must keep its full grid of soft tokens."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    image = Image.new("RGB", (20 * 48, 14 * 48), color=(30, 120, 200))

    with vllm_runner(
        MODEL_ID,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1, "audio": 0, "video": 0},
        mm_processor_kwargs={"do_resize": False},
    ) as vllm_model:
        outputs = vllm_model.llm.generate(
            vllm_model.get_inputs([prompt], images=[image]),
            sampling_params=SamplingParams(temperature=0, max_tokens=8),
        )

    assert len(outputs) == 1
    assert outputs[0].outputs[0].token_ids
    assert outputs[0].prompt_token_ids.count(processor.image_token_id) == 20 * 14
