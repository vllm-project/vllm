# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone generation tests for Moondream3 model.

These tests verify end-to-end inference capabilities including:
- Basic model loading and generation
- Multi-skill support (Query, Caption, Detect, Point)
- Tensor parallelism (TP=2)
- Various image sizes
"""

import pytest
import torch
from PIL import Image

from ....utils import large_gpu_mark

MOONDREAM3_MODEL_ID = "moondream/moondream3-preview"
MOONDREAM3_TOKENIZER = "moondream/starmie-v1"


def make_query_prompt(question: str) -> str:
    """Create a query prompt for Moondream3."""
    return f"<|endoftext|><image> \n\nQuestion: {question}\n\nAnswer:"


def make_caption_prompt() -> str:
    """Create a caption prompt for Moondream3."""
    return "<|endoftext|><image> \n\nDescribe this image.\n\n"


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
@large_gpu_mark(min_gb=48)
def test_model_loading(model_id: str):
    """Test that the model loads without errors."""
    from vllm import LLM

    llm = LLM(
        model=model_id,
        tokenizer=MOONDREAM3_TOKENIZER,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
    )
    assert llm is not None


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
@large_gpu_mark(min_gb=48)
def test_query_skill(model_id: str):
    """Test query (question answering) skill."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        tokenizer=MOONDREAM3_TOKENIZER,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
    )

    image = Image.new("RGB", (378, 378), color="blue")
    prompt = make_query_prompt("What color is this image?")

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=50, temperature=0),
    )

    output_text = outputs[0].outputs[0].text
    assert output_text is not None
    assert len(output_text) > 0


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
@large_gpu_mark(min_gb=48)
def test_caption_skill(model_id: str):
    """Test caption (image description) skill."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        tokenizer=MOONDREAM3_TOKENIZER,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
    )

    image = Image.new("RGB", (378, 378), color="green")
    prompt = make_caption_prompt()

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=100, temperature=0),
    )

    output_text = outputs[0].outputs[0].text
    assert output_text is not None
    assert len(output_text) > 0


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
@large_gpu_mark(min_gb=48)
def test_batched_inference(model_id: str):
    """Test batched inference with multiple images."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        tokenizer=MOONDREAM3_TOKENIZER,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
    )

    images = [
        Image.new("RGB", (378, 378), color="red"),
        Image.new("RGB", (378, 378), color="blue"),
    ]
    prompts = [
        {"prompt": make_query_prompt("What color?"), "multi_modal_data": {"image": img}}
        for img in images
    ]

    outputs = llm.generate(prompts, SamplingParams(max_tokens=20, temperature=0))

    assert len(outputs) == 2
    for output in outputs:
        assert output.outputs[0].text is not None


@pytest.mark.parametrize(
    "image_size",
    [(100, 100), (378, 378), (800, 600), (1920, 1080)],
)
@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
@large_gpu_mark(min_gb=48)
def test_various_image_sizes(image_size: tuple[int, int], model_id: str):
    """Test inference with various image sizes."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        tokenizer=MOONDREAM3_TOKENIZER,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
    )

    width, height = image_size
    image = Image.new("RGB", (width, height), color="purple")
    prompt = make_query_prompt("Describe.")

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=20, temperature=0),
    )

    assert outputs[0].outputs[0].text is not None


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for TP=2"
)
@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
@large_gpu_mark(min_gb=48)
def test_tensor_parallel(model_id: str):
    """Test model with tensor parallelism = 2."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        tokenizer=MOONDREAM3_TOKENIZER,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=2,
        max_model_len=2048,
        enforce_eager=True,
    )

    image = Image.new("RGB", (378, 378), color="red")
    prompt = make_query_prompt("What is this?")

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=20, temperature=0),
    )

    assert len(outputs) > 0
    assert outputs[0].outputs[0].text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
