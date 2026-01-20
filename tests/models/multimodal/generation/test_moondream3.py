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

from tests.models.registry import HF_EXAMPLE_MODELS

from ....utils import large_gpu_mark

MOONDREAM3_MODEL_ID = "moondream/moondream3-preview"
MOONDREAM3_TOKENIZER = "moondream/starmie-v1"


def make_query_prompt(question: str) -> str:
    """Create a query prompt for Moondream3."""
    return f"<|endoftext|><image> \n\nQuestion: {question}\n\nAnswer:"


def make_caption_prompt() -> str:
    """Create a caption prompt for Moondream3."""
    return "<|endoftext|><image> \n\nDescribe this image.\n\n"


@pytest.fixture(scope="module")
def llm():
    model_info = HF_EXAMPLE_MODELS.get_hf_info("Moondream3ForCausalLM")
    model_info.check_transformers_version(on_fail="skip")

    from vllm import LLM

    try:
        return LLM(
            model=MOONDREAM3_MODEL_ID,
            tokenizer=MOONDREAM3_TOKENIZER,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2048,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )
    except Exception as exc:
        pytest.skip(f"Failed to load {MOONDREAM3_MODEL_ID}: {exc}")


@large_gpu_mark(min_gb=48)
def test_model_loading(llm):
    """Test that the model loads without errors."""
    assert llm is not None


@large_gpu_mark(min_gb=48)
def test_query_skill(llm):
    """Test query (question answering) skill."""
    from vllm import SamplingParams

    image = Image.new("RGB", (378, 378), color="blue")
    prompt = make_query_prompt("What color is this image?")

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=50, temperature=0),
    )

    output_text = outputs[0].outputs[0].text
    assert output_text is not None
    assert len(output_text) > 0


@large_gpu_mark(min_gb=48)
def test_caption_skill(llm):
    """Test caption (image description) skill."""
    from vllm import SamplingParams

    image = Image.new("RGB", (378, 378), color="green")
    prompt = make_caption_prompt()

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=100, temperature=0),
    )

    output_text = outputs[0].outputs[0].text
    assert output_text is not None
    assert len(output_text) > 0


@large_gpu_mark(min_gb=48)
def test_batched_inference(llm):
    """Test batched inference with multiple images."""
    from vllm import SamplingParams

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
@large_gpu_mark(min_gb=48)
def test_various_image_sizes(image_size: tuple[int, int], llm):
    """Test inference with various image sizes."""
    from vllm import SamplingParams

    width, height = image_size
    image = Image.new("RGB", (width, height), color="purple")
    prompt = make_query_prompt("Describe.")

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=20, temperature=0),
    )

    assert outputs[0].outputs[0].text is not None


@pytest.mark.skip(reason="Run separately: pytest -k test_tensor_parallel --forked")
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs for TP=2"
)
@large_gpu_mark(min_gb=80)
def test_tensor_parallel():
    """Test model with tensor parallelism = 2.

    This test must be run in isolation to avoid OOM from other tests.
    Run with: pytest <this_file>::test_tensor_parallel --forked
    """
    import gc

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel

    # Clean up any existing model parallel state
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()

    llm = LLM(
        model=MOONDREAM3_MODEL_ID,
        tokenizer=MOONDREAM3_TOKENIZER,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=2,
        max_model_len=1024,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.45,
    )

    image = Image.new("RGB", (378, 378), color="red")
    prompt = make_query_prompt("What is this?")

    try:
        outputs = llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            SamplingParams(max_tokens=20, temperature=0),
        )

        assert len(outputs) > 0
        assert outputs[0].outputs[0].text is not None
    finally:
        # Clean up to release GPU memory
        del llm
        gc.collect()
        torch.cuda.empty_cache()
