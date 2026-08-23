# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation tests for Moondream3 query and caption support."""

import pytest

from tests.models.registry import HF_EXAMPLE_MODELS
from vllm.platforms import current_platform

from ....conftest import IMAGE_ASSETS, ImageTestAssets
from ....utils import large_gpu_mark, multi_gpu_test

MOONDREAM3_MODEL_ID = "moondream/moondream3-preview"
MOONDREAM3_TOKENIZER = "moondream/starmie-v1"

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": "<|endoftext|><image><|md_reserved_0|>query<|md_reserved_1|>What color is the stop sign?<|md_reserved_2|>",  # noqa: E501
        "cherry_blossom": "<|endoftext|><image><|md_reserved_0|>query<|md_reserved_1|>What color are the flowers?<|md_reserved_2|>",  # noqa: E501
    }
)


def make_query_prompt(question: str) -> str:
    """Create a direct-answer query prompt for Moondream3."""
    return (
        "<|endoftext|><image><|md_reserved_0|>query<|md_reserved_1|>"
        f"{question}<|md_reserved_2|>"
    )


def make_caption_prompt(length: str = "normal") -> str:
    """Create a caption prompt for Moondream3."""
    return (
        "<|endoftext|><image><|md_reserved_0|>"
        f"describe<|md_reserved_1|>{length}<|md_reserved_2|>"
    )


@multi_gpu_test(num_gpus=2)
@large_gpu_mark(min_gb=80)
def test_tensor_parallel(image_assets: ImageTestAssets):
    import gc

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel

    destroy_model_parallel()
    gc.collect()
    current_platform.empty_cache()

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

    image = image_assets[0].pil_image
    prompt = make_query_prompt("What color is the stop sign?")

    try:
        outputs = llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            SamplingParams(max_tokens=20, temperature=0),
        )

        assert len(outputs) > 0
        assert outputs[0].outputs[0].text is not None
    finally:
        del llm
        gc.collect()
        current_platform.empty_cache()


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
            gpu_memory_utilization=0.45,
        )
    except Exception as exc:
        pytest.skip(f"Failed to load {MOONDREAM3_MODEL_ID}: {exc}")


@large_gpu_mark(min_gb=48)
def test_model_loading(llm):
    assert llm is not None


@large_gpu_mark(min_gb=48)
def test_query_skill(llm, image_assets: ImageTestAssets):
    from vllm import SamplingParams

    image = image_assets[0].pil_image
    prompt = make_query_prompt("What color is the stop sign?")

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=50, temperature=0),
    )

    output_text = outputs[0].outputs[0].text
    assert output_text is not None
    assert len(output_text) > 0


@large_gpu_mark(min_gb=48)
def test_caption_skill(llm, image_assets: ImageTestAssets):
    from vllm import SamplingParams

    image = image_assets[1].pil_image
    prompt = make_caption_prompt()

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=100, temperature=0),
    )

    output_text = outputs[0].outputs[0].text
    assert output_text is not None
    assert len(output_text) > 0


@large_gpu_mark(min_gb=48)
def test_batched_inference(llm, image_assets: ImageTestAssets):
    from vllm import SamplingParams

    images = [asset.pil_image for asset in image_assets]
    prompts = [
        {"prompt": prompt, "multi_modal_data": {"image": img}}
        for img, prompt in zip(images, HF_IMAGE_PROMPTS)
    ]

    outputs = llm.generate(prompts, SamplingParams(max_tokens=50, temperature=0))

    assert len(outputs) == len(images)
    for output in outputs:
        assert output.outputs[0].text is not None
        assert len(output.outputs[0].text) > 0


@pytest.mark.parametrize("asset_name", ["stop_sign", "cherry_blossom"])
@large_gpu_mark(min_gb=48)
def test_image_assets(llm, image_assets: ImageTestAssets, asset_name: str):
    from vllm import SamplingParams

    asset_idx = 0 if asset_name == "stop_sign" else 1
    image = image_assets[asset_idx].pil_image
    prompt = HF_IMAGE_PROMPTS[asset_idx]

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": image}},
        SamplingParams(max_tokens=50, temperature=0),
    )

    output_text = outputs[0].outputs[0].text
    assert output_text is not None
    assert len(output_text) > 0
