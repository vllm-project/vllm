# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE To avoid overloading the CI pipeline, this test script will not
# be triggered on CI and is primarily intended for local testing and verification.

import vllm
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest

from ..utils import multi_gpu_test

MODEL_PATH = "google/gemma-4-E2B-it"

PROMPT_TEMPLATE = """<bos><|turn>user
<|image|>What is in the image?<turn|>
<|turn>model
"""

TEST_IMAGES = [
    ImageAsset("stop_sign"),
    ImageAsset("cherry_blossom"),
]

EXPECTED_OUTPUTS_VISION = [
    "A red stop sign stands prominently in the foreground.",
    "A majestic skyscraper stands tall, partially obscured by a vibrant "
    "canopy of cherry blossoms, against a clear blue sky.",
]


def generate_and_test(llm: vllm.LLM, lora_path: str, lora_id: int) -> None:
    prompts = [
        {
            "prompt": PROMPT_TEMPLATE,
            "multi_modal_data": {"image": asset.pil_image},
        }
        for asset in TEST_IMAGES
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=128)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path),
    )

    generated_texts = [output.outputs[0].text.strip() for output in outputs]
    for generated, expected in zip(generated_texts, EXPECTED_OUTPUTS_VISION):
        assert generated.startswith(expected), (
            f"Generated text {generated!r} does not match expected {expected!r}"
        )


def test_gemma4_lora(gemma4_vision_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        enforce_eager=True,
        max_loras=4,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
        enable_tower_connector_lora=True,
    )

    generate_and_test(llm, gemma4_vision_lora_files, lora_id=1)
    generate_and_test(llm, gemma4_vision_lora_files, lora_id=2)


@multi_gpu_test(num_gpus=2)
def test_gemma4_lora_tp2(gemma4_vision_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enforce_eager=True,
        enable_lora=True,
        max_loras=4,
        trust_remote_code=True,
        tensor_parallel_size=2,
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
        enable_tower_connector_lora=True,
    )

    generate_and_test(llm, gemma4_vision_lora_files, lora_id=1)
    generate_and_test(llm, gemma4_vision_lora_files, lora_id=2)


@multi_gpu_test(num_gpus=4)
def test_gemma4_lora_tp4(gemma4_vision_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enforce_eager=True,
        enable_lora=True,
        max_loras=4,
        trust_remote_code=True,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
        enable_tower_connector_lora=True,
    )

    generate_and_test(llm, gemma4_vision_lora_files, lora_id=1)
    generate_and_test(llm, gemma4_vision_lora_files, lora_id=2)
