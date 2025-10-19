# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

from ..utils import multi_gpu_test

MODEL_PATH = "openbmb/MiniCPM-Llama3-V-2_5"

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "(<image>./</image>)\nWhat is in the image?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

IMAGE_ASSETS = [
    ImageAsset("stop_sign"),
]

# After fine-tuning with LoRA, all generated content should start begin `A`.
EXPECTED_OUTPUT = [
    "A red and white stop sign with a Chinese archway in the background featuring red lanterns and gold accents.",  # noqa: E501
]


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=5,
        stop_token_ids=[128001, 128009],  # eos_id, eot_id
    )

    inputs = [
        {
            "prompt": PROMPT_TEMPLATE,
            "multi_modal_data": {"image": asset.pil_image},
        }
        for asset in IMAGE_ASSETS
    ]

    outputs = llm.generate(
        inputs,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None,
    )
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.xfail(
    current_platform.is_rocm(),
    reason="MiniCPM-V dependency xformers incompatible with ROCm",
)
def test_minicpmv_lora(minicpmv_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_num_seqs=2,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        enforce_eager=True,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 2, "video": 0},
        trust_remote_code=True,
    )
    output1 = do_sample(llm, minicpmv_lora_files, lora_id=1)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output1[i])
    output2 = do_sample(llm, minicpmv_lora_files, lora_id=2)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output2[i])


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
@pytest.mark.xfail(
    current_platform.is_rocm(),
    reason="MiniCPM-V dependency xformers incompatible with ROCm",
)
@multi_gpu_test(num_gpus=4)
def test_minicpmv_tp4_wo_fully_sharded_loras(minicpmv_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=2,
        max_loras=4,
        max_lora_rank=64,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": 2, "video": 0},
        trust_remote_code=True,
    )
    output_tp = do_sample(llm, minicpmv_lora_files, lora_id=1)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output_tp[i])


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
@pytest.mark.xfail(
    current_platform.is_rocm(),
    reason="MiniCPM-V dependency xformers incompatible with ROCm",
)
@multi_gpu_test(num_gpus=4)
def test_minicpmv_tp4_fully_sharded_loras(minicpmv_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=2,
        max_loras=2,
        max_lora_rank=8,
        tensor_parallel_size=4,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        fully_sharded_loras=True,
    )
    output_tp = do_sample(llm, minicpmv_lora_files, lora_id=1)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output_tp[i])
    output_tp = do_sample(llm, minicpmv_lora_files, lora_id=2)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output_tp[i])
