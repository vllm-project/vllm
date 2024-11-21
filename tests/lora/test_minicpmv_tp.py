from typing import List

import pytest

import vllm
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest

from ..utils import multi_gpu_test

MODEL_PATH = "openbmb/MiniCPM-Llama3-V-2_5"

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "(<image>./</image>)\nWhat is in the image?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n")

IMAGE_ASSETS = [
    ImageAsset("stop_sign"),
    ImageAsset("cherry_blossom"),
]

# After fine-tuning with LoRA, all generated content should start begin `A`.
EXPECTED_OUTPUT = [
    "A red and white stop sign with a Chinese archway in the background featuring red lanterns and gold accents.",  # noqa: E501
    "A pink cherry blossom tree with a blue sky in the background.",
]


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=5,
        stop_token_ids=[128001, 128009],  # eos_id, eot_id
    )

    inputs = [{
        "prompt": PROMPT_TEMPLATE,
        "multi_modal_data": {
            "image": asset.pil_image
        },
    } for asset in IMAGE_ASSETS]

    outputs = llm.generate(
        inputs,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None,
    )
    # Print the outputs.
    generated_texts: List[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("fully_sharded", [True, False])
def test_minicpmv_tp2(minicpmv_lora_files, fully_sharded):
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=2,
        max_loras=4,
        max_lora_rank=64,
        tensor_parallel_size=2,
        trust_remote_code=True,
        fully_sharded_loras=fully_sharded,
    )

    output_tp = do_sample(llm, minicpmv_lora_files, lora_id=1)

    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output_tp[i])


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("fully_sharded", [True, False])
def test_minicpmv_tp4(minicpmv_lora_files, fully_sharded):
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=2,
        max_loras=4,
        max_lora_rank=64,
        tensor_parallel_size=4,
        trust_remote_code=True,
        fully_sharded_loras=fully_sharded,
    )
    output_tp = do_sample(llm, minicpmv_lora_files, lora_id=1)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output_tp[i])
