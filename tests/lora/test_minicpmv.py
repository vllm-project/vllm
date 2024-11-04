from typing import List

import pytest

import vllm
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

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


@pytest.mark.xfail(
    current_platform.is_rocm(),
    reason="MiniCPM-V dependency xformers incompatible with ROCm")
def test_minicpmv_lora(minicpmv_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_num_seqs=2,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=64,
        trust_remote_code=True,
        gpu_memory_utilization=0.97  # This model is pretty big for CI gpus
    )
    output1 = do_sample(llm, minicpmv_lora_files, lora_id=1)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output1[i])
    output2 = do_sample(llm, minicpmv_lora_files, lora_id=2)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output2[i])
