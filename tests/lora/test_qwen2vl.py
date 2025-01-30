from typing import List

import pytest

import vllm
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

MODEL_PATH = "Qwen/Qwen2-VL-2B-Instruct"

PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n")

IMAGE_ASSETS = [
    ImageAsset("stop_sign"),
    ImageAsset("cherry_blossom"),
]

# After fine-tuning with LoRA, all generated content should start begin `A`.
EXPECTED_OUTPUT = [
    "A red stop sign stands prominently in the foreground, with a traditional Chinese gate and a black SUV in the background, illustrating a blend of modern and cultural elements.",  # noqa: E501
    "A majestic skyscraper stands tall, partially obscured by a vibrant canopy of cherry blossoms, against a clear blue sky.",  # noqa: E501
]


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=5,
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
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.xfail(
    current_platform.is_rocm(),
    reason="Qwen2-VL dependency xformers incompatible with ROCm")
def test_qwen2vl_lora(qwen2vl_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_num_seqs=2,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=16,
        trust_remote_code=True,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        max_model_len=4096,
    )
    output1 = do_sample(llm, qwen2vl_lora_files, lora_id=1)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output1[i])

    output2 = do_sample(llm, qwen2vl_lora_files, lora_id=2)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output2[i])
