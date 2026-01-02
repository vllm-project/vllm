# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.lora.request import LoRARequest

BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
PROMPT = "Question: <image> What is shown in the photo? Answer:"
EXPECTED_OUTPUT = "a stop sign"


def test_blip2_single_lora_vqa(blip2_vision_lora_files):
    llm = LLM(
        model=BLIP2_MODEL,
        enable_lora=True,
        enable_tower_connector_lora=True,
        max_loras=1,
        max_lora_rank=64,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        mm_processor_cache_gb=0,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0)
    image_asset = ImageAsset("stop_sign")
    inputs = [
        {
            "prompt": PROMPT,
            "multi_modal_data": {"image": image_asset.pil_image},
        }
    ]

    lora_request = LoRARequest("pokemon_lora", 1, blip2_vision_lora_files)
    outputs = llm.generate(inputs, sampling_params, lora_request=lora_request)

    assert len(outputs) == 1
    generated_text = outputs[0].outputs[0].text.strip().lower()
    assert generated_text == EXPECTED_OUTPUT
