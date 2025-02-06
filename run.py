from typing import List
from safetensors.torch import load_file

import pytest
import json

import torch
import vllm
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
import subprocess as sp
import time

MODEL_PATH = "/models/llama3.2-1b"
LORA_PATH = "/models/llama3.2-1b-lora"


def do_sample(llm: vllm.LLM, lora_id: int, weights: dict[str, torch.Tensor], config: dict) -> list[str]:
    prompts = [
        "Quote: Imagination is",
        "Quote: Be yourself;",
        "Quote: Painting is poetry that is seen rather than felt,",
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path=LORA_PATH) #lora_tensors=weights, lora_config=config)
        if lora_id else None)

    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


if __name__ == "__main__":
    weights = load_file(LORA_PATH + "/adapter_model.safetensors")
    with open(LORA_PATH + "/adapter_config.json") as f:
        config = json.load(f)

    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   tensor_parallel_size=4)

    start_time = time.time()
    output1 = do_sample(llm, lora_id=1, weights=weights, config=config)
    print("--- %s seconds ---" % (time.time() - start_time))