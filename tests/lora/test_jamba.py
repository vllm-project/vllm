from typing import List

import pytest
import torch

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "ai21labs/AI21-Jamba-1.5-Mini"

MAX_TOKENS = 40


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int,
              prompts: List[str]) -> List[str]:

    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
    generated_texts: List[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@pytest.mark.parametrize("tp_size", [4])
def test_jamba_lora(jamba_lora_files, tp_size):
    """Original test, the LoRA model has the common target modules, not all"""
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    prompts = ["Write a story about a sheep and a goat."]

    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        distributed_executor_backend="ray",
        tensor_parallel_size=tp_size,
    )

    expected_jamba_output = [
        """Once upon a time, in a lush green meadow, there lived a sheep named Clara and a goat named Billy. Clara was a gentle creature, always nibbling on the soft grass and humming"""  # noqa: E501
    ]
    assert do_sample(llm, jamba_lora_files, lora_id=1,
                     prompts=prompts) == expected_jamba_output
