from typing import List

import pytest
import torch

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
    prompts = [
        "Question: What happens if you touch the eyes of a blind man?\nChoices:\nB: The man's sight will be restored.\nA: Nothing happens if you touch the eyes of a blind man.\nC: The man will get his eyesight back.\nD: The man will no longer be blind.\nAnswer:",  # noqa: E501
        "Question: Which will increase the energy of motion of water molecules?\nChoices:\nA: add heat\nB: add ice\nC: remove some of the water\nD: place the water in the freezer\nAnswer:",  # noqa: E501
        "Since Craig threw aluminum cans in the trash and Benjamin recycled, _ was environmentally irresponsible.\nChoices:\n1: Craig\n2: Benjamin\nAnswer:",  # noqa: E501
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=256)
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
def test_mixtral_lora(mixtral_lora_files, tp_size):
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")

    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        distributed_executor_backend="ray",
        tensor_parallel_size=tp_size,
        max_lora_rank=32,
    )

    expected_lora_output = [
        "A: Nothing happens if you touch the eyes of a blind man.",
        "A: add heat",
        "1: Craig",
    ]

    assert do_sample(llm, mixtral_lora_files,
                     lora_id=1) == expected_lora_output
    assert do_sample(llm, mixtral_lora_files,
                     lora_id=2) == expected_lora_output
