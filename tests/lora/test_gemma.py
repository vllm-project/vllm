from typing import List

import pytest

import vllm
from vllm.lora.request import LoRARequest
from vllm.utils import is_hip

MODEL_PATH = "google/gemma-7b"


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
    prompts = [
        "Quote: Imagination is",
        "Quote: Be yourself;",
        "Quote: Painting is poetry that is seen rather than felt,",
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
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


@pytest.mark.xfail(is_hip(), reason="There can be output mismatch on ROCm")
def test_gemma_lora(gemma_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4)

    expected_lora_output = [
        "more important than knowledge.\nAuthor: Albert Einstein\n",
        "everyone else is already taken.\nAuthor: Oscar Wilde\n",
        "and poetry is painting that is felt rather than seen.\n"
        "Author: Leonardo da Vinci\n",
    ]

    output1 = do_sample(llm, gemma_lora_files, lora_id=1)
    for i in range(len(expected_lora_output)):
        assert output1[i].startswith(expected_lora_output[i])
    output2 = do_sample(llm, gemma_lora_files, lora_id=2)
    for i in range(len(expected_lora_output)):
        assert output2[i].startswith(expected_lora_output[i])
