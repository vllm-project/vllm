# SPDX-License-Identifier: Apache-2.0

import pytest

import vllm
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

MODEL_PATH = "google/gemma-7b"


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
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
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


# The V1 lora test for this model requires more than 24GB.
@pytest.mark.skip_v1
@pytest.mark.xfail(current_platform.is_rocm(),
                   reason="There can be output mismatch on ROCm")
def test_gemma_lora(gemma_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4,
                   enable_chunked_prefill=True)

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
