# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "openai/gpt-oss-20b"

PROMPT_TEMPLATE = "<｜begin▁of▁sentence｜>You are a helpful assistant.\n\nUser: {context}\n\nAssistant:"  # noqa: E501


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    prompts = [
        PROMPT_TEMPLATE.format(context="Who are you?"),
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=64)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None,
    )
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


# FIXME: Load gpt-oss adapter
def test_gptoss20b_lora(gptoss20b_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_loras=1,
        trust_remote_code=True,
    )

    expected_lora_output = [
        "I am an AI language model developed by OpenAI. "
        "I am here to help you with any questions or "
        "tasks you may have."
    ]

    output1 = do_sample(llm, gptoss20b_lora_files, lora_id=1)
    print(output1)
    for i in range(len(expected_lora_output)):
        assert output1[i].startswith(expected_lora_output[i])
