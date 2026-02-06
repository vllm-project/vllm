# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# NOTE To avoid overloading the CI pipeline, this test script will not
# be triggered on CI and is primarily intended for local testing and verification.

import vllm
from vllm.lora.request import LoRARequest

from ..utils import multi_gpu_test

MODEL_PATH = "Qwen/Qwen3-30B-A3B"

PROMPT_TEMPLATE = """<|im_start|>user
I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.
"
##Instruction:
candidate_poll contains tables such as candidate, people. Table candidate has columns such as Candidate_ID, People_ID, Poll_Source, Date, Support_rate, Consider_rate, Oppose_rate, Unsure_rate. Candidate_ID is the primary key.
Table people has columns such as People_ID, Sex, Name, Date_of_Birth, Height, Weight. People_ID is the primary key.
The People_ID of candidate is the foreign key of People_ID of people.


###Input:
{context}

###Response:<|im_end|>
<|im_start|>assistant"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "<think>\n\n</think>\n\nSELECT count(*) FROM candidate",
    "<think>\n\n</think>\n\nSELECT count(*) FROM candidate",
    "<think>\n\n</think>\n\nSELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
    "<think>\n\n</think>\n\nSELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
]


def generate_and_test(llm: vllm.LLM, lora_path: str, lora_id: int) -> None:
    prompts = [
        PROMPT_TEMPLATE.format(context="How many candidates are there?"),
        PROMPT_TEMPLATE.format(context="Count the number of candidates."),
        PROMPT_TEMPLATE.format(
            context="Which poll resource provided the most number of candidate information?"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(
            context="Return the poll resource associated with the most candidates."
        ),
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

    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert generated_texts[i].startswith(EXPECTED_LORA_OUTPUT[i])


def test_qwen3moe_lora(qwen3moe_lora_files):
    # We enable enforce_eager=True here to reduce VRAM usage for lora-test CI,
    # Otherwise, the lora-test will fail due to CUDA OOM.
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
    )

    generate_and_test(llm, qwen3moe_lora_files, lora_id=1)
    generate_and_test(llm, qwen3moe_lora_files, lora_id=2)


@multi_gpu_test(num_gpus=2)
def test_qwen3moe_lora_tp2(qwen3moe_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=2,
    )

    generate_and_test(llm, qwen3moe_lora_files, lora_id=1)
    generate_and_test(llm, qwen3moe_lora_files, lora_id=2)


@multi_gpu_test(num_gpus=4)
def test_qwen3moe_lora_tp4(qwen3moe_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        tensor_parallel_size=4,
    )

    generate_and_test(llm, qwen3moe_lora_files, lora_id=1)
    generate_and_test(llm, qwen3moe_lora_files, lora_id=2)
