# SPDX-License-Identifier: Apache-2.0

import pytest

import vllm
from vllm.lora.request import LoRARequest

MODEL_PATH = "ibm-granite/granite-3b-code-base"


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE candidate (people_id VARCHAR, unsure_rate INTEGER); CREATE TABLE people (sex VARCHAR, people_id VARCHAR)\n\n question: which gender got the highest average uncertain ratio. [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_28138035_4 (womens_doubles VARCHAR, mens_singles VARCHAR)\n\n question: Name the women's doubles for werner schlager [/user] [assistant]"  # noqa: E501
    ]
    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=256,
                                          stop=["[/assistant]"])
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    generated_texts: list[str] = []
    for output in outputs:
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
    return generated_texts


@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


# Skipping for V1 for now as we are hitting,
# "Head size 80 is not supported by FlashAttention." error.
@pytest.mark.skip_v1
@pytest.mark.parametrize("lora_bias", [True])
@pytest.mark.parametrize("fully_sharded", [True, False])
def test_lora_bias(lora_bias_files: str, lora_bias: bool, fully_sharded: bool):
    llm = vllm.LLM(MODEL_PATH,
                   enable_lora=True,
                   max_num_seqs=16,
                   max_lora_rank=8,
                   max_loras=1,
                   enable_lora_bias=lora_bias,
                   tensor_parallel_size=1,
                   fully_sharded_loras=fully_sharded)

    print("lora adapter created")
    output1 = do_sample(llm, lora_bias_files, lora_id=0)

    print("lora")
    output2 = do_sample(llm, lora_bias_files, lora_id=1)

    if lora_bias:
        assert output1 != output2
    else:
        assert output1 == output2
