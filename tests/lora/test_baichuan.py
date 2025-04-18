# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest

import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest

MODEL_PATH = "baichuan-inc/Baichuan-7B"

PROMPT_TEMPLATE = """I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n"\n##Instruction:\nconcert_singer contains tables such as stadium, singer, concert, singer_in_concert. Table stadium has columns such as Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average. Stadium_ID is the primary key.\nTable singer has columns such as Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male. Singer_ID is the primary key.\nTable concert has columns such as concert_ID, concert_Name, Theme, Stadium_ID, Year. concert_ID is the primary key.\nTable singer_in_concert has columns such as concert_ID, Singer_ID. concert_ID is the primary key.\nThe Stadium_ID of concert is the foreign key of Stadium_ID of stadium.\nThe Singer_ID of singer_in_concert is the foreign key of Singer_ID of singer.\nThe concert_ID of singer_in_concert is the foreign key of concert_ID of concert.\n\n###Input:\n{query}\n\n###Response:"""  # noqa: E501


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
    prompts = [
        PROMPT_TEMPLATE.format(query="How many singers do we have?"),
        PROMPT_TEMPLATE.format(
            query=
            "What is the average, minimum, and maximum age of all singers from France?"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(
            query=
            "Show name, country, age for all singers ordered by age from the oldest to the youngest."  # noqa: E501
        ),
    ]
    print(prompts)
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


@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


def test_baichuan_lora(baichuan_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4,
                   max_lora_rank=64,
                   trust_remote_code=True)

    expected_lora_output = [
        "SELECT count(*) FROM singer",
        "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE Country  =  'France'",  # noqa: E501
        "SELECT name ,  country ,  age FROM singer ORDER BY age ASC",
    ]

    output1 = do_sample(llm, baichuan_lora_files, lora_id=1)
    for i in range(len(expected_lora_output)):
        assert output1[i] == expected_lora_output[i]
    output2 = do_sample(llm, baichuan_lora_files, lora_id=2)
    for i in range(len(expected_lora_output)):
        assert output2[i] == expected_lora_output[i]


@pytest.mark.parametrize("fully_sharded", [True, False])
def test_baichuan_tensor_parallel_equality(baichuan_lora_files,
                                           num_gpus_available, fully_sharded):
    if num_gpus_available < 4:
        pytest.skip(f"Not enough GPUs for tensor parallelism {4}")

    llm_tp1 = vllm.LLM(MODEL_PATH,
                       enable_lora=True,
                       max_num_seqs=16,
                       max_loras=4,
                       max_lora_rank=64,
                       tensor_parallel_size=1,
                       trust_remote_code=True,
                       fully_sharded_loras=fully_sharded)
    output_tp1 = do_sample(llm_tp1, baichuan_lora_files, lora_id=1)

    del llm_tp1
    cleanup_dist_env_and_memory()

    llm_tp2 = vllm.LLM(MODEL_PATH,
                       enable_lora=True,
                       max_num_seqs=16,
                       max_loras=4,
                       max_lora_rank=64,
                       tensor_parallel_size=2,
                       trust_remote_code=True,
                       fully_sharded_loras=fully_sharded)
    output_tp2 = do_sample(llm_tp2, baichuan_lora_files, lora_id=2)

    del llm_tp2
    cleanup_dist_env_and_memory()

    assert output_tp1 == output_tp2

    llm_tp4 = vllm.LLM(MODEL_PATH,
                       enable_lora=True,
                       max_num_seqs=16,
                       max_loras=4,
                       max_lora_rank=64,
                       tensor_parallel_size=4,
                       trust_remote_code=True,
                       fully_sharded_loras=fully_sharded)
    output_tp4 = do_sample(llm_tp4, baichuan_lora_files, lora_id=2)

    del llm_tp4
    cleanup_dist_env_and_memory()

    assert output_tp1 == output_tp4
