# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

from ..models.utils import check_logprobs_close
from ..utils import create_new_process_for_each_test, multi_gpu_test

MODEL_PATH = "vllm-project/Ilama-3.2-1B"

PROMPT_TEMPLATE = """I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n"\n##Instruction:\nconcert_singer contains tables such as stadium, singer, concert, singer_in_concert. Table stadium has columns such as Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average. Stadium_ID is the primary key.\nTable singer has columns such as Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male. Singer_ID is the primary key.\nTable concert has columns such as concert_ID, concert_Name, Theme, Stadium_ID, Year. concert_ID is the primary key.\nTable singer_in_concert has columns such as concert_ID, Singer_ID. concert_ID is the primary key.\nThe Stadium_ID of concert is the foreign key of Stadium_ID of stadium.\nThe Singer_ID of singer_in_concert is the foreign key of Singer_ID of singer.\nThe concert_ID of singer_in_concert is the foreign key of concert_ID of concert.\n\n###Input:\n{query}\n\n###Response:"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM singer",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",  # noqa: E501
    "SELECT DISTINCT Country FROM singer WHERE Age  >  20",
]


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    prompts = [
        PROMPT_TEMPLATE.format(query="How many singers do we have?"),
        PROMPT_TEMPLATE.format(
            query="What is the average, minimum, and maximum age of all singers from France?"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(
            query="What are all distinct countries where singers above age 20 are from?"  # noqa: E501
        ),
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
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


def test_ilama_lora(ilama_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=16,
        trust_remote_code=True,
        enable_chunked_prefill=True,
    )

    output1 = do_sample(llm, ilama_lora_files, lora_id=1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, ilama_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
@multi_gpu_test(num_gpus=4)
@create_new_process_for_each_test()
def test_ilama_lora_tp4(ilama_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=16,
        tensor_parallel_size=4,
        trust_remote_code=True,
        fully_sharded_loras=False,
        enable_chunked_prefill=True,
    )

    output1 = do_sample(llm, ilama_lora_files, lora_id=1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, ilama_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]


@pytest.mark.skipif(
    current_platform.is_cuda_alike(), reason="Skipping to avoid redundant model tests"
)
@multi_gpu_test(num_gpus=4)
@create_new_process_for_each_test()
def test_ilama_lora_tp4_fully_sharded_loras(ilama_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=16,
        tensor_parallel_size=4,
        trust_remote_code=True,
        fully_sharded_loras=True,
        enable_chunked_prefill=True,
    )
    output1 = do_sample(llm, ilama_lora_files, lora_id=1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, ilama_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]


@create_new_process_for_each_test()
def test_lora_on_input_embeddings_not_named_embed_tokens(
    tmp_path, hf_runner, vllm_runner
):
    """GPT-NeoX's input embedding is `embed_in`, not `embed_tokens`."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "EleutherAI/pythia-70m"
    lora_path = str(tmp_path / "lora")
    merged_path = str(tmp_path / "merged")
    lora_config = LoraConfig(r=8, target_modules=["embed_in"], init_lora_weights=False)
    peft_model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(model_path), lora_config
    )
    peft_model.save_pretrained(lora_path)
    peft_model.merge_and_unload().save_pretrained(merged_path)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(merged_path)

    prompts = ["The capital of France is", "def fibonacci(n):"]
    with hf_runner(merged_path, dtype="half") as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(prompts, 16, 5)
    with vllm_runner(
        model_path,
        dtype="half",
        enable_lora=True,
        max_lora_rank=8,
        max_model_len=256,
        gpu_memory_utilization=0.3,
    ) as vllm_model:
        lora_request = LoRARequest("embed_in", 1, lora_path)
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            prompts, 16, 5, lora_request=lora_request
        )
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
