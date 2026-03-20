# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import AutoTokenizer

import vllm
import vllm.config
from vllm.lora.request import LoRARequest

from ..utils import create_new_process_for_each_test, multi_gpu_test

MODEL_PATH = "Qwen/Qwen3.5-4B"

PROMPT_TEMPLATE = """Write a SQL query for the given database.\nSchema:\nTables:\n  - stadium(Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average)\n  - singer(Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male)\n  - concert(concert_ID, concert_Name, Theme, Stadium_ID, Year)\n  - singer_in_concert(concert_ID, Singer_ID)\n\nQuestion:\n{query}"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM singer",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",
    "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)",
]


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
    prompts = [
        PROMPT_TEMPLATE.format(query="How many singers do we have?"),
        PROMPT_TEMPLATE.format(
            query=(
                "What is the average, minimum, and maximum "
                "age of all singers from France?"
            )
        ),
        PROMPT_TEMPLATE.format(
            query=("What are the names of the stadiums without any concerts?")
        ),
    ]
    input_templates = []
    for prmpt in prompts:
        messages = [{"role": "user", "content": prmpt}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # disable thinking
        )
        input_templates.append(prompt)
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=512)
    outputs = llm.generate(
        input_templates,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None,
    )

    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


@create_new_process_for_each_test()
def test_qwen35_dense_model_lora(qwen35_dense_model_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=512,
        enable_lora=True,
        max_loras=2,
        max_num_seqs=16,
        max_lora_rank=8,
        trust_remote_code=True,
    )

    output1 = do_sample(llm, qwen35_dense_model_lora_files, lora_id=1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, qwen35_dense_model_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]


@multi_gpu_test(num_gpus=4)
def test_qwen35_dense_model_lora_tp4(qwen35_dense_model_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=1024,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        max_num_seqs=16,
        tensor_parallel_size=4,
        trust_remote_code=True,
        fully_sharded_loras=False,
        compilation_config=vllm.config.CompilationConfig(  # Avoid OOM
            cudagraph_specialize_lora=False,
        ),
    )

    output1 = do_sample(llm, qwen35_dense_model_lora_files, lora_id=1)
    print(output1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, qwen35_dense_model_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]


@multi_gpu_test(num_gpus=4)
def test_qwen35_dense_model_lora_tp4_fully_sharded_loras(qwen35_dense_model_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=512,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        tensor_parallel_size=4,
        trust_remote_code=True,
        fully_sharded_loras=True,
        gpu_memory_utilization=0.8,
        compilation_config=vllm.config.CompilationConfig(  # Avoid OOM
            cudagraph_specialize_lora=False,
        ),
    )
    output1 = do_sample(llm, qwen35_dense_model_lora_files, lora_id=1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, qwen35_dense_model_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]
