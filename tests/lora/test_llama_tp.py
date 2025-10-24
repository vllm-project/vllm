# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys

import pytest

import vllm
import vllm.config
from vllm import LLM
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

from ..utils import VLLM_PATH, create_new_process_for_each_test, multi_gpu_test

MODEL_PATH = "meta-llama/Llama-2-7b-hf"

EXPECTED_LORA_OUTPUT = [
    "  SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' ",  # noqa: E501
    "  SELECT nationality FROM table_name_11 WHERE elector = 'anchero pantaleone' ",
    "  SELECT one_mora FROM table_name_95 WHERE gloss = 'low tone mora with a gloss of /˩okiru/' [òkìɽɯ́] AND accented_mora = 'low tone mora with a gloss of /˩okiru/' [òkìɽɯ́] ",  # noqa: E501
    "  SELECT sex FROM people WHERE people_id IN (SELECT people_id FROM candidate GROUP BY sex ORDER BY COUNT(people_id) DESC LIMIT 1) ",  # noqa: E501
    "  SELECT pick FROM table_name_60 WHERE former_wnba_team = 'Minnesota Lynx' ",
    "  SELECT womens_doubles FROM table_28138035_4 WHERE mens_singles = 'Werner Schlager' ",  # noqa: E501
]


def do_sample(
    llm: vllm.LLM,
    lora_path: str,
    lora_id: int,
    tensorizer_config_dict: dict | None = None,
) -> list[str]:
    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_95 (one_mora VARCHAR, gloss VARCHAR, accented_mora VARCHAR)\n\n question: What is the one mora for a low tone mora with a gloss of /˩okiru/ [òkìɽɯ́]? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE candidate (people_id VARCHAR, unsure_rate INTEGER); CREATE TABLE people (sex VARCHAR, people_id VARCHAR)\n\n question: which gender got the highest average uncertain ratio. [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_60 (pick INTEGER, former_wnba_team VARCHAR)\n\n question: What pick was a player that previously played for the Minnesota Lynx? [/user] [assistant]",  # noqa: E501
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_28138035_4 (womens_doubles VARCHAR, mens_singles VARCHAR)\n\n question: Name the women's doubles for werner schlager [/user] [assistant]",  # noqa: E501
    ]

    sampling_params = vllm.SamplingParams(
        temperature=0, max_tokens=256, skip_special_tokens=False, stop=["[/assistant]"]
    )

    if tensorizer_config_dict is not None:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest(
                str(lora_id),
                lora_id,
                lora_path,
                tensorizer_config_dict=tensorizer_config_dict,
            )
            if lora_id
            else None,
        )
    else:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
            if lora_id
            else None,
        )
    # Print the outputs.
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def generate_and_test(llm, sql_lora_files, tensorizer_config_dict: dict | None = None):
    print("lora adapter created")
    print("lora 1")
    assert (
        do_sample(
            llm,
            sql_lora_files,
            tensorizer_config_dict=tensorizer_config_dict,
            lora_id=1,
        )
        == EXPECTED_LORA_OUTPUT
    )

    print("lora 2")
    assert (
        do_sample(
            llm,
            sql_lora_files,
            tensorizer_config_dict=tensorizer_config_dict,
            lora_id=2,
        )
        == EXPECTED_LORA_OUTPUT
    )

    print("removing lora")


@create_new_process_for_each_test()
@pytest.mark.parametrize("cudagraph_specialize_lora", [True, False])
def test_llama_lora(sql_lora_files, cudagraph_specialize_lora: bool):
    llm = vllm.LLM(
        MODEL_PATH,
        tokenizer=sql_lora_files,
        enable_lora=True,
        # also test odd max_num_seqs
        max_num_seqs=13,
        max_loras=4,
        compilation_config=vllm.config.CompilationConfig(
            cudagraph_specialize_lora=cudagraph_specialize_lora,
        ),
    )
    generate_and_test(llm, sql_lora_files)


@multi_gpu_test(num_gpus=4)
def test_llama_lora_tp4(sql_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        tokenizer=sql_lora_files,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        tensor_parallel_size=4,
    )
    generate_and_test(llm, sql_lora_files)


@multi_gpu_test(num_gpus=4)
def test_llama_lora_tp4_fully_sharded_loras(sql_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        tokenizer=sql_lora_files,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        tensor_parallel_size=4,
        fully_sharded_loras=True,
    )
    generate_and_test(llm, sql_lora_files)


@multi_gpu_test(num_gpus=2)
def test_tp2_serialize_and_deserialize_lora(
    tmp_path, sql_lora_files, sql_lora_huggingface_id
):
    # Run the tensorizing of the LoRA adapter and the model in a subprocess
    # to guarantee cleanup

    tp_size = 2
    model_name = "model-rank-%03d.tensors"

    model_ref = MODEL_PATH
    lora_path = sql_lora_huggingface_id
    suffix = "test"
    try:
        result = subprocess.run(
            [
                sys.executable,
                f"{VLLM_PATH}/examples/others/tensorize_vllm_model.py",
                "--model",
                MODEL_PATH,
                "--lora-path",
                lora_path,
                "--tensor-parallel-size",
                str(tp_size),
                "serialize",
                "--serialized-directory",
                str(tmp_path),
                "--suffix",
                suffix,
                "--serialization-kwargs",
                '{"limit_cpu_concurrency": 4}',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Tensorizing failed.")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    print("STDOUT:\n", result.stdout)

    model_uri = tmp_path / "vllm" / model_ref / suffix / model_name
    tensorizer_config = TensorizerConfig(tensorizer_uri=str(model_uri))

    loaded_llm = LLM(
        model=model_ref,
        tokenizer=sql_lora_files,
        load_format="tensorizer",
        enable_lora=True,
        enforce_eager=True,
        model_loader_extra_config=tensorizer_config,
        max_num_seqs=13,
        tensor_parallel_size=2,
        max_loras=2,
    )

    tc_as_dict = tensorizer_config.to_serializable()

    print("lora adapter created")
    print("lora 1")
    assert (
        do_sample(
            loaded_llm, sql_lora_files, tensorizer_config_dict=tc_as_dict, lora_id=1
        )
        == EXPECTED_LORA_OUTPUT
    )
