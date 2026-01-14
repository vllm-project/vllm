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

PROMPT_TEMPLATE = """<|eot_id|><|start_header_id|>user<|end_header_id|>
I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.
"
##Instruction:
candidate_poll contains tables such as candidate, people. Table candidate has columns such as Candidate_ID, People_ID, Poll_Source, Date, Support_rate, Consider_rate, Oppose_rate, Unsure_rate. Candidate_ID is the primary key.
Table people has columns such as People_ID, Sex, Name, Date_of_Birth, Height, Weight. People_ID is the primary key.
The People_ID of candidate is the foreign key of People_ID of people.
###Input:
{context}
###Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM candidate",
    "SELECT count(*) FROM candidate",
    "SELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
    "SELECT poll_source FROM candidate GROUP BY poll_source ORDER BY count(*) DESC LIMIT 1",  # noqa: E501
]

MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"


def do_sample(
    llm: vllm.LLM,
    lora_path: str,
    lora_id: int,
    tensorizer_config_dict: dict | None = None,
) -> list[str]:
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

    sampling_params = vllm.SamplingParams(
        temperature=0, max_tokens=64, stop=["<|im_end|>"]
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
    lora_request = LoRARequest(str(lora_id), lora_id, lora_path) if lora_id else None
    generated_texts: list[str] = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # The output should include  correct lora_request info
        if lora_request is not None:
            assert output.lora_request.lora_name == lora_request.lora_name
            assert output.lora_request.lora_int_id == lora_request.lora_int_id
            assert output.lora_request.lora_path == lora_request.lora_path
        else:
            assert output.lora_request is None
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


def generate_and_test(
    llm, llama32_lora_files, tensorizer_config_dict: dict | None = None
):
    print("lora adapter created")
    print("lora 1")
    assert (
        do_sample(
            llm,
            llama32_lora_files,
            tensorizer_config_dict=tensorizer_config_dict,
            lora_id=1,
        )
        == EXPECTED_LORA_OUTPUT
    )

    print("lora 2")
    assert (
        do_sample(
            llm,
            llama32_lora_files,
            tensorizer_config_dict=tensorizer_config_dict,
            lora_id=2,
        )
        == EXPECTED_LORA_OUTPUT
    )

    print("removing lora")


@create_new_process_for_each_test()
@pytest.mark.parametrize("cudagraph_specialize_lora", [True, False])
def test_llama_lora(llama32_lora_files, cudagraph_specialize_lora: bool):
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        # also test odd max_num_seqs
        max_num_seqs=7,
        max_model_len=1024,
        max_loras=4,
        compilation_config=vllm.config.CompilationConfig(
            cudagraph_specialize_lora=cudagraph_specialize_lora,
        ),
    )
    generate_and_test(llm, llama32_lora_files)


@multi_gpu_test(num_gpus=4)
def test_llama_lora_tp4(llama32_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=7,
        max_model_len=1024,
        max_loras=4,
        tensor_parallel_size=4,
    )
    generate_and_test(llm, llama32_lora_files)


@multi_gpu_test(num_gpus=4)
def test_llama_lora_tp4_fully_sharded_loras(llama32_lora_files):
    llm = vllm.LLM(
        MODEL_PATH,
        enable_lora=True,
        max_num_seqs=8,
        max_loras=4,
        max_model_len=1024,
        tensor_parallel_size=4,
        fully_sharded_loras=True,
    )
    generate_and_test(llm, llama32_lora_files)


@multi_gpu_test(num_gpus=2)
def test_tp2_serialize_and_deserialize_lora(
    tmp_path,
    llama32_lora_files,
):
    # Run the tensorizing of the LoRA adapter and the model in a subprocess
    # to guarantee cleanup

    tp_size = 2
    model_name = "model-rank-%03d.tensors"

    model_ref = MODEL_PATH
    lora_path = llama32_lora_files
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
        load_format="tensorizer",
        enable_lora=True,
        enforce_eager=True,
        model_loader_extra_config=tensorizer_config,
        max_num_seqs=7,
        max_model_len=1024,
        tensor_parallel_size=2,
        max_loras=2,
    )

    tc_as_dict = tensorizer_config.to_serializable()

    print("lora adapter created")
    print("lora 1")
    assert (
        do_sample(
            loaded_llm, llama32_lora_files, tensorizer_config_dict=tc_as_dict, lora_id=1
        )
        == EXPECTED_LORA_OUTPUT
    )
