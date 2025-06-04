# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def test_llama_single_lora():
    sql_lora_files = snapshot_download(
        repo_id="yard1/llama-2-7b-sql-lora-test")
    llm = LLM(model="meta-llama/Llama-2-7b-hf",
              tensor_parallel_size=2,
              max_num_seqs=4,
              max_model_len=512,
              use_v2_block_manager=True,
              override_neuron_config={
                  "sequence_parallel_enabled": False,
                  "skip_warmup": True,
                  "lora_modules": [{
                      "name": "lora_id_1",
                      "path": sql_lora_files
                  }]
              },
              enable_lora=True,
              max_loras=1,
              max_lora_rank=256,
              device="neuron")
    """For multi-lora requests using NxDI as the backend, only the lora_name 
    needs to be specified. The lora_id and lora_path are supplied at the LLM 
    class/server initialization, after which the paths are handled by NxDI"""
    lora_req_1 = LoRARequest("lora_id_1", 0, " ")
    prompts = [
        "The president of the United States is",
        "The capital of France is",
    ]
    outputs = llm.generate(prompts,
                           SamplingParams(top_k=1),
                           lora_request=[lora_req_1, lora_req_1])

    expected_outputs = [
        " the head of state and head of government of the United States. "
        "The president direct",
        " a city of contrasts. The city is home to the Eiffel Tower"
    ]

    for expected_output, output in zip(expected_outputs, outputs):
        generated_text = output.outputs[0].text
        assert (expected_output == generated_text)


def test_llama_multiple_lora():
    sql_lora_files = snapshot_download(
        repo_id="yard1/llama-2-7b-sql-lora-test")
    llm = LLM(model="meta-llama/Llama-2-7b-hf",
              tensor_parallel_size=2,
              max_num_seqs=4,
              max_model_len=512,
              use_v2_block_manager=True,
              override_neuron_config={
                  "sequence_parallel_enabled":
                  False,
                  "skip_warmup":
                  True,
                  "lora_modules": [{
                      "name": "lora_id_1",
                      "path": sql_lora_files
                  }, {
                      "name": "lora_id_2",
                      "path": sql_lora_files
                  }]
              },
              enable_lora=True,
              max_loras=2,
              max_lora_rank=256,
              device="neuron")
    """For multi-lora requests using NxDI as the backend, only the lora_name 
    needs to be specified. The lora_id and lora_path are supplied at the LLM 
    class/server initialization, after which the paths are handled by NxDI"""
    lora_req_1 = LoRARequest("lora_id_1", 0, " ")
    lora_req_2 = LoRARequest("lora_id_2", 1, " ")
    prompts = [
        "The president of the United States is",
        "The capital of France is",
    ]
    outputs = llm.generate(prompts,
                           SamplingParams(top_k=1),
                           lora_request=[lora_req_1, lora_req_2])

    expected_outputs = [
        " the head of state and head of government of the United States. "
        "The president direct",
        " a city of contrasts. The city is home to the Eiffel Tower"
    ]

    for expected_output, output in zip(expected_outputs, outputs):
        generated_text = output.outputs[0].text
        assert (expected_output == generated_text)
