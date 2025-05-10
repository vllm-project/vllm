# SPDX-License-Identifier: Apache-2.0

import os

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.lora.request import LoRARequest

os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
os.environ['NEURON_RT_DBG_RDH_CC'] = '0'
os.environ['NEURON_RT_INSPECT_ENABLE'] = '0'
os.environ['XLA_HANDLE_SPECIAL_SCALAR'] = '1'
os.environ['UNSAFE_FP8FNCAST'] = '1'


def run_vllm():
    llm = LLM(model="/home/ubuntu/models/llama-3.1-8b",
              tensor_parallel_size=32,
              max_num_seqs=4,
              max_model_len=512,
              use_v2_block_manager=True,
              override_neuron_config={
                  "sequence_parallel_enabled": False,
              },
              lora_modules=[
                  LoRAModulePath(name="lora_id_1",
                                 path="~/models/llama-3.1-8b-lora-adapter"),
                  LoRAModulePath(name="lora_id_2",
                                 path="~/models/llama-3.1-8b-lora-adapter")
              ],
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
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    return [output.prompt + output.outputs[0].text for output in outputs]


if __name__ == "__main__":
    vllm_outputs = run_vllm()
    for actual_seq in vllm_outputs:
        print(f"actual: {actual_seq}")
