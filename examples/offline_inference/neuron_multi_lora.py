# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.lora.request import LoRARequest


def run_vllm():
    llm = LLM(
        model="meta-llama/Llama-3.1-8B",
        tensor_parallel_size=32,
        max_num_seqs=4,
        max_model_len=512,
        use_v2_block_manager=True,
        override_neuron_config={
            "sequence_parallel_enabled": False,
        },
        lora_modules=[
            LoRAModulePath(name="lora_id_1",
                           path="mkopecki/chess-lora-adapter-llama-3.1-8b"),
            LoRAModulePath(name="lora_id_2",
                           path="mkopecki/chess-lora-adapter-llama-3.1-8b")
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
