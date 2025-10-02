# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def read_prompts():
    """Read prompts from output.txt"""
    prompts = []
    try:
        with open("output.txt") as f:
            for line in f:
                prompts.append(line.strip())
        print(f"Loaded {len(prompts)} prompts from output.txt")
        return prompts
    except FileNotFoundError:
        print("Error: output.txt file not found")
        exit(-1)


def main():
    prompts = read_prompts()
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=64,
        max_num_seqs=16,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SharedStorageConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        ),
    )  # , max_model_len=2048, max_num_batched_tokens=2048)

    # 1ST generation (prefill instance)
    outputs = llm.generate(prompts, sampling_params)

    print("-" * 30)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 30)


if __name__ == "__main__":
    main()
