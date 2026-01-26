# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def read_prompts():
    """Read prompts from prefill_output.txt"""
    prompts = []
    try:
        with open("prefill_output.txt") as f:
            for line in f:
                prompts.append(line.strip())
        print(f"Loaded {len(prompts)} prompts from prefill_output.txt")
        return prompts
    except FileNotFoundError:
        print("Error: prefill_output.txt file not found")
        exit(-1)


def main():
    prompts = read_prompts()
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmented-prefill", action="store_true", help="Simulate gaps in KV cache"
    )

    args = parser.parse_args()

    if args.segmented_prefill:
        ktc = KVTransferConfig(
            kv_connector="SegmentedPrefillExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": "local_storage",
            },
            kv_connector_module_path="segmented_prefill_example_connector",
        )
        out_file = "segmented_prefill_decode_output.txt"
    else:
        ktc = KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "shared_storage_path": "local_storage",
            },
        )
        out_file = "decode_output.txt"

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,  # no CUDA graphs, so layer-wise API will be called
        gpu_memory_utilization=0.8,
        kv_transfer_config=ktc,
    )

    outputs = llm.generate(prompts, sampling_params)

    sep_str = "-" * 30 + "\n"
    with open(out_file, "w", encoding="utf-8") as f:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            out_str = f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n"
            print(out_str)
            print(sep_str)
            f.write(out_str)
            f.write(sep_str)


if __name__ == "__main__":
    main()
