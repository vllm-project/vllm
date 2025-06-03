# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from vllm import LLM, SamplingParams

prompts = [
    "A robot may not injure a human being",
    "It is only with the heart that one can see rightly;",
    "The greatest glory in living lies not in never falling,",
]
answers = [
    " or, through inaction, allow a human being to come to harm.",
    " what is essential is invisible to the eye.",
    " but in rising every time we fall.",
]
N = 1
# Currently, top-p sampling is disabled. `top_p` should be 1.0.
sampling_params = SamplingParams(temperature=0, top_p=1.0, n=N, max_tokens=16)


def main():
    parser = argparse.ArgumentParser(description="TPU offline inference example")
    parser.add_argument("--use-spmd", action="store_true", help="Enable SPMD mode")
    args = parser.parse_args()

    llm_args = {
        "model": "Qwen/Qwen2-1.5B-Instruct",
        "max_num_batched_tokens": 64,
        "max_num_seqs": 4,
        "max_model_len": 128,
    }
    if args.use_spmd:
        os.environ["VLLM_XLA_USE_SPMD"] = "1"
        # Can only hardcode the number of chips for now.
        # calling xr.global_runtime_device_count() beforeing init SPMD env in
        # torch_xla will mess up the distributed env.
        llm_args["tensor_parallel_size"] = 8
        # Use Llama, for num_kv_heads = 8.
        llm_args["model"] = "meta-llama/Llama-3.1-8B-Instruct"

    # Set `enforce_eager=True` to avoid ahead-of-time compilation.
    # In real workloads, `enforace_eager` should be `False`.
    llm = LLM(**llm_args)
    outputs = llm.generate(prompts, sampling_params)
    print("-" * 50)
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        assert generated_text.startswith(answer)
        print("-" * 50)


if __name__ == "__main__":
    main()
