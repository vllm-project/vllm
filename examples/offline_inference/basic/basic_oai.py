# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 1. Setup flash-attention-os-mini
# git clone git@github.com:vllm-project/flash-attention-os-mini.git
# cd flash-attention-os-mini
# git submodule update --init --recursive
# 2. Build vLLM
# cd vllm-oai
# export VLLM_FLASH_ATTN_SRC_DIR=/data/lily/flash-attention-os-mini
# pip install -e .

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "How are you?",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1, logprobs=10)


def main():
    # Create an LLM.
    llm = LLM(
        model="/data/xmo/os-mini/models/raw-weights",
        tokenizer="/data/xmo/os-mini/models/hf-converted",
        tensor_parallel_size=4,
        # Set these to make dummy run faster
        enforce_eager=True,
        max_num_batched_tokens=8,
        max_num_seqs=1,
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logprobs = output.outputs[0].logprobs
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        for logprob in logprobs[0].values():
            print(f" {logprob.decoded_token}: {logprob.logprob}")
        print("-" * 60)


if __name__ == "__main__":
    main()
