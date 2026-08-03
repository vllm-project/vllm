# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, RequestOutput, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def print_prompts_and_outputs(outputs: list[RequestOutput]) -> None:
    print("-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


def main():
    # Create an LLM without loading real weights
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        load_format="dummy",
        enforce_eager=True,
        tensor_parallel_size=4,
    )
    outputs = llm.generate(prompts, sampling_params)
    print("\nOutputs do not make sense:")
    print_prompts_and_outputs(outputs)

    # Update load format from `dummy` to `auto`
    llm.collective_rpc(
        "update_config", args=({"load_config": {"load_format": "auto"}},)
    )
    # Now reload real weights inplace
    llm.collective_rpc("reload_weights")

    # Check outputs make sense
    outputs = llm.generate(prompts, sampling_params)
    print("\nOutputs make sense after loading real weights:")
    print_prompts_and_outputs(outputs)


if __name__ == "__main__":
    main()
