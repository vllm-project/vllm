# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


async def aync_main():
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        load_format="dummy",
        enforce_eager=True,
    )
    # Create an engine without loading real weights
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # Update load format from `dummy` to `auto`
    await engine.collective_rpc("update_load_config", kwargs={"load_format": "auto"})
    # Now load real weights inplace
    await engine.collective_rpc("load_model")

    # Check outputs make sense
    prompt = "What is LLM?"
    results_generator = engine.generate(
        prompt=prompt,
        sampling_params=SamplingParams(temperature=0.0),
        request_id="0",
    )
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    assert final_output is not None
    print("\nAsync engine Outputs:\n" + "-" * 60)
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {final_output.outputs[0].text!r}")
    print("-" * 60)


def main():
    # Create an LLM without loading real weights
    llm = LLM(
        model="facebook/opt-125m",
        load_format="dummy",
        enforce_eager=True,
    )

    # Update load format from `dummy` to `auto`
    llm.collective_rpc("update_load_config", kwargs={"load_format": "auto"})
    # Now load real weights inplace
    llm.collective_rpc("load_model")

    # Check outputs make sense
    outputs = llm.generate(prompts, sampling_params)
    print("\nLLM Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
    asyncio.run(aync_main())
