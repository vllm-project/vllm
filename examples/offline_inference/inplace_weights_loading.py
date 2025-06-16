# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine.engine.vllm_config.load_config.load_format = "auto"
    await engine.collective_rpc("load_model")
    # start the generation
    results_generator = engine.generate(
        prompt="What is LLM?",
        sampling_params=SamplingParams(temperature=0.0),
        request_id="0",
    )
    # get the results
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    assert final_output is not None
    print("Async engine output:", final_output.outputs[0].text)


def main():
    # Create an LLM without loading real weights
    llm = LLM(
        model="facebook/opt-125m",
        load_format="dummy",
    )

    # llm.llm_engine.model_executor.driver_worker.worker.\
    #    model_runner.vllm_config.load_config.load_format = "auto"
    # Now load real weights inplace
    # llm.llm_engine.vllm_config.load_config.load_format = "auto"
    llm.collective_rpc("load_model")

    # Check real weights are loaded
    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
    # asyncio.run(aync_main())
