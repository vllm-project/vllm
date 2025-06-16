# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# def aync_main():
#     engine = AsyncLLMEngine.from_engine_args(engine_args)
#     example_input = {
#         "prompt": "What is LLM?",
#         "stream": False, # assume the non-streaming case
#         "temperature": 0.0,
#         "request_id": 0,
#     }
#     # start the generation
#     results_generator = engine.generate(
#     example_input["prompt"],
#     SamplingParams(temperature=example_input["temperature"]),
#     example_input["request_id"])
#     # get the results
#     final_output = None
#     async for request_output in results_generator:
#         if await request.is_disconnected():
#             # Abort the request if the client disconnects.
#             await engine.abort(request_id)
#             # Return or raise an error
#             ...
#         final_output = request_output


def main():
    # Create an LLM without loading real weights
    llm = LLM(
        model="facebook/opt-125m",
        load_format="dummy",
    )

    # Now load real weights inplace
    llm.llm_engine.vllm_config.load_config.load_format = "auto"
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
