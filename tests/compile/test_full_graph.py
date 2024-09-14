import os

import pytest


@pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B"])
def test_full_graph(model):
    # make sure these models can be captured in full graph mode
    os.environ["VLLM_TEST_DYNAMO_GRAPH_CAPTURE"] = "1"

    from vllm import LLM, SamplingParams
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model=model, enforce_eager=True)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
