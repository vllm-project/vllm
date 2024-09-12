import os

import pytest

from vllm.utils import cuda_device_count_stateless

from ..utils import fork_new_process_for_each_test

TEST_MODELS=[
    ("facebook/opt-125m", {})
    ("nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change", {"dtype":torch.float16, "quantization":"compressed-tensors"})
    ("neuralmagic/Meta-Llama-3-8B-Instruct-FP8", {"dtype":torch.float16, "quantization":"fp8"})
    ("nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Dyn-Per-Token-2048-Samples", {"quantization":"compressed-tensors"})
    ("meta-llama/Meta-Llama-3-8B", {})
]

@pytest.mark.parametrize("model", TEST_MODELS)
@pytest.mark.parametrize("tp_size", [1, 2])
@fork_new_process_for_each_test
def test_full_graph(model, tp_size):

    # Skip the test if there are not enough CUDA devices.
    if cuda_device_count_stateless() < tp_size:
        pytest.skip("Not enough CUDA devices for the test.")

    # make sure these models can be captured in full graph mode
    if "VLLM_TEST_DYNAMO_GRAPH_CAPTURE" not in os.environ:
        os.environ["VLLM_TEST_DYNAMO_GRAPH_CAPTURE"] = "1"

    from vllm import LLM, SamplingParams
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model=model,
              enforce_eager=True,
              load_format="dummy",
              *model_kwargs)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
