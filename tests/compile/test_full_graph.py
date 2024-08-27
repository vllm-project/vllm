import os

import pytest
import torch

from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispacther


class MyMod(torch.nn.Module):

    def forward(self, x: torch.Tensor, a: int):
        return (x + a) * 2


class MyWrapper(TorchCompileWrapperWithCustomDispacther):

    def __init__(self, model):
        self.model = model
        compiled_callable = torch.compile(self.forward, backend="eager")
        super().__init__(compiled_callable)

    def forward(self, x: torch.Tensor, a: int):
        # this is the function to be compiled
        return self.model(x, a)

    def __call__(self, x: torch.Tensor, a: int):
        # let torch.compile compile twice
        if len(self.compiled_codes) >= 2:
            with self.dispatch_to_code(0):
                return self.forward(x, a)
        else:
            return self.compiled_callable(x, a)


def test_torch_compile_wrapper():
    mod = MyMod()
    wrapper = MyWrapper(mod)
    x = torch.tensor([1.0])
    wrapper(x, 0)  # first time, compile
    wrapper(x, 1)  # second time, compile
    wrapper(x, 2)  # third time, dispatch to the first compiled code
    assert len(wrapper.compiled_codes) == 2


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
    llm = LLM(model="meta-llama/Meta-Llama-3-8B")
    llm.generate(prompts, sampling_params)
