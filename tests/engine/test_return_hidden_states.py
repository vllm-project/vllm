import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_return_hidden_states(model: str):
    # This test checks if stepping the LLM successfully runs iterations
    # and returns hidden states.
    prompt = (
        "You are a helpful assistant. How do I build a car from cardboard and "
        "paper clips? Is there an easy to follow video tutorial available "
        "online for free?")
    prompt2 = (
        " Please recommend to me some resources where I can learn not only to "
        "handle technical difficulties of building a car, but also "
        "decoration.")

    engine_args = EngineArgs(model=model, return_hidden_states=True)

    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams()

    engine.add_request("0", prompt + prompt2, sampling_params)
    engine.add_request("1", prompt + prompt2, sampling_params)
    step1_out = engine.step()
    assert isinstance(step1_out, list)
    assert isinstance(step1_out[0], RequestOutput)
    assert step1_out[0].prompt_hidden_states is not None
    assert step1_out[0].outputs[0].hidden_states is not None
    step2_out = engine.step()
    assert isinstance(step2_out, list)
    assert isinstance(step2_out[0], RequestOutput)
    assert step2_out[0].prompt_hidden_states is not None
    assert torch.equal(step1_out[0].prompt_hidden_states,
                       step2_out[0].prompt_hidden_states)
    assert step2_out[0].outputs[0].hidden_states is not None
