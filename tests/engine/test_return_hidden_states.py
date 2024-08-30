import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_return_hidden_states(model: str):
    # This test checks if stepping the LLM successfully runs iterations
    # and returns hidden states for each request.
    engine_args = EngineArgs(model=model,
                             return_hidden_states=True,
                             enforce_eager=True)

    engine = LLMEngine.from_engine_args(engine_args)
    tokenizer = engine.tokenizer.tokenizer
    sampling_params = SamplingParams()
    prompt1 = (
        "You are a helpful assistant. How do I build a car from cardboard and "
        "paper clips? Is there an easy to follow video tutorial available "
        "online for free?")
    prompt1_tokens = tokenizer(prompt1)['input_ids']
    prompt2a = (
        " Please recommend to me some resources where I can learn not only to "
        "handle technical difficulties of building a car, but also "
        "decoration.")
    prompt2b = (" Please only recommend resources that build cars capable of "
                "supersonic speeds.")

    prompt_2a_tokens = tokenizer(prompt2a)['input_ids']
    prompt_2b_tokens = tokenizer(prompt2b)['input_ids']
    assert len(prompt_2a_tokens) > len(prompt_2b_tokens)
    engine.add_request("0", prompt1 + prompt2a, sampling_params)
    engine.add_request("1", prompt1 + prompt2b, sampling_params)
    step1_out = engine.step()
    assert isinstance(step1_out, list)
    request1_out, request2_out = step1_out
    assert (isinstance(request1_out, RequestOutput)
            and isinstance(request2_out, RequestOutput))
    # Ensure prompt hidden states are returned.
    assert (request1_out.prompt_hidden_states is not None
            and request2_out.prompt_hidden_states is not None)
    # Ensure the prompt hidden states match numerically what we would expect.
    assert torch.equal(request1_out.prompt_hidden_states[:len(prompt1_tokens)],
                       request2_out.prompt_hidden_states[:len(prompt1_tokens)])
    assert not torch.equal(
        request1_out.prompt_hidden_states[len(prompt1_tokens):],
        request2_out.prompt_hidden_states[len(prompt1_tokens):])
    # Minus 1 for the additional stop tokens added
    # when the sequences are tokenized separately.
    assert request1_out.prompt_hidden_states.shape[0] == (
        len(prompt1_tokens) + len(prompt_2a_tokens) - 1)
    assert request2_out.prompt_hidden_states.shape[0] == (
        len(prompt1_tokens) + len(prompt_2b_tokens) - 1)
    # Ensure hidden states are returned from step 1.
    assert (request1_out.outputs[0].hidden_states is not None
            and request2_out.outputs[0].hidden_states is not None)
    # Ensure step 1 hidden states match numerically what we would expect.
    assert not torch.equal(request1_out.outputs[0].hidden_states,
                           request2_out.outputs[0].hidden_states)
    request1_hidden_states = (request1_out.outputs[0].hidden_states.clone())
    request2_hidden_states = (request2_out.outputs[0].hidden_states.clone())

    step2_out = engine.step()
    assert isinstance(step2_out, list)
    assert isinstance(step2_out[0], RequestOutput)
    request1_out, request2_out = step2_out
    assert (isinstance(request1_out, RequestOutput)
            and isinstance(request2_out, RequestOutput))
    # Ensure hidden states are being accumulated.
    assert (request1_out.outputs[0].hidden_states is not None
            and request2_out.outputs[0].hidden_states is not None)
    # Ensure hidden states are being accumulated correctly.
    assert request1_out.outputs[0].hidden_states.shape[0] == 2
    assert torch.equal(request1_hidden_states,
                       request1_out.outputs[0].hidden_states[:1])
    assert request2_out.outputs[0].hidden_states.shape[0] == 2
    assert torch.equal(request2_hidden_states,
                       request2_out.outputs[0].hidden_states[:1])
    request1_hidden_states = (request1_out.outputs[0].hidden_states.clone())
    request2_hidden_states = (request2_out.outputs[0].hidden_states.clone())
    step3_out = engine.step()
    assert isinstance(step3_out, list)
    assert isinstance(step3_out[0], RequestOutput)
    request1_out, request2_out = step3_out
    assert (isinstance(request1_out, RequestOutput)
            and isinstance(request2_out, RequestOutput))
    # Ensure hidden states are being accumulated.
    assert (request1_out.outputs[0].hidden_states is not None
            and request2_out.outputs[0].hidden_states is not None)
    # Ensure hidden states are being accumulated correctly.
    assert request1_out.outputs[0].hidden_states.shape[0] == 3
    assert torch.equal(request1_hidden_states,
                       request1_out.outputs[0].hidden_states[:2])
    assert request2_out.outputs[0].hidden_states.shape[0] == 3
    assert torch.equal(request2_hidden_states,
                       request2_out.outputs[0].hidden_states[:2])
