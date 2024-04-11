import pytest

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
def test_skip_tokenizer_initialization(model: str):
    # This test checks if the flag skip_tokenizer_init skips the initialization
    # of tokenizer and detokenizer. The generated output is expected to contain
    # token ids.
    prompt = (
        "You are a helpful assistant. How do I build a car from cardboard and "
        "paper clips? Is there an easy to follow video tutorial available "
        "online for free?")

    llm = LLM(model=model, skip_tokenizer_init=True)
    sampling_params = SamplingParams(max_tokens=10,
                                     temperature=0.0,
                                     detokenize=False)

    with pytest.raises(ValueError) as err:
        llm.generate(prompt, sampling_params)
    assert "prompts must be None if" in str(err.value)
