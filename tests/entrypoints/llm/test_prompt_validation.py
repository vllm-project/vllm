import pytest

from vllm import LLM


def test_empty_prompt():
    llm = LLM(model="gpt2", enforce_eager=True)
    with pytest.raises(ValueError, match='Prompt cannot be empty'):
        llm.generate([""])


def test_out_of_vocab_token():
    llm = LLM(model="gpt2", enforce_eager=True)
    with pytest.raises(ValueError, match='out of vocabulary'):
        llm.generate({"prompt_token_ids": [999999]})
