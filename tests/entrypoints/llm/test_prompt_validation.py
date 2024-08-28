import pytest

from vllm import LLM


def test_empty_prompt():
    llm = LLM(model="gpt2")
    with pytest.raises(ValueError, match='Prompt cannot be empty'):
        llm.generate([""])
