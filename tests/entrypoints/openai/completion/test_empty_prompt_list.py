import pytest

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError


def test_empty_prompt_list_rejected():
    with pytest.raises(VLLMValidationError, match="prompt or prompt_embeds"):
        CompletionRequest(model="m", prompt=[])


def test_nonempty_prompt_list_ok():
    req = CompletionRequest(model="m", prompt=["hello"])
    assert req.prompt == ["hello"]
