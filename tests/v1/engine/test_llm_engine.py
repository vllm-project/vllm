"""LLMEngine tests"""
import pytest

from vllm import LLM, SamplingParams
from vllm.v1.engine.utils import STR_LLM_ENGINE_PROMPT_LP_APC_UNSUPPORTED


def test_llm_engine_refuses_prompt_logprobs_with_apc(monkeypatch):
    """Test passes if LLMEngine raises an exception when it is configured
    for automatic prefix caching and it receives a request with
    prompt_logprobs enabled, which is incompatible."""
    monkeypatch.setenv("VLLM_USE_V1", "1")
    with pytest.raises(ValueError) as excinfo:
        (LLM(model="facebook/opt-125m", enable_prefix_caching=True).generate(
            "Hello, my name is",
            SamplingParams(temperature=0.8, top_p=0.95, prompt_logprobs=5)))
    # Validate exception string is correct
    assert str(excinfo.value) == STR_LLM_ENGINE_PROMPT_LP_APC_UNSUPPORTED
