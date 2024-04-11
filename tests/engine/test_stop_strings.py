from typing import Any, List, Optional

import pytest

from vllm import CompletionOutput, LLMEngine, SamplingParams

MODEL = "meta-llama/llama-2-7b-hf"
MAX_TOKENS = 200


@pytest.fixture(scope="session")
def vllm_model(vllm_runner):
    return vllm_runner(MODEL)


@pytest.mark.skip_global_cleanup
def test_stop_basic(vllm_model):
    _test_stopping(vllm_model.model.llm_engine,
                   stop=["."],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason=".")

    _test_stopping(vllm_model.model.llm_engine,
                   stop=["."],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization.",
                   expected_reason=".")


@pytest.mark.skip_global_cleanup
def test_stop_multi_tokens(vllm_model):
    _test_stopping(
        vllm_model.model.llm_engine,
        stop=["group of peo", "short"],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer organization. We are a ",
        expected_reason="group of peo")

    _test_stopping(
        vllm_model.model.llm_engine,
        stop=["group of peo", "short"],
        include_in_output=True,
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of peo",
        expected_reason="group of peo")


@pytest.mark.skip_global_cleanup
def test_stop_partial_token(vllm_model):
    _test_stopping(vllm_model.model.llm_engine,
                   stop=["gani"],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer or",
                   expected_reason="gani")

    _test_stopping(vllm_model.model.llm_engine,
                   stop=["gani"],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organi",
                   expected_reason="gani")


@pytest.mark.skip_global_cleanup
def test_stop_token_id(vllm_model):
    # token id 13013 => " organization"

    _test_stopping(vllm_model.model.llm_engine,
                   stop_token_ids=[13013],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer",
                   expected_reason=13013)

    _test_stopping(vllm_model.model.llm_engine,
                   stop_token_ids=[13013],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason=13013)


def _test_stopping(llm_engine: LLMEngine,
                   expected_output: str,
                   expected_reason: Any,
                   stop: Optional[List[str]] = None,
                   stop_token_ids: Optional[List[int]] = None,
                   include_in_output: bool = False) -> None:
    llm_engine.add_request(
        "id", "A story about vLLM:\n",
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stop=stop,
            stop_token_ids=stop_token_ids,
            include_stop_str_in_output=include_in_output,
        ), None)

    output: Optional[CompletionOutput] = None
    output_text = ""
    stop_reason = None
    while llm_engine.has_unfinished_requests():
        (request_output, ) = llm_engine.step()
        (output, ) = request_output.outputs

        # Ensure we don't backtrack
        assert output.text.startswith(output_text)
        output_text = output.text
        stop_reason = output.stop_reason

    assert output is not None
    assert output_text == expected_output
    assert stop_reason == expected_reason
