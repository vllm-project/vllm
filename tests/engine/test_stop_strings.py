# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import pytest

from vllm import CompletionOutput, LLMEngine, SamplingParams

MODEL = "meta-llama/llama-2-7b-hf"
MAX_TOKENS = 200

IS_ASYNC = False


@pytest.fixture(scope="session")
def vllm_model(vllm_runner):
    with vllm_runner(MODEL) as vllm_model:
        yield vllm_model


def _test_stopping(llm_engine: LLMEngine,
                   expected_output: str,
                   expected_reason: Any,
                   stop: Optional[list[str]] = None,
                   stop_token_ids: Optional[list[int]] = None,
                   include_in_output: bool = False,
                   use_async_output_proc: bool = False) -> None:
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

    if use_async_output_proc:
        llm_engine.step()

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


def _set_async_mode(llm_engine, is_async):
    llm_engine.scheduler[0].use_async_output_proc = is_async


def _stop_basic(llm_engine, is_async):
    _test_stopping(llm_engine,
                   stop=["."],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason=".",
                   use_async_output_proc=is_async)

    _test_stopping(llm_engine,
                   stop=["."],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization.",
                   expected_reason=".",
                   use_async_output_proc=is_async)


def _stop_multi_tokens(llm_engine, is_async):
    _test_stopping(
        llm_engine,
        stop=["group of peo", "short"],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer organization. We are a ",
        expected_reason="group of peo",
        use_async_output_proc=is_async)

    _test_stopping(
        llm_engine,
        stop=["group of peo", "short"],
        include_in_output=True,
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of peo",
        expected_reason="group of peo",
        use_async_output_proc=is_async)


def _stop_partial_token(llm_engine, is_async):
    _test_stopping(llm_engine,
                   stop=["gani"],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer or",
                   expected_reason="gani",
                   use_async_output_proc=is_async)

    _test_stopping(llm_engine,
                   stop=["gani"],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organi",
                   expected_reason="gani",
                   use_async_output_proc=is_async)


def _stop_token_id(llm_engine, is_async):
    # token id 13013 => " organization"

    _test_stopping(llm_engine,
                   stop_token_ids=[13013],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer",
                   expected_reason=13013,
                   use_async_output_proc=is_async)

    _test_stopping(llm_engine,
                   stop_token_ids=[13013],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason=13013,
                   use_async_output_proc=is_async)


@pytest.mark.skip_global_cleanup
def test_stop_basic(vllm_model):
    _set_async_mode(vllm_model.model.llm_engine, True)
    _stop_basic(vllm_model.model.llm_engine, is_async=True)

    _set_async_mode(vllm_model.model.llm_engine, False)
    _stop_basic(vllm_model.model.llm_engine, is_async=False)


@pytest.mark.skip_global_cleanup
def test_stop_multi_tokens(vllm_model):
    _set_async_mode(vllm_model.model.llm_engine, True)
    _stop_multi_tokens(vllm_model.model.llm_engine, is_async=True)

    _set_async_mode(vllm_model.model.llm_engine, False)
    _stop_multi_tokens(vllm_model.model.llm_engine, is_async=False)


@pytest.mark.skip_global_cleanup
def test_stop_partial_token(vllm_model):
    _set_async_mode(vllm_model.model.llm_engine, True)
    _stop_partial_token(vllm_model.model.llm_engine, is_async=True)

    _set_async_mode(vllm_model.model.llm_engine, False)
    _stop_partial_token(vllm_model.model.llm_engine, is_async=False)


@pytest.mark.skip_global_cleanup
def test_stop_token_id(vllm_model):
    _set_async_mode(vllm_model.model.llm_engine, True)
    _stop_token_id(vllm_model.model.llm_engine, is_async=True)

    _set_async_mode(vllm_model.model.llm_engine, False)
    _stop_token_id(vllm_model.model.llm_engine, is_async=False)
