# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm import LLM, SamplingParams

MODEL = "meta-llama/llama-2-7b-hf"
MAX_TOKENS = 200


def _test_stopping(
    llm: LLM,
    expected_output: str,
    expected_reason: Any,
    stop: list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    include_in_output: bool = False,
) -> None:
    output = llm.generate(
        "A story about vLLM:\n",
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stop=stop,
            stop_token_ids=stop_token_ids,
            include_stop_str_in_output=include_in_output,
        ),
    )[0].outputs[0]

    assert output is not None
    assert output.text == expected_output
    assert output.stop_reason == expected_reason


def _stop_basic(llm):
    _test_stopping(
        llm,
        stop=["."],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer organization",
        expected_reason=".",
    )

    _test_stopping(
        llm,
        stop=["."],
        include_in_output=True,
        expected_output="VLLM is a 100% volunteer organization.",
        expected_reason=".",
    )


def _stop_multi_tokens(llm):
    _test_stopping(
        llm,
        stop=["group of peo", "short"],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer organization. We are a ",
        expected_reason="group of peo",
    )

    _test_stopping(
        llm,
        stop=["group of peo", "short"],
        include_in_output=True,
        expected_output="VLLM is a 100% volunteer organization. We are a group of peo",
        expected_reason="group of peo",
    )


def _stop_partial_token(llm):
    _test_stopping(
        llm,
        stop=["gani"],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer or",
        expected_reason="gani",
    )

    _test_stopping(
        llm,
        stop=["gani"],
        include_in_output=True,
        expected_output="VLLM is a 100% volunteer organi",
        expected_reason="gani",
    )


def _stop_token_id(llm):
    # token id 13013 => " organization"

    _test_stopping(
        llm,
        stop_token_ids=[13013],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer",
        expected_reason=13013,
    )

    _test_stopping(
        llm,
        stop_token_ids=[13013],
        include_in_output=True,
        expected_output="VLLM is a 100% volunteer organization",
        expected_reason=13013,
    )


@pytest.mark.skip_global_cleanup
def test_stop_strings():
    llm = LLM(MODEL, enforce_eager=True)

    _stop_basic(llm)
    _stop_multi_tokens(llm)
    _stop_partial_token(llm)
    # FIXME: this does not respect include_in_output=False
    # _stop_token_id(llm)
