# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional, Union

import pytest

from vllm import LLM, SamplingParams, envs

MODEL = "meta-llama/llama-2-7b-hf"
MAX_TOKENS = 200


def _test_stopping(llm: LLM,
                   expected_output: str,
                   expected_reason: Any,
                   stop: Optional[list[str]] = None,
                   stop_token_ids: Optional[list[Union[int,
                                                       list[int]]]] = None,
                   include_in_output: bool = False,
                   detokenize: bool = True) -> None:
    output = llm.generate(
        "A story about vLLM:\n",
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stop=stop,
            stop_token_ids=stop_token_ids,
            include_stop_str_in_output=include_in_output,
            detokenize=detokenize,
        ))[0].outputs[0]

    assert output is not None
    if not detokenize:
        # Detokenize manually
        output_text = llm.get_tokenizer().decode(list(output.token_ids))
    else:
        output_text = output.text
    assert output_text == expected_output
    assert output.stop_reason == expected_reason


def _set_async_mode(llm, is_async):
    llm.llm_engine.scheduler[0].use_async_output_proc = is_async


def _stop_basic(llm):
    _test_stopping(llm,
                   stop=["."],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason=".")

    _test_stopping(llm,
                   stop=["."],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization.",
                   expected_reason=".")


def _stop_multi_tokens(llm):
    _test_stopping(
        llm,
        stop=["group of peo", "short"],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer organization. We are a ",
        expected_reason="group of peo")

    _test_stopping(
        llm,
        stop=["group of peo", "short"],
        include_in_output=True,
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of peo",
        expected_reason="group of peo")


def _stop_partial_token(llm):
    _test_stopping(llm,
                   stop=["gani"],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer or",
                   expected_reason="gani")

    _test_stopping(llm,
                   stop=["gani"],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organi",
                   expected_reason="gani")


def _stop_token_id(llm):
    # token id 13013 => " organization"

    _test_stopping(llm,
                   stop_token_ids=[13013],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer",
                   expected_reason=13013)

    _test_stopping(llm,
                   stop_token_ids=[13013],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason=13013)


def _stop_token_id_multi(llm):
    # token id 13013 => " organization"
    # token id 3273 => "short"
    # token id 2318 => "group"
    # token ids [2318, 310, 2305] => "group of people"

    # single grouped stop token id
    _test_stopping(llm,
                   stop_token_ids=[[13013]],
                   include_in_output=False,
                   expected_output="VLLM is a 100% volunteer",
                   expected_reason="[13013]")

    _test_stopping(llm,
                   stop_token_ids=[[13013]],
                   include_in_output=True,
                   expected_output="VLLM is a 100% volunteer organization",
                   expected_reason="[13013]")

    # mixed stop token ids
    _test_stopping(
        llm,
        stop_token_ids=[[2318, 310, 2305], 3273],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer organization. We are a",
        expected_reason="[2318, 310, 2305]")

    _test_stopping(
        llm,
        stop_token_ids=[[2318, 310, 2305], 3273],
        include_in_output=True,
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of people",
        expected_reason="[2318, 310, 2305]")

    # case where no detokenizer is used
    _test_stopping(
        llm,
        stop_token_ids=[[2318, 310, 2305], 3273],
        include_in_output=False,
        detokenize=False,  # required to pass list[list[int]] to stop 
        # include_in_output=False does not work as token ids are not truncated
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of people",
        expected_reason="[2318, 310, 2305]")

    _test_stopping(
        llm,
        stop_token_ids=[[2318, 310, 2305], 3273],
        include_in_output=True,
        detokenize=False,
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of people",
        expected_reason="[2318, 310, 2305]")

    # Case where single stop token id and grouped token ids overlap (2318)
    _test_stopping(
        llm,
        stop_token_ids=[2318, [2318, 310, 2305]],
        include_in_output=False,
        expected_output="VLLM is a 100% volunteer organization. We are a",
        expected_reason=2318)

    _test_stopping(
        llm,
        stop_token_ids=[2318, [2318, 310, 2305]],
        include_in_output=True,
        expected_output="VLLM is a 100% volunteer organization. We are a group",
        expected_reason=2318)

    _test_stopping(
        llm,
        stop_token_ids=[2305, [2318, 310, 2305]],
        include_in_output=True,
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of people",
        expected_reason='[2318, 310, 2305]')

    _test_stopping(
        llm,
        stop_token_ids=[2305, [2318, 310, 2305]],
        detokenize=False,
        include_in_output=True,
        expected_output=
        "VLLM is a 100% volunteer organization. We are a group of people",
        expected_reason='[2318, 310, 2305]')


@pytest.mark.skip_global_cleanup
def test_stop_strings():
    # If V0, must set enforce_eager=False since we use
    # async output processing below.
    llm = LLM(MODEL, enforce_eager=envs.VLLM_USE_V1)

    if envs.VLLM_USE_V1:
        _stop_basic(llm)
    else:
        _set_async_mode(llm, True)
        _stop_basic(llm)

        _set_async_mode(llm, False)
        _stop_basic(llm)

    if envs.VLLM_USE_V1:
        _stop_multi_tokens(llm)
    else:
        _set_async_mode(llm, True)
        _stop_multi_tokens(llm)

        _set_async_mode(llm, False)
        _stop_multi_tokens(llm)

    if envs.VLLM_USE_V1:
        _stop_partial_token(llm)
    else:
        _set_async_mode(llm, True)
        _stop_partial_token(llm)

        _set_async_mode(llm, False)
        _stop_partial_token(llm)

    if envs.VLLM_USE_V1:
        _stop_token_id(llm)
        _stop_token_id_multi(llm)
    else:
        _set_async_mode(llm, True)
        _stop_token_id(llm)

        _set_async_mode(llm, False)
        _stop_token_id(llm)
