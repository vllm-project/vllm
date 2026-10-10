# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for StructuredOutputRequest grammar-future polling."""

from concurrent.futures import Future

from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.structured_output.request import StructuredOutputRequest


def _request_with_future(fut: Future) -> StructuredOutputRequest:
    req = StructuredOutputRequest(params=StructuredOutputsParams(regex="a+"))
    req.grammar = fut
    return req


def test_pending_grammar_future_is_not_ready():
    req = _request_with_future(Future())
    assert not req.is_grammar_ready
    assert req.grammar is None


def test_grammar_compile_raising_timeout_error_is_a_failure():
    """A compile that raised TimeoutError must be treated as a failed compile,
    not as still-compiling (#53130)."""
    fut: Future = Future()
    fut.set_exception(TimeoutError("compiler timed out"))
    req = _request_with_future(fut)
    assert req.is_grammar_ready
    assert isinstance(req.grammar, TimeoutError)
