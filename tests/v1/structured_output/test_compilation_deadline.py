# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for engine-side compilation deadline and error handling
(GHSA-q9q8-922v-2c42, GHSA-29wg-4hmv-h32g)."""

import time
from concurrent.futures import Future
from unittest.mock import MagicMock

from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.structured_output.request import StructuredOutputRequest


class TestCheckGrammarCompletion:
    """_check_grammar_completion catches compile errors and timeouts."""

    def test_successful_completion(self):
        req = StructuredOutputRequest(params=StructuredOutputsParams(regex="[a-z]+"))
        mock_grammar = MagicMock()
        future: Future = Future()
        future.set_result(mock_grammar)
        req._grammar = future

        assert req._check_grammar_completion() is True
        assert req._grammar is mock_grammar
        assert req.compile_error is None

    def test_timeout_returns_false(self):
        req = StructuredOutputRequest(params=StructuredOutputsParams(regex="[a-z]+"))
        future: Future = Future()
        req._grammar = future

        assert req._check_grammar_completion() is False
        assert isinstance(req._grammar, Future)

    def test_compile_exception_captured(self):
        req = StructuredOutputRequest(params=StructuredOutputsParams(regex="[a-z]+"))
        future: Future = Future()
        future.set_exception(ValueError("compilation failed"))
        req._grammar = future

        assert req._check_grammar_completion() is True
        assert req._grammar is None
        assert isinstance(req.compile_error, ValueError)
        assert "compilation failed" in str(req.compile_error)

    def test_grammar_property_returns_none_on_error(self):
        req = StructuredOutputRequest(params=StructuredOutputsParams(regex="[a-z]+"))
        future: Future = Future()
        future.set_exception(RuntimeError("internal error"))
        req._grammar = future

        assert req.grammar is None
        assert req.compile_error is not None


class TestCompilationStartTime:
    """compilation_start_time is recorded for deadline checks."""

    def test_start_compilation_records_time(self):
        req = StructuredOutputRequest(params=StructuredOutputsParams(regex="[a-z]+"))
        assert req.compilation_start_time == 0.0

        before = time.monotonic()
        req.start_compilation()
        after = time.monotonic()

        assert before <= req.compilation_start_time <= after
