# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for grammar and JSON schema compilation timeout guard.

Verifies that deeply-nested schemas or complex grammars that would cause
exponential state-space explosion are rejected with a timeout rather than
hanging worker threads indefinitely (#54003).
"""

import json
import time
from unittest.mock import patch

import pytest

from vllm.v1.structured_output.utils import compile_grammar_with_timeout


class TestCompileGrammarWithTimeout:
    """Unit tests for the compile_grammar_with_timeout utility."""

    def test_normal_json_schema_compiles_successfully(self):
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        result = compile_grammar_with_timeout(
            lambda s: "compiled",
            schema,
            grammar_spec=schema,
            grammar_type="JSON schema",
        )
        assert result == "compiled"

    def test_normal_grammar_compiles_successfully(self):
        grammar = 'root ::= "hello" | "world"'
        result = compile_grammar_with_timeout(
            lambda g: "compiled",
            grammar,
            grammar_spec=grammar,
            grammar_type="Grammar",
        )
        assert result == "compiled"

    def test_timeout_raises_value_error(self):
        def slow_compile(spec: str):
            time.sleep(10)
            return "never"

        with (
            patch("vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_S", 1),
            pytest.raises(ValueError, match="JSON schema compilation timed out"),
        ):
            compile_grammar_with_timeout(
                slow_compile,
                '{"type": "slow"}',
                grammar_spec='{"type": "slow"}',
                grammar_type="JSON schema",
            )

    def test_timeout_disabled_when_zero(self):
        result = None
        with patch("vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_S", 0):
            result = compile_grammar_with_timeout(
                lambda s: "no_timeout",
                '{"type": "test"}',
                grammar_spec='{"type": "test"}',
                grammar_type="JSON schema",
            )
        assert result == "no_timeout"

    def test_compilation_error_propagates(self):
        def failing_compile(spec: str):
            raise RuntimeError("compilation failed")

        with pytest.raises(RuntimeError, match="compilation failed"):
            compile_grammar_with_timeout(
                failing_compile,
                "bad",
                grammar_spec="bad",
                grammar_type="JSON schema",
            )

    def test_spec_included_in_error_message(self):
        def slow_compile(spec: str):
            time.sleep(10)
            return "never"

        schema = '{"oneOf": [{"type": "string"}]}'
        with (
            patch("vllm.envs.VLLM_GRAMMAR_COMPILATION_TIMEOUT_S", 1),
            pytest.raises(ValueError, match=r'Spec: \{"oneOf"'),
        ):
            compile_grammar_with_timeout(
                slow_compile,
                schema,
                grammar_spec=schema,
                grammar_type="JSON schema",
            )


def generate_nested_schema(depth: int) -> dict:
    """Generate a nested oneOf schema that causes exponential branching."""
    schema: dict = {"type": "string"}
    for _ in range(depth):
        schema = {
            "oneOf": [
                {"type": "object", "properties": {"x": schema}, "required": ["x"]},
                {"type": "object", "properties": {"y": schema}, "required": ["y"]},
            ]
        }
    return schema


def test_real_xgrammar_compilation_timeout():
    """Real E2E test verifying that actual xgrammar compilation aborts on timeout."""
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tok_info)

    # 1. Normal shallow schema compiles quickly (<0.5s)
    shallow_schema = json.dumps(generate_nested_schema(4))
    t0 = time.time()
    result = compile_grammar_with_timeout(
        compiler.compile_json_schema,
        shallow_schema,
        grammar_spec=shallow_schema,
        grammar_type="JSON schema",
        timeout=5,
    )
    assert result is not None
    assert (time.time() - t0) < 1.0

    # 2. Deep schema taking >3s to compile must abort at 1s timeout
    deep_schema = json.dumps(generate_nested_schema(13))
    t0 = time.time()
    with pytest.raises(ValueError, match="JSON schema compilation timed out after 1s"):
        compile_grammar_with_timeout(
            compiler.compile_json_schema,
            deep_schema,
            grammar_spec=deep_schema,
            grammar_type="JSON schema",
            timeout=1,
        )
    elapsed = time.time() - t0
    assert 0.9 <= elapsed <= 2.5, f"Expected ~1s timeout, took {elapsed:.2f}s"
