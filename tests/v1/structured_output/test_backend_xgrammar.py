# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output import backend_xgrammar
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

pytestmark = pytest.mark.cpu_test


@pytest.mark.skip_global_cleanup
def test_xgrammar_backend_serializes_compile_calls(monkeypatch):
    compiler_state = {
        "active_calls": 0,
        "max_active_calls": 0,
    }
    compiler_state_lock = threading.Lock()

    class FakeTokenizerInfo:
        @staticmethod
        def from_huggingface(tokenizer, vocab_size):
            return (tokenizer, vocab_size)

    class FakeGrammarCompiler:
        def __init__(self, tokenizer_info, **kwargs):
            self.tokenizer_info = tokenizer_info
            self.kwargs = kwargs

        def compile_grammar(self, grammar_spec):
            with compiler_state_lock:
                compiler_state["active_calls"] += 1
                compiler_state["max_active_calls"] = max(
                    compiler_state["max_active_calls"],
                    compiler_state["active_calls"],
                )
            try:
                time.sleep(0.1)
                return grammar_spec
            finally:
                with compiler_state_lock:
                    compiler_state["active_calls"] -= 1

    class FakeGrammarMatcher:
        def __init__(self, ctx, max_rollback_tokens):
            self.ctx = ctx
            self.max_rollback_tokens = max_rollback_tokens

        def accept_token(self, token):
            return True

        def rollback(self, num_tokens):
            pass

        def fill_next_token_bitmask(self, bitmask, idx):
            pass

        def is_terminated(self):
            return False

        def reset(self):
            pass

    monkeypatch.setattr(
        backend_xgrammar,
        "xgr",
        SimpleNamespace(
            TokenizerInfo=FakeTokenizerInfo,
            GrammarCompiler=FakeGrammarCompiler,
            GrammarMatcher=FakeGrammarMatcher,
        ),
    )
    monkeypatch.setattr(
        backend_xgrammar, "is_mistral_tokenizer", lambda tokenizer: False
    )

    backend = backend_xgrammar.XgrammarBackend(
        VllmConfig(
            structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
            device_config=DeviceConfig("cpu"),
        ),
        tokenizer=object(),
        vocab_size=32,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                backend.compile_grammar,
                StructuredOutputOptions.GRAMMAR,
                'root ::= "ok"',
            )
            for _ in range(2)
        ]
        grammars = [future.result() for future in futures]

    assert compiler_state["max_active_calls"] == 1
    assert grammars[0].matcher is not grammars[1].matcher
