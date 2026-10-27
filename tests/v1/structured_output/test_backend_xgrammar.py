# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER = "gpt2"


def _make_backend() -> XgrammarBackend:
    structured_outputs_config = StructuredOutputsConfig(backend="xgrammar")
    vllm_config = VllmConfig(
        structured_outputs_config=structured_outputs_config,
        # Platform auto-detection does not work on every dev machine
        # (e.g. macOS); the backend itself is platform-independent.
        device_config=DeviceConfig(device="cpu"),
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=50257)


def test_backend_xgrammar_validate_tokens_after_terminated():
    # Once the FSM has terminated (the grammar accepted the stop token),
    # validate_tokens() must not keep probing the underlying matcher.
    # accept_tokens() already short-circuits on self._is_terminated
    # (and so does the guidance backend's validate_tokens()); xgrammar's
    # validate_tokens() was missing the same guard, so speculative
    # decoding could keep calling into an already-terminated matcher
    # after every verification step. See
    # https://github.com/vllm-project/vllm/issues/55313
    backend = _make_backend()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    prompt = tokenizer.encode('{"a": "b"}')
    assert len(prompt) > 1
    for token in prompt:
        assert grammar.accept_tokens("", [token])
    assert not grammar.is_terminated()

    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()

    # Draft/speculative tokens proposed after termination must be
    # rejected outright, without the matcher being probed at all: a
    # terminated xgrammar matcher logs a C++-level warning (and, for a
    # grammar whose vocabulary happens to accept trailing whitespace,
    # can be driven to emit it) for every token fed to it post-
    # termination.
    dummy_wrong = tokenizer.encode('{"a"}')
    with patch.object(
        grammar.matcher, "accept_token", wraps=grammar.matcher.accept_token
    ) as accept_token:
        assert grammar.validate_tokens(dummy_wrong) == []
        assert grammar.validate_tokens([tokenizer.eos_token_id]) == []
        accept_token.assert_not_called()

    # The matcher itself is untouched: the grammar is still terminated,
    # and a subsequent accept_tokens() call also correctly no-ops.
    assert grammar.is_terminated()
    assert not grammar.accept_tokens("", dummy_wrong)
