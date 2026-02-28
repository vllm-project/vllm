# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig

lmformatenforcer = pytest.importorskip("lmformatenforcer")

from vllm.v1.structured_output.backend_lm_format_enforcer import (  # noqa: E402
    LMFormatEnforcerBackend,
)
from vllm.v1.structured_output.backend_types import (  # noqa: E402
    StructuredOutputOptions,
)

TOKENIZER = "gpt2"


def test_max_num_spec_tokens_property():
    """Test the max_num_spec_tokens property on LMFormatEnforcerBackend."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    # Test without speculative config - should return 0
    vllm_config_no_spec = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="lm-format-enforcer"),
    )
    backend_no_spec = LMFormatEnforcerBackend(
        vllm_config_no_spec,
        tokenizer=tokenizer,
        vocab_size=50257,
    )
    assert backend_no_spec.max_num_spec_tokens == 0

    # Note: LMFormatEnforcerBackend does NOT support speculative tokens
    # (raises ValueError in compile_grammar if max_num_spec_tokens > 0)
    # So we only test the property returns the value, but don't test
    # speculative decoding functionality


def test_fill_bitmasks_batch_basic():
    """Test fill_bitmasks_batch with basic usage."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="lm-format-enforcer"),
    )
    backend = LMFormatEnforcerBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    # Create a few grammars
    grammar1 = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )
    grammar2 = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "string"}'
    )

    # Allocate bitmask for 2 sequences
    bitmask = backend.allocate_token_bitmask(2)
    full_mask = bitmask.new_full((bitmask.shape[1],), -1)

    # Zero out bitmask to verify it gets filled
    bitmask.zero_()

    # Create batch request list: (grammar, start_index, apply_bitmask, spec_tokens)
    requests: list[Any] = [
        (grammar1, 0, True, []),
        (grammar2, 1, True, []),
    ]

    backend.fill_bitmasks_batch(requests, bitmask, full_mask)

    # Verify bitmasks are filled (not all zeros)
    assert bitmask[0].sum() != 0, "Bitmask for grammar1 should not be all zeros"
    assert bitmask[1].sum() != 0, "Bitmask for grammar2 should not be all zeros"


def test_fill_bitmasks_batch_speculative_not_supported():
    """Test that LMFormatEnforcerBackend raises error with speculative config."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="lm-format-enforcer"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=3),
    )
    backend = LMFormatEnforcerBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    # Should raise ValueError when trying to compile grammar with spec tokens
    with pytest.raises(ValueError, match="does not support speculative tokens"):
        backend.compile_grammar(StructuredOutputOptions.JSON, '{"type": "object"}')


def test_fill_bitmasks_batch_terminated_and_disabled():
    """Test edge cases: terminated grammar and apply_bitmask=False."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="lm-format-enforcer"),
    )
    backend = LMFormatEnforcerBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    # Create grammars
    grammar_terminated = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )
    grammar_disabled = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    # Terminate one grammar by accepting a valid sequence + EOS
    prompt = tokenizer.encode('{"a": "b"}')
    for token in prompt:
        grammar_terminated.accept_tokens("", [token])
    grammar_terminated.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar_terminated.is_terminated()

    # Allocate bitmask
    bitmask = backend.allocate_token_bitmask(2)
    full_mask = bitmask.new_full((bitmask.shape[1],), -1)

    # Zero out bitmask
    bitmask.zero_()

    requests: list[Any] = [
        # Terminated grammar -> should fill with full_mask
        (grammar_terminated, 0, True, []),
        # apply_bitmask=False -> should fill with full_mask
        (grammar_disabled, 1, False, []),
    ]

    backend.fill_bitmasks_batch(requests, bitmask, full_mask)

    # Both should be filled with full_mask (-1)
    assert (bitmask[0] == full_mask).all(), "Terminated grammar should have full_mask"
    assert (bitmask[1] == full_mask).all(), "Disabled bitmask should have full_mask"


def test_accept_tokens_batch():
    """Test accept_tokens_batch with multiple grammars."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="lm-format-enforcer"),
    )
    backend = LMFormatEnforcerBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    # Create multiple grammars
    grammar1 = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )
    grammar2 = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )
    grammar3 = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    # Get valid tokens for JSON object
    valid_tokens = tokenizer.encode("{")

    # Test batch accept
    requests: list[Any] = [
        (grammar1, valid_tokens),
        (grammar2, valid_tokens),
        (grammar3, valid_tokens),
    ]

    results = backend.accept_tokens_batch(requests)

    # Verify returns list of booleans
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, bool) for r in results)

    # Valid tokens should be accepted
    assert results[0] is True
    assert results[1] is True
    assert results[2] is True
