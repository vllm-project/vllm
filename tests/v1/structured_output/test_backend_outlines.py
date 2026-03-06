# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig

oc = pytest.importorskip("outlines_core")

from vllm.v1.structured_output.backend_outlines import OutlinesBackend  # noqa: E402
from vllm.v1.structured_output.backend_types import (  # noqa: E402
    StructuredOutputOptions,
)

TOKENIZER = "gpt2"


def test_max_num_spec_tokens_property():
    """Test the max_num_spec_tokens property on OutlinesBackend."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    # Test without speculative config - should return 0
    vllm_config_no_spec = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="outlines"),
    )
    backend_no_spec = OutlinesBackend(
        vllm_config_no_spec,
        tokenizer=tokenizer,
        vocab_size=50257,
    )
    assert backend_no_spec.max_num_spec_tokens == 0

    # Test with speculative config - should return configured value
    vllm_config_with_spec = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="outlines"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=5),
    )
    backend_with_spec = OutlinesBackend(
        vllm_config_with_spec,
        tokenizer=tokenizer,
        vocab_size=50257,
    )
    assert backend_with_spec.max_num_spec_tokens == 5


def test_fill_bitmasks_batch_basic():
    """Test fill_bitmasks_batch with basic usage."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="outlines"),
    )
    backend = OutlinesBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    # Create a few grammars (outlines uses regex, so use JSON schema)
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


def test_fill_bitmasks_batch_with_speculative_tokens():
    """Test batch fill with speculative decoding tokens."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="outlines"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=3),
    )
    backend = OutlinesBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    # Accept initial tokens to set up a valid state
    prompt = tokenizer.encode('{"a": "b"}')
    assert grammar.accept_tokens("", prompt[:1])
    assert not grammar.is_terminated()

    # Allocate bitmask for speculative tokens (1 base + 3 spec = 4 positions)
    num_positions = 4
    bitmask = backend.allocate_token_bitmask(num_positions)
    full_mask = bitmask.new_full((bitmask.shape[1],), -1)

    # Spec tokens that would advance and then need rollback
    spec_tokens = prompt[1:4]  # 3 tokens

    requests: list[Any] = [
        (grammar, 0, True, spec_tokens),
    ]

    backend.fill_bitmasks_batch(requests, bitmask, full_mask)

    # After batch call, grammar should NOT be terminated (rollback should have happened)
    assert not grammar.is_terminated(), (
        "Grammar should not be terminated after batch call with rollback"
    )


def test_fill_bitmasks_batch_terminated_and_disabled():
    """Test edge cases: terminated grammar and apply_bitmask=False."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="outlines"),
    )
    backend = OutlinesBackend(
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

    # For outlines, we need to check is_terminated behavior
    # The grammar tracks _prev_finished state
    prompt = tokenizer.encode("{}")
    for token in prompt:
        grammar_terminated.accept_tokens("", [token])

    # Allocate bitmask
    bitmask = backend.allocate_token_bitmask(2)
    full_mask = bitmask.new_full((bitmask.shape[1],), -1)

    # Zero out bitmask
    bitmask.zero_()

    requests: list[Any] = [
        # Grammar may or may not be terminated depending on state
        (grammar_terminated, 0, True, []),
        # apply_bitmask=False -> should fill with full_mask
        (grammar_disabled, 1, False, []),
    ]

    backend.fill_bitmasks_batch(requests, bitmask, full_mask)

    # Disabled bitmask should be filled with full_mask (-1)
    assert (bitmask[1] == full_mask).all(), "Disabled bitmask should have full_mask"


def test_accept_tokens_batch():
    """Test accept_tokens_batch with multiple grammars."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="outlines"),
    )
    backend = OutlinesBackend(
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
