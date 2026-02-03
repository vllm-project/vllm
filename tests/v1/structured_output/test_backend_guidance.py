# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from concurrent.futures import Future
from typing import Any

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

TOKENIZER = "gpt2"


def test_backend_guidance_rollback_terminated():
    # Test that the backend guidance successfully rollbacks from a
    # terminated state. This can happen with speculative decoding,
    # where the draft model proposes EOS and it is verified by the
    # guidance backend. In that case we are in a stopped state, but
    # it should be reverted in case EOS is not accepted by the target
    # model.
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(
            backend="guidance",
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    backend = GuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    grammar = backend.compile_grammar(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )

    prompt = tokenizer.encode('{"a": "b"}')
    assert len(prompt) > 1
    dummy_wrong = tokenizer.encode('{"a"}')
    for token in prompt:
        assert grammar.accept_tokens("", [token])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Giving any other token should also be accepted
    assert grammar.accept_tokens("", dummy_wrong)
    # Rollback is done from where state was terminated, so from '}' not EOS
    grammar.rollback(len(prompt) - 1)
    assert not grammar.is_terminated()
    assert grammar.validate_tokens([tokenizer.eos_token_id]) == []
    assert grammar.validate_tokens(dummy_wrong) != dummy_wrong
    assert grammar.accept_tokens("", prompt[1:])
    assert not grammar.is_terminated()
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()
    # Rollback of <= 0 should not change the terminated state
    grammar.rollback(0)
    assert grammar.is_terminated()
    grammar.rollback(-1)
    assert grammar.is_terminated()


def test_grammar_bitmask_with_specdec():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=3),
    )
    structured_output_manager = StructuredOutputManager(vllm_config)

    for i in range(1, 2):
        sampling_params = SamplingParams(
            structured_outputs=StructuredOutputsParams(
                json='{"type": "object"}',
            ),
        )
        sampling_params.structured_outputs._backend = "guidance"

        my_req_id = f"my_req_id_{i}"
        request = Request(
            my_req_id,
            prompt_token_ids=prompt[:i],
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=tokenizer.eos_token_id,
        )

        structured_output_manager.grammar_init(request)

        def grammar_bitmask(req: Request, tokens: list[int]) -> None:
            structured_output_manager.grammar_bitmask(
                requests={req.request_id: req},
                structured_output_request_ids={req.request_id: 0},
                scheduled_spec_decode_tokens={req.request_id: tokens},
            )
            # At this point, we rolled-back, so should not be terminated
            assert not req.structured_output_request.grammar.is_terminated()

        # The grammar might not yet be compiled, so we wait for it
        while not request.structured_output_request._check_grammar_completion():
            continue

        assert request.structured_output_request.grammar.accept_tokens(
            request.request_id, prompt[:i]
        )

        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])
        grammar_bitmask(
            request, prompt[i:] + [tokenizer.eos_token_id] + prompt
        )  # EOS not the final token
        grammar_bitmask(request, prompt[i:])  # EOS not present
        grammar_bitmask(request, prompt[i:] + [tokenizer.eos_token_id])


@pytest.mark.parametrize("async_grammar", [True, False])
def test_grammar_init_async_and_sync(async_grammar):
    """Test grammar initialization works correctly in both async and sync modes.

    This test validates that the distributed_executor_backend config option
    correctly controls whether grammar compilation happens asynchronously
    (via executor.submit) or synchronously. When set to "external_launcher",
    grammar compilation is synchronous to avoid deadlocks.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode('{"a": "b"}')

    # Use "external_launcher" for sync mode, None for async mode
    executor_backend = None if async_grammar else "external_launcher"
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        parallel_config=ParallelConfig(distributed_executor_backend=executor_backend),
    )
    structured_output_manager = StructuredOutputManager(vllm_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(
            json='{"type": "object"}',
        ),
    )
    sampling_params.structured_outputs._backend = "guidance"

    request = Request(
        "test_request",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=tokenizer.eos_token_id,
    )

    structured_output_manager.grammar_init(request)

    # Check the internal _grammar type immediately after init
    # Before _check_grammar_completion is called, async mode should have a Future
    raw_grammar = request.structured_output_request._grammar
    if async_grammar:
        assert isinstance(raw_grammar, Future), (
            "Async mode should store a Future before completion"
        )
    else:
        assert not isinstance(raw_grammar, Future), (
            "Sync mode should store the grammar directly, not a Future"
        )

    # Wait for grammar to be ready (handles both async and sync cases)
    start_time = time.time()
    while not request.structured_output_request._check_grammar_completion():
        if time.time() - start_time > 5:  # 5-second timeout
            pytest.fail("Grammar compilation timed out")
        time.sleep(0.01)

    # After completion, _grammar should no longer be a Future
    assert not isinstance(request.structured_output_request._grammar, Future)

    # Verify grammar is properly initialized and functional
    grammar = request.structured_output_request.grammar
    assert grammar is not None
    assert not grammar.is_terminated()

    # Verify the grammar can accept valid tokens
    assert grammar.accept_tokens(request.request_id, prompt)


def test_max_num_spec_tokens_property():
    """Test the max_num_spec_tokens property on GuidanceBackend."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    # Test without speculative config - should return 0
    vllm_config_no_spec = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend_no_spec = GuidanceBackend(
        vllm_config_no_spec,
        tokenizer=tokenizer,
        vocab_size=50257,
    )
    assert backend_no_spec.max_num_spec_tokens == 0

    # Test with speculative config - should return configured value
    vllm_config_with_spec = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=5),
    )
    backend_with_spec = GuidanceBackend(
        vllm_config_with_spec,
        tokenizer=tokenizer,
        vocab_size=50257,
    )
    assert backend_with_spec.max_num_spec_tokens == 5


def test_fill_bitmasks_batch_basic():
    """Test fill_bitmasks_batch with basic usage."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        decoding_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend = GuidanceBackend(
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


def test_fill_bitmasks_batch_with_speculative_tokens():
    """Test batch fill with speculative decoding tokens."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(model="[ngram]", num_speculative_tokens=3),
    )
    backend = GuidanceBackend(
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
        decoding_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend = GuidanceBackend(
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
        decoding_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend = GuidanceBackend(
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
    invalid_tokens = tokenizer.encode("invalid text that breaks json")

    # Test batch accept
    requests: list[Any] = [
        (grammar1, valid_tokens),
        (grammar2, valid_tokens),
        (grammar3, invalid_tokens),  # This should still return True for guidance
    ]

    results = backend.accept_tokens_batch(requests)

    # Verify returns list of booleans
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, bool) for r in results)

    # Valid tokens should be accepted
    assert results[0] is True
    assert results[1] is True
