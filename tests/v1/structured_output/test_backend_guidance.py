# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.tokenizers import get_tokenizer
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

TOKENIZER = "openai-community/gpt2"


@pytest.fixture(scope="module")
def mistral_tokenizer():
    return get_tokenizer(
        tokenizer_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        tokenizer_mode="mistral",
    )


def test_backend_guidance_rollback_terminated():
    # Test that the backend guidance successfully rollbacks from a
    # terminated state. This can happen with speculative decoding,
    # where the draft model proposes EOS and it is verified by the
    # guidance backend. In that case we are in a stopped state, but
    # it should be reverted in case EOS is not accepted by the target
    # model.
    structured_outputs_config = StructuredOutputsConfig(backend="guidance")
    vllm_config = VllmConfig(structured_outputs_config=structured_outputs_config)
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
        sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

        my_req_id = f"my_req_id_{i}"
        request = Request(
            my_req_id,
            prompt_token_ids=prompt[:i],
            sampling_params=sampling_params,
            pooling_params=None,
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
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

    request = Request(
        "test_request",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
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


@pytest.mark.parametrize(
    "request_type,grammar_spec",
    [
        pytest.param(
            StructuredOutputOptions.JSON,
            '{"type": "object"}',
            id="json",
        ),
        pytest.param(
            StructuredOutputOptions.GRAMMAR,
            'start: "hello" | "world"',
            id="lark",
        ),
    ],
)
def test_mistral_tokenizer_compile_grammar(
    mistral_tokenizer,
    request_type: StructuredOutputOptions,
    grammar_spec: str,
) -> None:
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend = GuidanceBackend(
        vllm_config,
        tokenizer=mistral_tokenizer,
        vocab_size=mistral_tokenizer.vocab_size,
    )
    assert backend.ll_tokenizer is mistral_tokenizer.llg_tokenizer

    grammar = backend.compile_grammar(request_type, grammar_spec)
    assert grammar is not None
    assert not grammar.is_terminated()


def test_compile_grammar_no_race_condition():
    """Concurrent compile_grammar calls must not cross-contaminate schemas.

    Regression test for a race condition where the shared instance attribute
    self.serialized_grammar allowed one request's grammar spec to overwrite
    another's under the default async thread-pool compilation path.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
    )
    backend = GuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=50257,
    )

    schema_a = json.dumps(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        }
    )
    schema_b = json.dumps(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        }
    )

    tokens_a = tokenizer.encode('{"name": "hello"}')
    tokens_b = tokenizer.encode('{"count": 42}')

    barrier = threading.Barrier(2)

    def compile_synced(schema: str):
        barrier.wait(timeout=5)
        return backend.compile_grammar(StructuredOutputOptions.JSON, schema)

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            for _ in range(200):
                barrier.reset()
                future_a = executor.submit(compile_synced, schema_a)
                future_b = executor.submit(compile_synced, schema_b)

                grammar_a = future_a.result()
                grammar_b = future_b.result()

                accepted_a = grammar_a.validate_tokens(tokens_a)
                assert accepted_a == tokens_a, (
                    f"Grammar A rejected its own valid tokens "
                    f"(accepted {len(accepted_a)}/{len(tokens_a)})"
                )

                accepted_b = grammar_b.validate_tokens(tokens_b)
                assert accepted_b == tokens_b, (
                    f"Grammar B rejected its own valid tokens "
                    f"(accepted {len(accepted_b)}/{len(tokens_b)})"
                )

                cross_a = grammar_a.validate_tokens(tokens_b)
                assert cross_a != tokens_b, (
                    "Grammar A should not accept all of Grammar B's tokens"
                )

                cross_b = grammar_b.validate_tokens(tokens_a)
                assert cross_b != tokens_a, (
                    "Grammar B should not accept all of Grammar A's tokens"
                )
    finally:
        sys.setswitchinterval(old_interval)
