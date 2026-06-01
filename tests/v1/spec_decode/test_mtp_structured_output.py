# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduction tests for MTP-style speculative decoding + structured output.

These tests target integration paths that are exercised by MTP / EAGLE /
draft-model speculative decoding when combined with grammar-constrained
generation, but are NOT covered by existing tests (`tests/v1/structured_output/`
has only one ngram-based grammar+spec test; `tests/v1/spec_decode/test_mtp.py`
does not touch grammar).

The bug surface is on the structured-output side:
  1. `StructuredOutputManager.grammar_bitmask` advances the FSM for each
     scheduled spec token then rolls back. It contains an `assert accepted`
     that fires if FSM rejects a token it should have already validated.
  2. In async scheduling, invalid draft tokens are padded with `-1`
     (`scheduler.update_draft_token_ids_in_output`). `grammar_bitmask` must
     handle this padding without tripping the assertion or producing wrong
     bitmasks for downstream positions.
  3. If the grammar terminates mid-spec window, subsequent positions must
     remain a no-op for FSM advancement.

We use SpeculativeConfig(model="[ngram]") as a stand-in to avoid loading an
MTP model — the structured-output integration paths under test are
method-agnostic, so this is sufficient to exercise them.
"""

from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

TOKENIZER = "gpt2"
NUM_SPEC_TOKENS = 4


def _make_manager_and_request(prompt_str: str = '{"a": "b"}'):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode(prompt_str)

    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
        speculative_config=SpeculativeConfig(
            model="[ngram]", num_speculative_tokens=NUM_SPEC_TOKENS
        ),
    )
    manager = StructuredOutputManager(vllm_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    sampling_params.structured_outputs._backend = "guidance"
    sampling_params.update_from_generation_config({}, tokenizer.eos_token_id)

    request = Request(
        "mtp_req",
        prompt_token_ids=prompt,
        sampling_params=sampling_params,
        pooling_params=None,
    )
    manager.grammar_init(request)
    while not request.structured_output_request._check_grammar_completion():
        continue

    return tokenizer, manager, request, prompt


def test_bitmask_with_padded_invalid_drafts():
    """Async path: validate_tokens filters invalid drafts and pads with -1.

    `grammar_bitmask` must:
      - not trip `assert accepted` for the padded positions,
      - stop FSM advancement at the first -1,
      - return a bitmask sized for (num_padded_drafts + 1) positions.
    """
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    # Seed FSM with the prompt so it is in a valid mid-state.
    assert grammar.accept_tokens(request.request_id, prompt)

    # Simulate the post-validate_tokens output: 2 valid drafts followed by 2
    # `-1` padding slots (representing tokens dropped by the grammar).
    valid_drafts = [tokenizer.encode(",")[0], tokenizer.encode('"')[0]]
    padded = valid_drafts + [-1, -1]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: padded},
    )

    # One bitmask row per scheduled spec position + one for the bonus token.
    assert bitmask is not None
    assert bitmask.shape[0] == len(padded) + 1

    # FSM state must be unchanged after the bitmask is computed (rollback ran).
    assert not grammar.is_terminated()


def test_bitmask_when_grammar_terminates_mid_window():
    """If the grammar terminates partway through the spec window, subsequent
    positions must be a no-op (no FSM advancement, no assertion fire)."""
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    # Bring grammar near the end: accept the full prompt then submit drafts
    # whose first element is EOS so the grammar terminates immediately.
    assert grammar.accept_tokens(request.request_id, prompt)
    eos = tokenizer.eos_token_id
    drafts = [eos] + [tokenizer.encode(" ")[0]] * (NUM_SPEC_TOKENS - 1)

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == NUM_SPEC_TOKENS + 1
    # FSM should be back to pre-call state (rollback ran).
    assert not grammar.is_terminated()


def test_bitmask_idempotent_across_calls():
    """Calling `grammar_bitmask` repeatedly with the same input must not drift
    FSM state. This catches rollback-accounting bugs that would manifest after
    several scheduler iterations of MTP + grammar."""
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    drafts = [tokenizer.encode(",")[0], -1, -1, -1]

    first = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )
    second = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert first is not None and second is not None
    assert (first == second).all()
    assert not grammar.is_terminated()


def test_bonus_position_constrained_after_invalid_drafts():
    """Regression for #44006.

    When the async scheduling path pads invalid drafts with -1, the
    bonus-token position (the trailing slot the GPU runner uses to
    sample after every spec position) must still receive a
    grammar-constrained bitmask. Otherwise the model is free to emit
    tokens that the FSM excludes at the bonus position, which later
    trips `accept_tokens` in `Scheduler._update_from_output` and
    terminates the request with an internal-server-error in the live
    server.
    """
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    # 1 valid draft + 3 -1 paddings: forces the in-loop apply_bitmask
    # to flip False before reaching the trailing bonus slot.
    valid = tokenizer.encode(" ")[0]
    drafts = [valid, -1, -1, -1]
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1

    # The bonus row is the last one. The unconstrained full mask is
    # int32 -1 (all bits set); a grammar-constrained row must not be
    # all -1.
    bonus_row = bitmask[-1]
    assert not (bonus_row == -1).all(), (
        "bonus position bitmask was filled with the unconstrained full "
        "mask; the model could emit tokens the grammar excludes at the "
        "bonus position"
    )
    assert not grammar.is_terminated()


def test_validate_tokens_then_bitmask_round_trip():
    """End-to-end of the async-path filtering contract:
    `validate_tokens` returns the accepted prefix, scheduler pads with -1,
    and the padded list is what `grammar_bitmask` consumes. The assertion at
    structured_output/__init__.py:286 relies on every non-padding token in
    the input being acceptable to the FSM at the corresponding position."""
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    # Mimic raw MTP drafts: a valid token followed by something the grammar
    # would reject (a random unrelated token id).
    raw_drafts = [tokenizer.encode(",")[0], 99999, 12345, 67890]

    # Step 1: validate_tokens returns the accepted prefix (does not advance).
    accepted = grammar.validate_tokens(raw_drafts)
    assert len(accepted) <= len(raw_drafts)

    # Step 2: scheduler pads with -1 up to the originally scheduled length.
    padded = accepted + [-1] * (len(raw_drafts) - len(accepted))
    assert len(padded) == len(raw_drafts)

    # Step 3: grammar_bitmask consumes the padded list and must not fire the
    # `assert accepted` because every non-(-1) entry was just validated.
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: padded},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == len(padded) + 1
    assert not grammar.is_terminated()
