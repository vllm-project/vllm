# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""grammar_bitmask under spec-decode draft padding (#44006)."""

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
    """Bitmask handles -1 padded drafts and returns N+1 rows."""
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    valid_drafts = [tokenizer.encode(",")[0], tokenizer.encode('"')[0]]
    padded = valid_drafts + [-1, -1]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: padded},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == len(padded) + 1
    assert not grammar.is_terminated()


def test_bitmask_when_grammar_terminates_mid_window():
    """Drafts following an EOS that terminates the grammar are a no-op."""
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

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
    assert not grammar.is_terminated()


def test_bitmask_idempotent_across_calls():
    """Repeated calls with the same input return the same bitmask."""
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
    """Regression for #44006: bonus row stays constrained after -1 padding."""
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    valid = tokenizer.encode(" ")[0]
    drafts = [valid, -1, -1, -1]
    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1

    assert not (bitmask[-1] == -1).all()
    assert not grammar.is_terminated()


def test_bitmask_constrained_when_reasoning_ends_midwindow():
    """Drafts after a mid-window reasoning-end marker stay constrained."""
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    marker = tokenizer.encode("\n")[0]

    class StubReasoner:
        def __init__(self, *_, **__):
            self.end_token_id = marker

        def is_reasoning_end(self, input_ids):
            return marker in list(input_ids)

        def is_reasoning_end_streaming(self, input_ids, delta_ids):
            return marker in list(delta_ids)

    manager.reasoner_cls = StubReasoner
    request.structured_output_request.reasoner = StubReasoner()
    request.structured_output_request.reasoning_ended = False

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode(",")[0]
    drafts = [pre, marker, post]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1
    assert (bitmask[0] == -1).all()
    assert (bitmask[1] == -1).all()
    assert not (bitmask[2] == -1).all()
    assert not (bitmask[-1] == -1).all()
    assert not grammar.is_terminated()


def test_bitmask_post_reasoning_end_drafts_skip_grammar_advance():
    """Post-marker drafts predate the bitmask and may be grammar-invalid;
    grammar_bitmask must skip the grammar advance instead of asserting.
    """
    tokenizer, manager, request, prompt = _make_manager_and_request(prompt_str="{")
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)
    assert not grammar.is_terminated()

    marker = tokenizer.encode("\n")[0]

    class StubReasoner:
        def __init__(self, *_, **__):
            self.end_token_id = marker

        def is_reasoning_end(self, input_ids):
            return marker in list(input_ids)

        def is_reasoning_end_streaming(self, input_ids, delta_ids):
            return marker in list(delta_ids)

    manager.reasoner_cls = StubReasoner
    request.structured_output_request.reasoner = StubReasoner()
    request.structured_output_request.reasoning_ended = False

    pre = tokenizer.encode(" ")[0]
    # A token that the JSON grammar would reject as the first post-marker
    # token; without the fix grammar.accept_tokens fires the assertion.
    invalid_post = tokenizer.encode("z")[0]
    drafts = [pre, marker, invalid_post]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: drafts},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == len(drafts) + 1
    # Post-marker position is still bitmask-constrained.
    assert not (bitmask[2] == -1).all()
    # Grammar must not have advanced through the unvalidated draft.
    assert not grammar.is_terminated()


def test_validate_tokens_then_bitmask_round_trip():
    """validate_tokens -> pad with -1 -> grammar_bitmask must not assert."""
    tokenizer, manager, request, prompt = _make_manager_and_request()
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    raw_drafts = [tokenizer.encode(",")[0], 99999, 12345, 67890]
    accepted = grammar.validate_tokens(raw_drafts)
    assert len(accepted) <= len(raw_drafts)

    padded = accepted + [-1] * (len(raw_drafts) - len(accepted))
    assert len(padded) == len(raw_drafts)

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: padded},
    )
    assert bitmask is not None
    assert bitmask.shape[0] == len(padded) + 1
    assert not grammar.is_terminated()
