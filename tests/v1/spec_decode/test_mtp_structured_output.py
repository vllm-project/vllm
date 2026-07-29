# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""grammar_bitmask under spec-decode draft padding (#44006)."""

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

TOKENIZER = "gpt2"
NUM_SPEC_TOKENS = 4


def _make_manager_and_request(backend: str, prompt_str: str = '{"a": "b"}'):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = tokenizer.encode(prompt_str)

    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend=backend),
        speculative_config=SpeculativeConfig(
            model="[ngram]", num_speculative_tokens=NUM_SPEC_TOKENS
        ),
    )
    manager = StructuredOutputManager(vllm_config)

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
    )
    sampling_params.structured_outputs._backend = backend
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


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_with_padded_invalid_drafts(backend):
    """Bitmask handles -1 padded drafts and returns N+1 rows."""
    tokenizer, manager, request, prompt = _make_manager_and_request(
        backend, prompt_str='{"a"'
    )
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    valid_drafts = [tokenizer.encode(":")[0], tokenizer.encode(' "')[0]]
    padded = valid_drafts + [-1, -1]

    bitmask = manager.grammar_bitmask(
        requests={request.request_id: request},
        structured_output_request_ids=[request.request_id],
        scheduled_spec_decode_tokens={request.request_id: padded},
    )

    assert bitmask is not None
    assert bitmask.shape[0] == len(padded) + 1
    assert not grammar.is_terminated()


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_when_grammar_terminates_mid_window(backend):
    """Drafts following an EOS that terminates the grammar are a no-op."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
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


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_idempotent_across_calls(backend):
    """Repeated calls with the same input return the same bitmask."""
    tokenizer, manager, request, prompt = _make_manager_and_request(
        backend, prompt_str='{"a"'
    )
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    drafts = [tokenizer.encode(":")[0], -1, -1, -1]

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


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bonus_position_constrained_after_invalid_drafts(backend):
    """Regression for #44006: bonus row stays constrained after -1 padding."""
    tokenizer, manager, request, prompt = _make_manager_and_request(
        backend, prompt_str='{"a"'
    )
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    valid = tokenizer.encode(":")[0]
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


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_constrained_when_reasoning_ends_midwindow(backend):
    """Drafts after a mid-window reasoning-end marker stay constrained."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
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


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_post_reasoning_end_drafts_skip_grammar_advance(backend):
    """Post-marker drafts predate the bitmask and may be grammar-invalid;
    grammar_bitmask must skip the grammar advance instead of asserting.
    """
    tokenizer, manager, request, prompt = _make_manager_and_request(
        backend, prompt_str="{"
    )
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


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_validate_tokens_then_bitmask_round_trip(backend):
    """validate_tokens -> pad with -1 -> grammar_bitmask must not assert."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
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


class _MarkerReasoner:
    """Stub reasoner whose reasoning-end marker is a single fixed token."""

    def __init__(self, marker: int):
        self.marker = marker

    def is_reasoning_end(self, input_ids):
        return self.marker in list(input_ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids):
        return self.marker in list(delta_ids)


def _setup_boundary_request(backend: str):
    """Request with a structural-tag key and reasoning not yet ended."""
    from vllm.v1.structured_output.backend_types import StructuredOutputOptions

    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    marker = tokenizer.encode("\n")[0]
    structured_req = request.structured_output_request
    # The grammar itself is JSON (cheap to build); only the key kind matters
    # for the should_advance structural-tag branch, so pre-seed the cached
    # property.
    structured_req.__dict__["structured_output_key"] = (
        StructuredOutputOptions.STRUCTURAL_TAG,
        "",
    )
    manager.reasoner_cls = _MarkerReasoner
    structured_req.reasoner = _MarkerReasoner(marker)
    structured_req.reasoning_ended = False
    return tokenizer, manager, request, prompt, marker


def test_should_advance_records_reasoning_end_index():
    """Regression for #44006 on post-#42452 main: the boundary step must
    record where reasoning ends so the scheduler can trim before advancing.
    """
    tokenizer, manager, request, prompt, marker = _setup_boundary_request("xgrammar")
    structured_req = request.structured_output_request

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode("{")[0]
    request.append_output_token_ids([pre, marker, post])

    assert manager.should_advance(request)
    assert structured_req.reasoning_ended
    # Marker sits at absolute index len(prompt) + 1.
    assert structured_req.reasoning_end_token_index == len(prompt) + 1


def test_trim_reasoning_for_advance():
    """trim drops the marker and everything before it; later steps and
    requests without a recorded boundary pass through unchanged.
    """
    tokenizer, manager, request, prompt, marker = _setup_boundary_request("xgrammar")
    structured_req = request.structured_output_request

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode("{")[0]

    # No boundary recorded yet: pass-through.
    assert manager.trim_reasoning_for_advance(request, [pre]) == [pre]

    # Boundary step: marker mid-step keeps only the suffix.
    step_tokens = [pre, marker, post]
    request.append_output_token_ids(step_tokens)
    assert manager.should_advance(request)
    assert manager.trim_reasoning_for_advance(request, step_tokens) == [post]

    # Boundary step variant: marker last (the #44006 crash shape
    # [198, </think>]) trims to empty -> scheduler skips accept_tokens.
    structured_req.reasoning_end_token_index = len(request.all_token_ids) - 1
    assert manager.trim_reasoning_for_advance(request, step_tokens) == []

    # Later steps: tokens are past the boundary, returned unchanged.
    structured_req.reasoning_end_token_index = len(prompt) + 1
    next_step = [post, post]
    request.append_output_token_ids(next_step)
    assert manager.trim_reasoning_for_advance(request, next_step) == next_step
