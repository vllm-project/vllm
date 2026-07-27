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


def test_should_advance_boundary_applies_to_json_grammars():
    """With speculative decoding, the boundary step carries tokens sampled
    after the reasoning-end marker for *every* grammar kind, not just
    structural tags. Deferring the advance desyncs the FSM from the emitted
    tokens (the next step re-derives from the initial state and duplicates
    the grammar prefix, e.g. '{"' + '{"city": ...}').
    """
    from vllm.v1.structured_output.backend_types import StructuredOutputOptions

    tokenizer, manager, request, prompt, marker = _setup_boundary_request("xgrammar")
    structured_req = request.structured_output_request
    structured_req.__dict__["structured_output_key"] = (
        StructuredOutputOptions.JSON,
        "{}",
    )

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode("{")[0]
    step_tokens = [pre, marker, post]
    request.append_output_token_ids(step_tokens)

    assert manager.should_advance(request)
    assert structured_req.reasoning_ended
    assert structured_req.reasoning_end_token_index == len(prompt) + 1
    # The scheduler advances with the trimmed suffix: only the post-marker
    # token reaches the grammar.
    assert manager.trim_reasoning_for_advance(request, step_tokens) == [post]


class _K3StyleReasoner:
    """Multi-token markers with last-close-after-last-open semantics,
    mirroring KimiK3ReasoningParser (close=[8,2,9], open=[7,2,9])."""

    _CLOSE = [8, 2, 9]
    _OPEN = [7, 2, 9]

    @staticmethod
    def _last(ids: list[int], sub: list[int]) -> int:
        for i in range(len(ids) - len(sub), -1, -1):
            if ids[i : i + len(sub)] == sub:
                return i
        return -1

    def is_reasoning_end(self, input_ids) -> bool:
        ids = list(input_ids)
        last_close = self._last(ids, self._CLOSE)
        last_open = self._last(ids, self._OPEN)
        if last_open == -1:
            return last_close != -1
        return last_close > last_open

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return self.is_reasoning_end(input_ids)


def test_find_reasoning_end_index_is_frame_independent():
    """Regression: the boundary index must anchor at the prompt end, not the
    step frame. With spec decode + async scheduling the frame can start past
    the marker; a frame-anchored walk fires immediately for full-sequence
    parsers and reports the frame start, so the trim drops grammar tokens
    (or keeps marker tokens) and the FSM desyncs from the emitted stream.
    """
    # Prompt: stale close from a prior turn, then the current think-open
    # (the agent-continuation shape).
    prompt = [1, 8, 2, 9, 5, 7, 2, 9]
    # Output: reasoning tokens, close marker, then grammar tokens.
    out = [11, 12, 8, 2, 9, 30, 31]
    all_ids = prompt + out
    marker_end = len(prompt) + 4

    reasoner = _K3StyleReasoner()
    assert not reasoner.is_reasoning_end(prompt)
    assert reasoner.is_reasoning_end(all_ids)

    # Independent of any frame: always the marker's final-token index.
    got = StructuredOutputManager._find_reasoning_end_index(
        reasoner, all_ids, len(prompt)
    )
    assert got == marker_end
