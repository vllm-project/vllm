# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend and engine-reasoner checks that are not manager-flow tests."""

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
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


def test_xgrammar_accept_tokens_stops_at_termination(capfd):
    """Tokens after a terminating EOS do not reach the matcher."""
    tokenizer, _, request, prompt = _make_manager_and_request("xgrammar")
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    eos = tokenizer.eos_token_id
    trailing = tokenizer.encode("\n")[0]
    processed_before = grammar.num_processed_tokens

    assert grammar.accept_tokens(request.request_id, [eos, trailing])
    assert grammar.is_terminated()
    assert grammar.num_processed_tokens == processed_before + 1
    assert "trying to accept new token" not in capfd.readouterr().err

    processed_after_eos = grammar.num_processed_tokens
    assert grammar.accept_tokens(request.request_id, [trailing])
    assert grammar.num_processed_tokens == processed_after_eos
    assert "trying to accept new token" not in capfd.readouterr().err

    grammar.reset()
    assert not grammar.is_terminated()
    assert grammar.num_processed_tokens == 0


def test_xgrammar_validate_tokens_stops_at_termination(capfd):
    """Validation rolls back after reaching a terminating EOS."""
    tokenizer, _, request, prompt = _make_manager_and_request("xgrammar")
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    eos = tokenizer.eos_token_id
    trailing = tokenizer.encode("\n")[0]

    assert grammar.validate_tokens([eos, trailing]) == [eos]
    assert "trying to accept new token" not in capfd.readouterr().err
    assert not grammar.matcher.is_terminated()

    assert grammar.accept_tokens(request.request_id, [eos])
    assert grammar.is_terminated()

    assert grammar.validate_tokens([trailing]) == []
    assert "trying to accept new token" not in capfd.readouterr().err


class _EngineReasonerStub(ParserEngineReasoningAdapter):
    """Adapter-typed reasoner with a fixed end-token set and no real engine."""

    def __init__(self, end_token_ids):
        self._end_token_ids = frozenset(end_token_ids)
        self.windows: list[list[int]] = []

    @property
    def reasoning_end_token_ids(self):
        return self._end_token_ids

    def find_reasoning_end_offset(self, token_ids):
        self.windows.append(list(token_ids))
        for offset, token in enumerate(token_ids):
            if token in self._end_token_ids:
                return offset
        return len(token_ids)

    def is_reasoning_end(self, input_ids):
        return any(token in self._end_token_ids for token in input_ids)

    def is_reasoning_end_streaming(self, input_ids, delta_ids):
        raise AssertionError("engine path must not rescan draft prefixes")


@pytest.mark.parametrize("backend", ["xgrammar", "guidance"])
def test_bitmask_engine_reasoner_ends_midwindow_with_padding(backend):
    """Engine reasoners see the draft window once, without -1 padding."""
    tokenizer, manager, request, prompt = _make_manager_and_request(backend)
    grammar = request.structured_output_request.grammar

    assert grammar.accept_tokens(request.request_id, prompt)

    marker = tokenizer.encode("\n")[0]
    reasoner = _EngineReasonerStub({marker})
    manager.reasoner_cls = _EngineReasonerStub
    request.structured_output_request.reasoner = reasoner
    request.structured_output_request.reasoning_ended = False

    pre = tokenizer.encode(" ")[0]
    post = tokenizer.encode(",")[0]
    drafts = [pre, marker, post, -1]

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
    assert reasoner.windows == [[pre, marker, post]]
    assert not grammar.is_terminated()
