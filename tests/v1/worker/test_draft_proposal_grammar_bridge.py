# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the post-verification draft proposal grammar bridge."""

import json

import pytest
import torch

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.speculator import (
    DraftProposalGrammarContext,
    DraftProposalGrammarRequest,
    resolve_draft_proposal_grammar_readiness,
    suppress_unready_draft_tokens,
)


def _make_context() -> DraftProposalGrammarContext:
    input_batch = InputBatch.make_dummy(
        num_reqs=3,
        num_tokens=3,
        input_buffers=InputBuffers(3, 3, torch.device("cpu")),
    )
    num_sampled = torch.tensor([2, 1, 2], dtype=torch.int32)
    num_rejected = torch.tensor([0, 1, 0], dtype=torch.int32)
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor(
            [[11, 12, -1], [21, -1, -1], [31, 32, -1]],
            dtype=torch.int64,
        ),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )
    return DraftProposalGrammarContext.from_sampler_output(
        input_batch,
        sampler_output,
        num_sampled,
        num_rejected,
    )


def test_request_descriptor_preserves_serializable_grammar_setup():
    sampling_params = SamplingParams(
        stop_token_ids=[7],
        structured_outputs=StructuredOutputsParams(json={"type": "object"}),
    )
    sampling_params.update_from_generation_config({}, eos_token_id=2)

    request = DraftProposalGrammarRequest.from_sampling_params(
        "grammar-request", sampling_params
    )

    assert request.request_id == "grammar-request"
    assert sampling_params.structured_outputs is not None
    assert request.backend == sampling_params.structured_outputs._backend
    assert request.structured_output_key is not None
    assert request.structured_output_key[0] is StructuredOutputOptions.JSON
    assert json.loads(request.structured_output_key[1]) == {"type": "object"}
    assert set(request.stop_token_ids) == {2, 7}


def test_context_carries_only_verified_output_in_batch_order():
    context = _make_context()

    assert len(context.request_ids) == 3
    assert context.request_state_indices.shape == (3,)
    assert context.sampled_token_ids.tolist() == [
        [11, 12, -1],
        [21, -1, -1],
        [31, 32, -1],
    ]
    assert context.num_sampled.tolist() == [2, 1, 2]
    assert context.num_rejected.tolist() == [0, 1, 0]


def test_missing_bridge_readiness_withholds_every_draft():
    context = _make_context()
    readiness = resolve_draft_proposal_grammar_readiness(context, None)
    draft_tokens = torch.tensor([[101, 102], [201, 202], [301, 302]])

    suppress_unready_draft_tokens(draft_tokens, readiness)

    assert torch.equal(draft_tokens, torch.full_like(draft_tokens, -1))


def test_ready_bridge_withholds_only_unready_draft_rows():
    context = _make_context()
    readiness = resolve_draft_proposal_grammar_readiness(
        context, torch.tensor([True, False, True])
    )
    draft_tokens = torch.tensor([[101, 102], [201, 202], [301, 302]])

    suppress_unready_draft_tokens(draft_tokens, readiness)

    assert draft_tokens.tolist() == [[101, 102], [-1, -1], [301, 302]]


def test_bridge_readiness_requires_one_bool_per_request():
    context = _make_context()

    with pytest.raises(ValueError, match="one entry per request"):
        resolve_draft_proposal_grammar_readiness(context, torch.tensor([True, False]))
