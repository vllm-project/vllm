# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

pytestmark = pytest.mark.skip_global_cleanup

# ---------------------------------------------------------------------------
# SamplingParams field
# ---------------------------------------------------------------------------


def test_sampling_params_trace_field_defaults_to_none():
    params = SamplingParams(max_tokens=10)
    assert params.trace_decode_token_ids is None


def test_sampling_params_trace_field_accepts_list():
    ids = [100, 200, 300]
    params = SamplingParams(trace_decode_token_ids=ids)
    assert params.trace_decode_token_ids == ids


def test_sampling_params_trace_field_preserved_by_clone():
    ids = [1, 2, 3]
    params = SamplingParams(trace_decode_token_ids=ids)
    cloned = params.clone()
    assert cloned.trace_decode_token_ids == ids
    assert cloned.trace_decode_token_ids is not params.trace_decode_token_ids


def test_sampling_params_trace_field_rejects_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        SamplingParams(trace_decode_token_ids=[])


@pytest.mark.parametrize("invalid_ids", [[-1, 5], [1, "2"]])
def test_sampling_params_trace_field_rejects_invalid_token_ids(invalid_ids):
    with pytest.raises(ValueError, match="non-negative integers"):
        SamplingParams(trace_decode_token_ids=invalid_ids)


def _make_model_config(vocab_size: int):
    from unittest.mock import Mock

    model_config = Mock()
    model_config.get_vocab_size = lambda: vocab_size
    return model_config


def test_validate_trace_decode_token_ids_accepts_in_vocab():
    params = SamplingParams(trace_decode_token_ids=[0, 50, 99])
    # Should not raise.
    params._validate_trace_decode_token_ids(_make_model_config(vocab_size=100))


def test_validate_trace_decode_token_ids_rejects_out_of_vocab():
    # The non-negative check passes at construction, but the token id exceeds
    # the vocabulary; verify() must reject it before it reaches the sampler.
    params = SamplingParams(trace_decode_token_ids=[0, 100])
    with pytest.raises(ValueError, match="out-of-vocab"):
        params._validate_trace_decode_token_ids(_make_model_config(vocab_size=100))


def test_validate_trace_decode_token_ids_noop_when_unset():
    params = SamplingParams(max_tokens=4)
    # Should not raise when the field is unset.
    params._validate_trace_decode_token_ids(_make_model_config(vocab_size=100))


# ---------------------------------------------------------------------------
# Sampler trace-replay injection
# ---------------------------------------------------------------------------


def _make_sampling_metadata(
    output_token_ids: list[list[int]],
    trace_decode_token_ids: list[list[int]] | None,
) -> SamplingMetadata:
    batch_size = len(output_token_ids)
    return SamplingMetadata(
        temperature=torch.zeros(batch_size),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size),
        presence_penalties=torch.zeros(batch_size),
        repetition_penalties=torch.ones(batch_size),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        trace_decode_token_ids=trace_decode_token_ids,
    )


def test_sampler_forward_injects_trace_tokens():
    sampler = Sampler()
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    metadata = _make_sampling_metadata(
        output_token_ids=[[], [1, -1]],
        trace_decode_token_ids=[[2, 1], [3, 2]],
    )

    output = sampler(logits, metadata)

    assert output.sampled_token_ids.tolist() == [[2], [2]]


def test_sampler_forward_skips_trace_injection_when_unset():
    sampler = Sampler()
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    metadata = _make_sampling_metadata(
        output_token_ids=[[], [1]],
        trace_decode_token_ids=None,
    )

    output = sampler(logits, metadata)

    assert output.sampled_token_ids.tolist() == [[3], [0]]


def test_inject_trace_tokens_step_0():
    """At decode step 0, trace token at index 0 is injected."""
    sampled = torch.tensor([42, 99], dtype=torch.long)
    trace = [[100, 200, 300], [400, 500]]
    output_token_ids: list[list[int]] = [[], []]  # both at step 0

    Sampler._inject_trace_tokens(sampled, trace, output_token_ids)

    assert sampled[0].item() == 100  # trace_tokens[0][0]
    assert sampled[1].item() == 400  # trace_tokens[1][0]


def test_inject_trace_tokens_mixed_steps():
    """Each request can be at a different decode step."""
    sampled = torch.tensor([1, 2, 3], dtype=torch.long)
    trace = [[10, 11, 12], [], [20, 21]]
    output_token_ids = [
        [5],  # req 0 already produced 1 token -> inject trace[0][1] = 11
        [6, 7],  # req 1 has empty trace -> unchanged
        [],  # req 2 at step 0 -> inject trace[2][0] = 20
    ]

    Sampler._inject_trace_tokens(sampled, trace, output_token_ids)

    assert sampled[0].item() == 11  # step 1 of req 0's trace
    assert sampled[1].item() == 2  # no trace for req 1, unchanged
    assert sampled[2].item() == 20  # step 0 of req 2's trace


def test_inject_trace_tokens_past_end_of_trace():
    """If step_idx >= len(trace), do nothing (engine already stopped)."""
    sampled = torch.tensor([99], dtype=torch.long)
    trace = [[10, 11]]
    output_token_ids = [[5, 6, 7]]  # 3 tokens already produced, trace length 2

    Sampler._inject_trace_tokens(sampled, trace, output_token_ids)

    assert sampled[0].item() == 99  # unchanged


def test_inject_trace_tokens_none_trace_is_noop():
    """When trace_decode_token_ids is None, nothing changes (guard in forward)."""
    # _inject_trace_tokens is only called when metadata.trace_decode_token_ids
    # is not None, so passing an empty outer list is the degenerate case.
    sampled = torch.tensor([7], dtype=torch.long)
    Sampler._inject_trace_tokens(sampled, [], [])
    assert sampled[0].item() == 7


def test_inject_trace_tokens_mismatched_output_ids_raises():
    """Mismatched output_token_ids raises IndexError."""
    sampled = torch.tensor([42], dtype=torch.long)
    trace = [[100, 200]]
    output_token_ids: list[list[int]] = []
    with pytest.raises(IndexError):
        Sampler._inject_trace_tokens(sampled, trace, output_token_ids)


def test_inject_trace_tokens_async_placeholder():
    """Step index ignores async -1 placeholders."""
    sampled = torch.tensor([99], dtype=torch.long)
    trace = [[10, 20, 30]]
    output_token_ids: list[list[int]] = [[]]

    Sampler._inject_trace_tokens(sampled, trace, output_token_ids)
    assert sampled[0].item() == 10

    sampled = torch.tensor([99], dtype=torch.long)
    output_token_ids_step1 = [[10]]
    Sampler._inject_trace_tokens(sampled, trace, output_token_ids_step1)
    assert sampled[0].item() == 20

    sampled = torch.tensor([99], dtype=torch.long)
    output_token_ids_with_placeholder = [[10, -1]]
    Sampler._inject_trace_tokens(sampled, trace, output_token_ids_with_placeholder)
    assert sampled[0].item() == 20
