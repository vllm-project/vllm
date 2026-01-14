# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from tests.v1.sample.utils import create_allowed_token_ids
from vllm.platforms import current_platform
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
from vllm.v1.sample.sampler import Sampler, SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

DEVICE = current_platform.device_type


@pytest.fixture
def rejection_sampler():
    mock_sampler = Mock(spec=Sampler)
    mock_sampler.logprobs_mode = "raw_logprobs"
    return RejectionSampler(mock_sampler)


def mock_sampler_output(
    rejection_sampler: RejectionSampler, bonus_token_ids: torch.Tensor
):
    rejection_sampler.sampler.return_value = SamplerOutput(
        sampled_token_ids=bonus_token_ids, logprobs_tensors=None
    )


def create_spec_decode_metadata(
    spec_tokens: list[list[int]], logits: torch.Tensor
) -> SpecDecodeMetadata:
    metadata = SpecDecodeMetadata.make_dummy(spec_tokens, device=logits.device)
    metadata.target_logits_indices = torch.arange(logits.shape[0])
    # Output bonus token ids are mocked, so the bonus logit indices should
    # be empty.
    metadata.bonus_logits_indices = torch.empty(0, dtype=torch.int32)
    return metadata


def create_logits_tensor(
    output_token_ids: list[list[int]],
    vocab_size: int = 100,
    token_idx_to_override: int | None = None,
) -> torch.Tensor:
    """Helper function to create logits tensor that
    will produce desired token ids on argmax"""
    token_ids = [tokens[:-1] for tokens in output_token_ids]
    num_total_tokens = sum(len(tokens) for tokens in token_ids)
    logits = torch.full((num_total_tokens, vocab_size), -100.0, device=DEVICE)
    start_loc = 0
    for tokens in token_ids:
        for j, token_id in enumerate(tokens):
            logits[start_loc + j, token_id] = 100.0
        start_loc += len(tokens)
    if token_idx_to_override:
        logits[:, token_idx_to_override] = 99.0
    return logits


def create_sampling_metadata(
    all_greedy: bool,
    output_token_ids: list[list[int]] | None = None,
    prompt_token_ids: torch.Tensor | None = None,
    spec_token_ids: torch.Tensor | None = None,
    temperature: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
    generators: dict[int, Any] | None = None,
    frequency_penalties: list[float] | None = None,
    presence_penalties: list[float] | None = None,
    repetition_penalties: list[float] | None = None,
    bad_words_token_ids: dict[int, list[list[int]]] | None = None,
    allowed_token_ids_mask: torch.Tensor | None = None,
) -> SamplingMetadata:
    """Create a v1 sampling metadata object with all_greedy set
    to the given value. Either all greedy or all random sampling
    is used.
    """
    generators = generators or {}
    if all_greedy:
        temperature = None
    else:
        assert temperature is not None

    if any([frequency_penalties, presence_penalties, repetition_penalties]):
        no_penalties = False

        assert output_token_ids
        assert len(output_token_ids) > 0

        frequency_penalties = torch.tensor(frequency_penalties, device=DEVICE)
        presence_penalties = torch.tensor(presence_penalties, device=DEVICE)
        repetition_penalties = torch.tensor(repetition_penalties, device=DEVICE)
    else:
        no_penalties = True
        frequency_penalties = torch.tensor([])
        presence_penalties = torch.tensor([])
        repetition_penalties = torch.tensor([])

    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=top_p,
        top_k=top_k,
        generators=generators,
        max_num_logprobs=None,
        no_penalties=no_penalties,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=frequency_penalties,
        presence_penalties=presence_penalties,
        repetition_penalties=repetition_penalties,
        output_token_ids=[] if output_token_ids is None else output_token_ids,
        spec_token_ids=[] if spec_token_ids is None else spec_token_ids,
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids={} if bad_words_token_ids is None else bad_words_token_ids,
        logitsprocs=LogitsProcessors(),
    )


########################### Tests for Greedy Sampling ###################
def test_perfect_match(rejection_sampler):
    """Test when output tokens perfectly match speculated tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [[1, 2, 3, 4]]  # 4 is the bonus token

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[1, 2, 3, 4]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_early_mismatch(rejection_sampler):
    """Test when there's an early mismatch in tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [[1, 5, 3, 4]]  # Mismatch at position 1

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 5, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_multiple_sequences(rejection_sampler):
    """Test handling multiple sequences of speculated tokens"""
    spec_tokens = [[1, 2], [3]]
    output_tokens = [[1, 2, 5], [3, 4]]  # Two sequences with bonus tokens 5 and 4

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [output_tokens[0][-1], output_tokens[1][-1]], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 2, 5], [3, 4, PLACEHOLDER_TOKEN_ID]], dtype=torch.int, device=logits.device
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_single_token_sequence(rejection_sampler):
    """Test handling sequences with single token"""
    spec_tokens = [[1]]
    output_tokens = [[1, 2]]  # Single token with bonus token 2

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[1, 2]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_empty_sequence(rejection_sampler):
    """Test handling empty sequence of speculated tokens"""
    spec_tokens: list[list[int]] = [[]]
    output_tokens = [[5]]  # Just the bonus token

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor([output_tokens[0][-1]], device=logits.device)
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor([[5]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_multiple_mismatches(rejection_sampler):
    """Test handling multiple sequences with mismatches"""
    spec_tokens = [[1, 2, 3], [4, 5, 6]]
    output_tokens = [[1, 2, 7, 6], [4, 8, 6, 9]]  # Mismatches in both sequences

    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [output_tokens[0][-1], output_tokens[1][-1]], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [
            [1, 2, 7, PLACEHOLDER_TOKEN_ID],
            [4, 8, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected",
    [
        ([[1, 2]], [[1, 2, 3]], [[1, 2, 3]]),  # Perfect match with bonus
        ([[1]], [[2, 3]], [[2, PLACEHOLDER_TOKEN_ID]]),  # First mismatch
        (
            [[1, 2], [3, 4]],
            [[1, 5, 6], [3, 4, 7]],
            [[1, 5, PLACEHOLDER_TOKEN_ID], [3, 4, 7]],
        ),  # Mixed matches
    ],
)
def test_parametrized_cases(rejection_sampler, spec_tokens, output_tokens, expected):
    """Parametrized test for various matching scenarios"""
    metadata = create_sampling_metadata(all_greedy=True)
    logits = create_logits_tensor(output_tokens)
    bonus_token_tensor = torch.tensor(
        [tokens[-1] for tokens in output_tokens], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)

    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected_tensor = torch.tensor(expected, dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected_tensor)


########################### Tests for Random Sampling ###################
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("vocab_size", [1000])
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("frac_seeded", [0.0, 0.5])
@pytest.mark.parametrize("n_rep", [20])
def test_deterministic_when_seeded(
    rejection_sampler,
    k: int,
    vocab_size: int,
    batch_size: int,
    frac_seeded: float,
    n_rep: int,
):
    num_tokens = batch_size * k
    draft_probs = torch.rand(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    draft_probs = F.softmax(draft_probs, dim=-1)
    target_logits = torch.rand_like(draft_probs)
    bonus_token_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, 1), dtype=torch.int64, device=DEVICE
    )
    draft_token_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, k), dtype=torch.int64, device=DEVICE
    )

    seeded_mask = torch.rand(batch_size, dtype=torch.float32) <= frac_seeded

    results = []
    for _ in range(n_rep):
        seeded_seqs = {
            i: torch.Generator(device=DEVICE).manual_seed(i)
            for i in range(batch_size)
            if seeded_mask[i]
        }

        temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE)
        sampling_metadata = create_sampling_metadata(
            all_greedy=False, temperature=temperature, generators=seeded_seqs
        )
        spec_decode_metadata = create_spec_decode_metadata(
            draft_token_ids.tolist(), target_logits
        )

        mock_sampler_output(rejection_sampler, bonus_token_ids)
        rep_result = rejection_sampler(
            spec_decode_metadata,
            draft_probs=None,
            logits=target_logits,
            sampling_metadata=sampling_metadata,
        )

        results.append(rep_result.sampled_token_ids)

    for i in range(batch_size):
        if seeded_mask[i]:
            for j in range(1, n_rep):
                assert torch.equal(results[j][i], results[0][i])


def test_rejection_sampling_approximates_target_distribution():
    """Verify rejection sampling approximates target distribution,
    despite sampling from a potentially distinct draft distribution.

    This is done by first creating a random target probability
    distribution and a random draft probability distribution. We then
    sample token ids from the rejection sampler using these draft
    and target distributions. The samples are used to estimate
    the output probability distribution, which we expect to approximate
    the target distribution.

    A basic distance metric is used to determine similarity between
    distributions.

    We expect that as we increase the number of samples,
    the distance between the observed distribution and the target
    distribution decreases. To measure this, we compare the distance
    of the observed distribution against both the target distribution
    and a uniform random distribution. We expect the distance between
    the observed distribution and the target distribution to improve
    much more than the distance improvement between the observed
    distribution and the random distribution.
    """
    torch.set_default_device(DEVICE)
    vocab_size = 10
    k = 2
    num_reference_probs = 100

    # Prepare draft, target, and reference probability distributions
    draft_probs = F.softmax(torch.rand(vocab_size, dtype=torch.float32), dim=-1)
    target_logits = torch.rand(vocab_size, dtype=torch.float32)
    target_probs = F.softmax(target_logits, dim=-1)
    reference_probs = F.softmax(
        torch.rand(num_reference_probs, vocab_size, dtype=torch.float32),
        dim=-1,
    )

    sample_sizes = [10, 100, 1_000, 10_000, 100_000]
    distance_wrt_reference: list[float] = []
    distance_wrt_target: list[float] = []

    for num_samples in sample_sizes:
        # Sample using rejection sampling.
        rej_sample_probs = estimate_rejection_sampling_pdf(
            draft_probs, target_logits, k, vocab_size, num_samples
        )
        rej_sample_probs = rej_sample_probs.to(DEVICE)

        # Average distance from reference probs.
        reference_vs_rejsample_dist = (
            torch.dist(reference_probs, rej_sample_probs).item()
            / reference_probs.shape[0]
        )
        target_vs_rejsample_dist = torch.dist(target_probs, rej_sample_probs).item()

        distance_wrt_reference.append(reference_vs_rejsample_dist)
        distance_wrt_target.append(target_vs_rejsample_dist)

        relative_change_in_distance_wrt_target = get_ratio_first_to_last(
            distance_wrt_target
        )
        relative_change_in_distance_wrt_reference = get_ratio_first_to_last(
            distance_wrt_reference
        )

        print(
            f"{num_samples=} {target_vs_rejsample_dist=:.05f} "
            f"{reference_vs_rejsample_dist=:.05f}"
        )
        print(
            f"{num_samples=} {relative_change_in_distance_wrt_target=:.02f} "
            f"{relative_change_in_distance_wrt_reference=:.02f}"
        )

    relative_change_in_distance_wrt_target = get_ratio_first_to_last(
        distance_wrt_target
    )
    relative_change_in_distance_wrt_reference = get_ratio_first_to_last(
        distance_wrt_reference
    )

    expected_improvement_multiplier = 20
    assert (
        relative_change_in_distance_wrt_target
        > relative_change_in_distance_wrt_reference * expected_improvement_multiplier
    )


def get_ratio_first_to_last(elements: list[float]) -> float:
    return elements[0] / elements[-1]


def estimate_rejection_sampling_pdf(
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    k: int,
    vocab_size: int,
    num_samples: int,
) -> torch.Tensor:
    """Estimate the probability distribution of the output tokens
    using rejection sampling.

    Args:
        draft_probs: Draft probability distribution.
        target_logits: Target logits.
        num_samples: Number of samples to draw.

    Returns:
        Estimated probability distribution of the output tokens.
    """
    mock_sampler = Mock(spec=Sampler)
    mock_sampler.logprobs_mode = "raw_logprobs"
    rejection_sampler = RejectionSampler(mock_sampler)
    num_tokens = num_samples * k
    # Repeat draft probs num_samples * k times.
    draft_probs = draft_probs.reshape(1, 1, vocab_size).repeat(num_samples, k, 1)

    # Repeat target probs num_tokens times.
    target_logits = target_logits.reshape(1, vocab_size).repeat(num_tokens, 1)

    # Randomly sample draft token ids from draft probs.
    draft_token_ids = torch.multinomial(
        draft_probs[:, 0, :], num_samples=k, replacement=True
    ).reshape(num_samples, k)
    draft_probs = draft_probs.view(num_tokens, vocab_size)

    # Bonus tokens not used but required.
    bonus_token_ids = torch.zeros((1, 1), dtype=torch.int64, device=DEVICE).repeat(
        num_samples, 1
    )

    temperature = torch.ones(num_samples, dtype=torch.float32, device=DEVICE)
    sampling_metadata = create_sampling_metadata(
        all_greedy=False, temperature=temperature
    )
    spec_decode_metadata = create_spec_decode_metadata(
        draft_token_ids.tolist(), target_logits
    )

    mock_sampler_output(rejection_sampler, bonus_token_ids)
    sampler_output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=draft_probs,
        logits=target_logits,
        sampling_metadata=sampling_metadata,
    )
    output_token_ids = sampler_output.sampled_token_ids[:, :-1].flatten()

    hist = torch.histogram(
        output_token_ids.to(dtype=torch.float, device="cpu"),
        bins=vocab_size,
        range=(0, vocab_size),
        density=True,
    )

    return hist.hist


def _test_masked_logits(
    rejection_sampler,
    batch_size: int,
    num_draft_tokens: int,
    vocab_size: int,
    target_logits: torch.Tensor,
    unmasked_indices: torch.Tensor,
    sampling_metadata: SamplingMetadata,
):
    # Set up test parameters
    num_tokens = batch_size * num_draft_tokens

    # Create random draft probabilities.
    draft_probs = torch.rand(
        (num_tokens, vocab_size), dtype=torch.float32, device=DEVICE
    )
    draft_probs = F.softmax(draft_probs, dim=-1)

    # Randomly sample draft token ids from draft probs
    draft_token_ids = torch.multinomial(draft_probs, num_samples=1)
    draft_token_ids = draft_token_ids.reshape(batch_size, num_draft_tokens)
    draft_token_ids = draft_token_ids.tolist()

    # Bonus tokens not used but required
    bonus_token_ids = torch.zeros((batch_size, 1), dtype=torch.int64, device=DEVICE)

    # Create spec decode metadata
    spec_decode_metadata = create_spec_decode_metadata(draft_token_ids, target_logits)

    # Run rejection sampling
    mock_sampler_output(rejection_sampler, bonus_token_ids)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=draft_probs,
        logits=target_logits,
        sampling_metadata=sampling_metadata,
    )

    # Remove bonus tokens and reshape
    output_token_ids = output.sampled_token_ids[:, :-1].flatten().tolist()

    # Check that all sampled tokens are within the unmasked indices.
    for i in range(num_tokens):
        token_id = output_token_ids[i]
        if token_id == PLACEHOLDER_TOKEN_ID:
            continue
        assert token_id in unmasked_indices[i]

@pytest.mark.parametrize("top_k", [1, 5, 99])
def test_top_k(rejection_sampler, top_k):
    """Test rejection sampling with top-k sampling"""
    vocab_size = 100
    batch_size = 100
    num_draft_tokens = 3
    num_tokens = batch_size * num_draft_tokens

    # Randomly create top-k indices.
    top_k_indices = [
        torch.randperm(vocab_size, device=DEVICE)[:top_k] for _ in range(num_tokens)
    ]
    top_k_indices = torch.stack(top_k_indices)

    # Create logits with the uniform distribution.
    target_logits = torch.zeros((num_tokens, vocab_size), device=DEVICE)

    # Increment the logits for top-k indices, a little bit more than the other
    # ones. If the masking is effective, the non-topk indices will never be
    # sampled despite the small difference in logits.
    for i in range(num_tokens):
        target_logits[i, top_k_indices[i]] += 0.1

    # Create sampling metadata
    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE)
    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=temperature,
        top_k=torch.tensor([top_k] * batch_size, device=DEVICE, dtype=torch.int64),
    )

    _test_masked_logits(
        rejection_sampler,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        vocab_size=vocab_size,
        target_logits=target_logits,
        unmasked_indices=top_k_indices,
        sampling_metadata=sampling_metadata,
    )

@pytest.mark.parametrize("top_p", [0.5, 0.9, 0.99])
def test_top_p(rejection_sampler, top_p):
    """Test rejection sampling with top-p sampling"""
    vocab_size = 100
    batch_size = 100
    num_draft_tokens = 3
    num_tokens = batch_size * num_draft_tokens

    # Create logits with the uniform distribution.
    target_logits = torch.randn((num_tokens, vocab_size), device=DEVICE)
    temperature = torch.ones(batch_size, dtype=torch.float32, device=DEVICE)
    rescaled_logits = target_logits / temperature

    logits_sort, logits_idx = rescaled_logits.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum <= 1 - top_p
    # at least one
    top_p_mask[:, -1] = False

    # Get the top-p indices.
    top_p_indices = []
    for i in range(num_tokens):
        top_p_indices.append(logits_idx[i][~top_p_mask[i]].tolist())

    # Create sampling metadata
    sampling_metadata = create_sampling_metadata(
        all_greedy=False,
        temperature=temperature,
        top_p=torch.tensor([top_p] * batch_size, device=DEVICE, dtype=torch.float32),
    )

    _test_masked_logits(
        rejection_sampler,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        vocab_size=vocab_size,
        target_logits=target_logits,
        unmasked_indices=top_p_indices,
        sampling_metadata=sampling_metadata,
    )


########################### Tests for Logit Processors ###################
def test_frequency_penalties(rejection_sampler):
    """Test rejection sampling with frequency penalties"""
    spec_tokens = [[1, 1, 1], [], [1, 1, 1]]
    output_tokens = [[1, 1, 1, 1], [7], [1, 1, 1, 1]]  # 1, 7 and 1 are the bonus tokens

    num_requsts = len(spec_tokens)
    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[2], [3], [4]],
        spec_token_ids=spec_tokens,
        prompt_token_ids=torch.tensor([[5, 6, 7], [6, 7, 8], [7, 8, 9]], device=DEVICE),
        frequency_penalties=[1.5, 1.5, 0.7],
        presence_penalties=[0.0] * num_requsts,
        repetition_penalties=[1.0] * num_requsts,
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = SpecDecodeMetadata.make_dummy(
        spec_tokens, device=logits.device
    )
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )
    expected = torch.tensor(
        [[1, 15, -1, -1], [7, -1, -1, -1], [1, 1, 15, -1]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_bad_words(rejection_sampler):
    """Test rejection sampling with bad words constraints.

    This test applies bad words to non-consecutive requests (0 and 2, but not 1)
    to verify correct logit indexing when iterating over requests with bad words.
    """
    spec_tokens = [[1, 2, 3], [1, 15, 3], [1, 2, 3]]
    output_tokens = [[1, 2, 3, 4], [1, 15, 3, 4], [1, 2, 3, 4]]

    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[2], [3], [4]],
        spec_token_ids=spec_tokens,
        bad_words_token_ids={
            0: [[2]],
            # Request 1 has no bad words (to test non-consecutive request handling)
            2: [[2]],
        },
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )

    # Request 0: bad word [2] matches prefix, so token 2 is rejected -> 15
    # Request 1: no bad words, all tokens match -> [1, 15, 3, 4]
    # Request 2: bad word [2] matches prefix, so token 2 is rejected -> 15
    expected = torch.tensor(
        [[1, 15, -1, -1], [1, 15, 3, 4], [1, 15, -1, -1]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)


def test_allowed_token_ids(rejection_sampler):
    """Test rejection sampling with allowed token ids"""
    spec_tokens = [[1, 2, 10], [10, 5, 3], [7, 10, 12]]
    output_tokens = [[1, 2, 10, 5], [10, 5, 10, 5], [7, 10, 12, 5]]
    # Not allowed tokens:
    # 0: 0-4
    # 1: 1-5
    # 2: 2-6
    num_allowed_token_ids = 5

    # Use the token 15 as the sampler choose if a token rejected
    logits = create_logits_tensor(output_tokens, token_idx_to_override=15)

    batch_size = len(output_tokens)
    _, vocab_size = logits.size()
    mask = create_allowed_token_ids(
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_allowed_token_ids=num_allowed_token_ids,
        device=logits.device,
    )
    metadata = create_sampling_metadata(
        all_greedy=True,
        output_token_ids=[[], [], []],
        spec_token_ids=spec_tokens,
        allowed_token_ids_mask=mask,
    )
    bonus_token_tensor = torch.tensor(
        [output_tokens[i][-1] for i in range(len(output_tokens))], device=logits.device
    )
    spec_decode_metadata = create_spec_decode_metadata(spec_tokens, logits)
    mock_sampler_output(rejection_sampler, bonus_token_tensor)
    output = rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,
        logits=logits,
        sampling_metadata=metadata,
    )

    expected = torch.tensor(
        [[15, -1, -1, -1], [10, 5, 10, -1], [7, 10, 12, 5]],
        dtype=torch.int,
        device=logits.device,
    )
    assert torch.equal(output.sampled_token_ids, expected)
