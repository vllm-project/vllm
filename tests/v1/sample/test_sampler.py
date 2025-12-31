# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from tests.v1.sample.utils import create_allowed_token_ids
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

PIN_MEMORY_AVAILABLE = is_pin_memory_available()
MAX_NUM_REQS = 256
VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
CUDA_DEVICES = [
    f"{current_platform.device_type}:{i}"
    for i in range(1 if current_platform.device_count() == 1 else 2)
]
MAX_NUM_PROMPT_TOKENS = 64


def _create_fake_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    fake_logits = torch.full((batch_size, vocab_size), 1e-2, dtype=torch.float)
    return fake_logits


def _create_penalty_tensor(
    batch_size: int, penalty_value: float, device: torch.device
) -> torch.Tensor:
    return torch.full(
        (batch_size,), fill_value=penalty_value, dtype=torch.float, device=device
    )


def _create_prompt_tokens_tensor(
    prompt_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    return make_tensor_with_pad(
        prompt_token_ids,
        pad=vocab_size,
        device=device,
        dtype=torch.int64,
        pin_memory=False,
    )


def _create_bad_words_token_ids(
    batch_size: int,
    vocab_size: int,
    bad_words_lengths: tuple[int, ...],
) -> dict[int, list[list[int]]]:
    bad_words_token_ids = {}
    for batch_idx in range(batch_size):
        token_ids_single_batch = []
        for bad_words_length in bad_words_lengths:
            token_ids = np.random.choice(
                vocab_size, size=bad_words_length, replace=True
            ).tolist()
            token_ids_single_batch.append(token_ids)
        bad_words_token_ids[batch_idx] = token_ids_single_batch
    if batch_size >= 2:
        # Test no bad_words for some batch
        no_bad_words_batch_idx = np.random.choice(batch_size)
        bad_words_token_ids.pop(no_bad_words_batch_idx, None)
    return bad_words_token_ids


# Returns all last tokens of bad word sequences that share the same prefix
# as `given_prefix` (excluding the last token).
def _collect_suffixes_with_same_prefix(
    given_prefix: list[int], bad_words_token_ids: list[list[int]]
) -> list[int]:
    return [bwt[-1] for bwt in bad_words_token_ids if bwt[:-1] == given_prefix]


# generate a valid token id that is not in bad_words_token_ids
def _generate_valid_token_id(
    bad_words_token_ids: list[list[int]], vocab_size: int
) -> int:
    forbidden_start_tokens = set()
    for bad_word in bad_words_token_ids:
        forbidden_start_tokens.add(bad_word[0])
    # Get a safe token that's not in forbidden starts
    safe_token_candidates = list(set(range(vocab_size)) - forbidden_start_tokens)
    # Pick a random safe token
    return np.random.choice(safe_token_candidates)


def _update_output_token_ids_for_bad_words(
    metadata: SamplingMetadata, vocab_size: int
) -> dict[int, list[int]]:
    bad_words_last_tokens = {}
    for batch_idx, bad_words_token_ids in metadata.bad_words_token_ids.items():
        output_token_ids = metadata.output_token_ids[batch_idx]
        bad_words_last_token: list[int] = []
        for i, bad_word_token_ids in enumerate(bad_words_token_ids):
            if len(bad_word_token_ids) == 1:
                # Single token id always affects logits
                bad_words_last_token.append(bad_word_token_ids[0])
            else:
                prefix_length = len(bad_word_token_ids) - 1
                has_bad_words = np.random.choice([True, False])
                if has_bad_words:
                    prefix = bad_word_token_ids[:-1]
                    output_token_ids[-prefix_length:] = prefix
                    # Collect all last tokens from other bad words
                    # that share this prefix
                    bad_words_last_token.extend(
                        _collect_suffixes_with_same_prefix(prefix, bad_words_token_ids)
                    )
                    break  # Maximum one update to output_token_ids
                else:  # Make sure no accidental match to bad words
                    output_token_ids[-1] = _generate_valid_token_id(
                        bad_words_token_ids, vocab_size
                    )
        bad_words_last_tokens[batch_idx] = bad_words_last_token
    return bad_words_last_tokens


def _create_default_sampling_metadata(
    num_output_tokens: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> SamplingMetadata:
    output_token_ids: list[list[int]] = []
    prompt_token_ids: list[list[int]] = []
    for _ in range(batch_size):
        output_token_ids.append(
            np.random.randint(0, vocab_size, size=num_output_tokens).tolist()
        )
        prompt_token_ids.append(
            np.random.randint(
                0, vocab_size, size=np.random.randint(1, MAX_NUM_PROMPT_TOKENS)
            ).tolist()
        )
    fake_sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size,), 0.0),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=_create_prompt_tokens_tensor(
            prompt_token_ids, vocab_size, device
        ),
        output_token_ids=output_token_ids,
        spec_token_ids=[[] for _ in range(batch_size)],
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    return fake_sampling_metadata


def _create_weighted_output_token_list(
    batch_size: int, vocab_size: int
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Creates an output token list where each token occurs a distinct
    number of times.

    For each batch, a random subset of token IDs is selected from the
    vocabulary. The selected tokens are then added to the output token
    list, each with a different frequency.

    Returns:
        tuple[list[list[int]], list[list[int]]]:
            - The first element is the output token list, where each sublist
              corresponds to a batch and contains tokens with weighted
              frequencies.
            - The second element is a list of distinct token IDs for each
              batch, ordered by their frequency in the corresponding output
              list.
    """
    output_token_ids: list[list[int]] = []
    sorted_token_ids_in_output: list[list[int]] = []
    for _ in range(batch_size):
        distinct_token_ids = np.random.choice(
            vocab_size, size=np.random.randint(1, 10), replace=False
        ).tolist()
        sorted_token_ids_in_output.append(distinct_token_ids)
        output_token_ids_for_batch = []
        for index, token_id in enumerate(distinct_token_ids):
            output_token_ids_for_batch.extend([token_id for _ in range(index + 1)])
        output_token_ids.append(output_token_ids_for_batch)
    return output_token_ids, sorted_token_ids_in_output


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("presence_penalty", [-2.0, 2.0])
def test_sampler_presence_penalty(
    device: str, batch_size: int, presence_penalty: float
):
    """
    Test to verify that if presence penalty is enabled then tokens
    are penalized as per their presence in the existing output.
    """
    torch.set_default_device(device)
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )
    output_token_ids = sampling_metadata.output_token_ids
    sampling_metadata.presence_penalties = _create_penalty_tensor(
        batch_size, presence_penalty, torch.device(device)
    )
    sampling_metadata.no_penalties = False
    sampler = Sampler()
    logits = sampler.apply_penalties(
        fake_logits, sampling_metadata, sampling_metadata.output_token_ids
    )
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        # Since all tokens initially have the same logits, the non-penalized
        # token ID will be the one with the highest logit value, while the
        # penalized token ID will be the one with the lowest logit value.
        non_penalized_token_id = logits[batch_idx].argmax().item()
        penalized_token_id = logits[batch_idx].argmin().item()
        if presence_penalty > 0:
            # If `presence_penalty` is set to a value greater than 0, it
            # indicates a preference for new tokens over those already
            # present in the output.
            # Verify that the penalized token ID exists in the output, while the
            # non-penalized token ID does not.
            assert penalized_token_id in output_token_ids[batch_idx]
            assert non_penalized_token_id not in output_token_ids[batch_idx]
        elif presence_penalty < 0:
            # If `presence_penalty` is set to a value less than 0, it indicates
            # a preference for existing tokens over new ones. Verify that the
            # non-penalized token ID exists in the output, while the penalized
            # token ID does not.
            assert non_penalized_token_id in output_token_ids[batch_idx]
            assert penalized_token_id not in output_token_ids[batch_idx]


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("frequency_penalty", [-2.0, 2.0])
def test_sampler_frequency_penalty(
    device: str, batch_size: int, frequency_penalty: float
):
    """
    Test to verify that if frequency penalty is enabled then tokens are
    penalized as per their frequency of occurrence.
    """
    torch.set_default_device(device)
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )
    sampling_metadata.frequency_penalties = _create_penalty_tensor(
        batch_size, frequency_penalty, torch.device(device)
    )
    output_token_ids, sorted_token_ids_in_output = _create_weighted_output_token_list(
        batch_size,
        VOCAB_SIZE,
    )
    sampling_metadata.output_token_ids = output_token_ids
    sampling_metadata.no_penalties = False
    sampler = Sampler()
    logits = sampler.apply_penalties(
        fake_logits, sampling_metadata, sampling_metadata.output_token_ids
    )
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        non_penalized_token_id = logits[batch_idx].argmax().item()
        penalized_token_id = logits[batch_idx].argmin().item()
        distinct_sorted_token_ids_in_output = sorted_token_ids_in_output[batch_idx]
        most_frequent_token_id = distinct_sorted_token_ids_in_output[
            len(distinct_sorted_token_ids_in_output) - 1
        ]
        if frequency_penalty > 0:
            # If `frequency_penalty` is set to > 0, it indicates
            # a preference for new tokens over existing ones. Verify that the
            # non-penalized token ID is not present in the output, while the
            # most penalized token is the one that occurs most frequently in
            # the output.
            assert non_penalized_token_id not in distinct_sorted_token_ids_in_output
            assert penalized_token_id == most_frequent_token_id
        elif frequency_penalty < 0:
            # If `frequency_penalty` is set to < 0, it indicates
            # a preference for existing tokens over new ones. Verify that the
            # non-penalized token ID is the one that occurs most frequently
            # in the output, while the penalized token ID is one that has not
            # yet appeared.
            assert non_penalized_token_id == most_frequent_token_id
            assert penalized_token_id not in distinct_sorted_token_ids_in_output


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("repetition_penalty", [0.1, 1.9])
def test_sampler_repetition_penalty(
    device: str, batch_size: int, repetition_penalty: float
):
    """
    Test to verify that when the repetition penalty is enabled, tokens
    are penalized based on their presence in the prompt or the existing
    output.
    """
    torch.set_default_device(device)
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )
    sampling_metadata.repetition_penalties = _create_penalty_tensor(
        batch_size, repetition_penalty, torch.device(device)
    )
    sampling_metadata.no_penalties = False
    sampler = Sampler()
    logits = sampler.apply_penalties(
        fake_logits, sampling_metadata, sampling_metadata.output_token_ids
    )
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        non_penalized_token_id = logits[batch_idx].argmax().item()
        penalized_token_id = logits[batch_idx].argmin().item()
        prompt_tokens = sampling_metadata.prompt_token_ids[batch_idx][:].tolist()
        output_tokens = sampling_metadata.output_token_ids[batch_idx]
        if repetition_penalty > 1.0:
            # If `repetition_penalty` > 1.0, verify that the non-penalized
            # token ID has not been seen before, while the penalized token ID
            # exists either in the prompt or the output.
            assert (
                non_penalized_token_id not in prompt_tokens
                and non_penalized_token_id not in output_tokens
            )
            assert (
                penalized_token_id in prompt_tokens
                or penalized_token_id in output_tokens
            )
        elif repetition_penalty < 1.0:
            # If `repetition_penalty` < 1.0, verify that the penalized
            # token ID has not been seen before, while the non-penalized
            # token ID exists either in the prompt or the output.
            assert (
                penalized_token_id not in prompt_tokens
                and penalized_token_id not in output_tokens
            )
            assert (
                non_penalized_token_id in prompt_tokens
                or non_penalized_token_id in output_tokens
            )


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("num_allowed_token_ids", [0, 1, 2])
def test_sampler_allowed_token_ids(
    device: str, batch_size: int, num_allowed_token_ids: int
):
    """
    Test to verify that when the repetition penalty is enabled, tokens
    are penalized based on their presence in the prompt or the existing
    output.
    """
    torch.set_default_device(device)
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )
    mask = create_allowed_token_ids(
        batch_size=batch_size,
        vocab_size=VOCAB_SIZE,
        num_allowed_token_ids=num_allowed_token_ids,
        device=device,
    )
    sampling_metadata.allowed_token_ids_mask = mask
    sampler = Sampler()
    logits = sampler.apply_logits_processors(
        fake_logits, sampling_metadata, predict_bonus_token=False
    )
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        logits_for_req = logits[batch_idx]
        if batch_idx % 2 == 1:
            assert torch.all(logits_for_req != -float("inf"))
            continue
        for token_id in range(VOCAB_SIZE):
            start = min(batch_idx, VOCAB_SIZE - 1)
            end = min(batch_idx + num_allowed_token_ids, VOCAB_SIZE - 1)
            if token_id >= start and token_id < end:
                assert logits_for_req[token_id] == -float("inf"), (
                    f"{batch_idx}, {token_id}"
                )
            else:
                assert logits_for_req[token_id] != -float("inf")


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("bad_words_lengths", [(1,), (1, 3), (2, 2)])
def test_sampler_bad_words(
    device: str, batch_size: int, bad_words_lengths: tuple[int, ...]
):
    """
    Test to verify that when the bad words restriction is present, tokens
    are penalized based on their match with the bad words.
    """
    torch.set_default_device(device)
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )
    sampling_metadata.bad_words_token_ids = _create_bad_words_token_ids(
        batch_size, VOCAB_SIZE, bad_words_lengths
    )
    bad_words_last_tokens = _update_output_token_ids_for_bad_words(
        sampling_metadata, VOCAB_SIZE
    )
    sampler = Sampler()
    logits = sampler.apply_logits_processors(
        fake_logits, sampling_metadata, predict_bonus_token=False
    )
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        logits_for_req = logits[batch_idx]
        for token_id in range(VOCAB_SIZE):
            if (
                batch_idx in bad_words_last_tokens
                and token_id in bad_words_last_tokens[batch_idx]
            ):
                assert logits_for_req[token_id] == -float("inf")
            else:
                assert logits_for_req[token_id] != -float("inf")


# ==============================================================================
# Tests for compute_tracked_logprobs
# ==============================================================================


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("num_tracked_tokens", [1, 5, 20])
def test_compute_tracked_logprobs_from_logits(
    device: str, batch_size: int, num_tracked_tokens: int
):
    """
    Test computing tracked logprobs from raw logits (is_logprobs=False).

    Verifies that:
    1. Output shape is correct [batch_size, num_tracked_tokens]
    2. Logprobs are computed correctly via log_softmax
    3. Token IDs are preserved in the output
    """
    torch.set_default_device(device)

    # Create random logits
    logits = torch.randn(batch_size, VOCAB_SIZE)

    # Create track_token_ids (sorted, unique)
    track_ids = torch.tensor(
        sorted(
            np.random.choice(VOCAB_SIZE, num_tracked_tokens, replace=False).tolist()
        ),
        device=device,
        dtype=torch.int64,
    )

    # Compute tracked logprobs
    result = Sampler.compute_tracked_logprobs(logits, track_ids, is_logprobs=False)

    # Verify shape
    assert result.logprobs.shape == (batch_size, num_tracked_tokens)
    assert result.token_ids.shape == (num_tracked_tokens,)

    # Verify values match manual extraction
    expected_logprobs = logits.log_softmax(dim=-1)[:, track_ids]
    torch.testing.assert_close(result.logprobs, expected_logprobs)

    # Verify token_ids are preserved
    torch.testing.assert_close(result.token_ids, track_ids)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_compute_tracked_logprobs_from_logprobs(device: str, batch_size: int):
    """
    Test computing tracked logprobs when input is already log probabilities.

    When is_logprobs=True, the function should NOT apply log_softmax again.
    """
    torch.set_default_device(device)
    num_tracked_tokens = 10

    # Create log probabilities (already normalized)
    logits = torch.randn(batch_size, VOCAB_SIZE)
    logprobs = logits.log_softmax(dim=-1)

    # Create track_token_ids
    track_ids = torch.tensor(
        sorted(
            np.random.choice(VOCAB_SIZE, num_tracked_tokens, replace=False).tolist()
        ),
        device=device,
        dtype=torch.int64,
    )

    # Compute tracked logprobs with is_logprobs=True
    result = Sampler.compute_tracked_logprobs(logprobs, track_ids, is_logprobs=True)

    # Should directly index without additional log_softmax
    expected = logprobs[:, track_ids]
    torch.testing.assert_close(result.logprobs, expected)


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_compute_tracked_logprobs_single_token(device: str):
    """Test tracking a single token ID."""
    torch.set_default_device(device)
    batch_size = 4

    logits = torch.randn(batch_size, VOCAB_SIZE)
    track_ids = torch.tensor([42], device=device, dtype=torch.int64)

    result = Sampler.compute_tracked_logprobs(logits, track_ids, is_logprobs=False)

    assert result.logprobs.shape == (batch_size, 1)
    assert result.token_ids.shape == (1,)

    expected = logits.log_softmax(dim=-1)[:, 42:43]
    torch.testing.assert_close(result.logprobs, expected)


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_compute_tracked_logprobs_boundary_tokens(device: str):
    """Test tracking tokens at vocabulary boundaries (0 and vocab_size-1)."""
    torch.set_default_device(device)
    batch_size = 4

    logits = torch.randn(batch_size, VOCAB_SIZE)

    # Track first and last token IDs
    track_ids = torch.tensor([0, VOCAB_SIZE - 1], device=device, dtype=torch.int64)

    result = Sampler.compute_tracked_logprobs(logits, track_ids, is_logprobs=False)

    assert result.logprobs.shape == (batch_size, 2)

    expected_logprobs = logits.log_softmax(dim=-1)
    torch.testing.assert_close(result.logprobs[:, 0], expected_logprobs[:, 0])
    torch.testing.assert_close(
        result.logprobs[:, 1], expected_logprobs[:, VOCAB_SIZE - 1]
    )


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_compute_tracked_logprobs_many_tokens(device: str):
    """Test tracking many tokens (e.g., 100-class classification)."""
    torch.set_default_device(device)
    batch_size = 8
    num_tracked = 100

    logits = torch.randn(batch_size, VOCAB_SIZE)
    track_ids = torch.tensor(
        sorted(np.random.choice(VOCAB_SIZE, num_tracked, replace=False).tolist()),
        device=device,
        dtype=torch.int64,
    )

    result = Sampler.compute_tracked_logprobs(logits, track_ids, is_logprobs=False)

    assert result.logprobs.shape == (batch_size, num_tracked)
    assert len(result.token_ids) == num_tracked


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_compute_tracked_logprobs_values_are_valid(device: str):
    """Test that all returned logprobs are valid (not NaN, not positive)."""
    torch.set_default_device(device)
    batch_size = 8
    num_tracked = 20

    logits = torch.randn(batch_size, VOCAB_SIZE)
    track_ids = torch.tensor(list(range(num_tracked)), device=device, dtype=torch.int64)

    result = Sampler.compute_tracked_logprobs(logits, track_ids, is_logprobs=False)

    # Check no NaN values
    assert not torch.isnan(result.logprobs).any(), "Found NaN in tracked logprobs"

    # Check all values are <= 0 (log probabilities)
    assert (result.logprobs <= 0).all(), "Log probabilities should be <= 0"


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_compute_tracked_logprobs_deterministic(device: str):
    """Test that the function is deterministic."""
    torch.set_default_device(device)
    batch_size = 4

    logits = torch.randn(batch_size, VOCAB_SIZE)
    track_ids = torch.tensor([10, 20, 30], device=device, dtype=torch.int64)

    result1 = Sampler.compute_tracked_logprobs(logits, track_ids, is_logprobs=False)
    result2 = Sampler.compute_tracked_logprobs(logits, track_ids, is_logprobs=False)

    torch.testing.assert_close(result1.logprobs, result2.logprobs)
    torch.testing.assert_close(result1.token_ids, result2.token_ids)
