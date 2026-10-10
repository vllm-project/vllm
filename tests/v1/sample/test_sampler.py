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
DEVICE_TYPE = current_platform.device_type
DEVICES = [
    f"{DEVICE_TYPE}:{i}"
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
    """Creates an output token list where each token occurs a distinct
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("presence_penalty", [-2.0, 2.0])
def test_sampler_presence_penalty(
    device: str, batch_size: int, presence_penalty: float
):
    """Test to verify that if presence penalty is enabled then tokens
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("frequency_penalty", [-2.0, 2.0])
def test_sampler_frequency_penalty(
    device: str, batch_size: int, frequency_penalty: float
):
    """Test to verify that if frequency penalty is enabled then tokens are
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("repetition_penalty", [0.1, 1.9])
def test_sampler_repetition_penalty(
    device: str, batch_size: int, repetition_penalty: float
):
    """Test to verify that when the repetition penalty is enabled, tokens
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
        assert sampling_metadata.prompt_token_ids is not None
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("num_allowed_token_ids", [0, 1, 2])
def test_sampler_allowed_token_ids(
    device: str, batch_size: int, num_allowed_token_ids: int
):
    """Test to verify that when the repetition penalty is enabled, tokens
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("bad_words_lengths", [(1,), (1, 3), (2, 2)])
def test_sampler_bad_words(
    device: str, batch_size: int, bad_words_lengths: tuple[int, ...]
):
    """Test to verify that when the bad words restriction is present, tokens
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


@pytest.mark.parametrize("device", DEVICES)
def test_no_valid_token_mixed_batch(device: str):
    """Test that a mixed batch (valid / all--inf / valid) correctly
    marks only the impossible row and neutralizes it for kernel safety,
    while valid rows sample normally.

    This tests the core invariant of the #57986 fix:
        all--inf rows → no_valid_token_mask=True, logits neutralized
        valid rows    → no_valid_token_mask=False, sampled normally
    """
    torch.set_default_device(device)
    batch_size = 3
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)

    # Row 0: valid — keep as-is
    # Row 1: all -inf — simulate the contradiction
    fake_logits[1, :] = float("-inf")
    # Row 2: valid — keep as-is

    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )

    sampler = Sampler()
    sampler_output = sampler.forward(fake_logits, sampling_metadata)

    no_valid_token_mask = sampler_output.no_valid_token_mask
    assert no_valid_token_mask is not None
    mask_cpu = no_valid_token_mask.cpu()

    # Row 0: valid → False
    assert not mask_cpu[0].item(), "Row 0 should have valid logits"
    # Row 1: all -inf → True
    assert mask_cpu[1].item(), "Row 1 should have no valid token"
    # Row 2: valid → False
    assert not mask_cpu[2].item(), "Row 2 should have valid logits"

    sampled = sampler_output.sampled_token_ids.cpu()
    # Rows 0 and 2 sampled normally (argmax of uniform logits → token 0)
    assert sampled[0, 0].item() == 0
    assert sampled[2, 0].item() == 0
    # Row 1 neutralized to [0.0, 0.0, ...]; argmax(0) also returns 0,
    # but this token is guaranteed discarded before becoming request output.
    # We assert it did produce *something* so downstream kernels don't break.
    assert sampled[1, 0].item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_no_valid_token_with_allowed_token_ids_and_bad_words(device: str):
    """Test that allowed_token_ids + bad_words contradiction correctly
    produces no_valid_token_mask=True at the sampler level.

    allowed_token_ids = [5]
    bad_words_token_ids = [[5]]  (single-token bad word)
    → every logit becomes -inf for that row
    """
    torch.set_default_device(device)
    batch_size = 3
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)

    # Row 0: no constraints → valid
    # Row 1: allowed only [5], bad words bans [5] → all -inf
    # Row 2: no constraints → valid

    # Build allowed_token_ids_mask: True = suppress (inverted)
    mask = torch.zeros(batch_size, VOCAB_SIZE, dtype=torch.bool, device=device)
    mask[1, :] = True
    mask[1, 5] = False  # only token 5 is allowed

    # Build bad_words_token_ids: single-token bad word [5]
    bad_words_token_ids = {1: [[5]]}
    output_token_ids: list[list[int]] = [
        np.random.randint(0, VOCAB_SIZE, size=NUM_OUTPUT_TOKENS).tolist()
        for _ in range(batch_size)
    ]
    prompt_token_ids: list[list[int]] = [
        np.random.randint(
            0, VOCAB_SIZE, size=np.random.randint(1, MAX_NUM_PROMPT_TOKENS)
        ).tolist()
        for _ in range(batch_size)
    ]
    prompt_tokens_tensor = _create_prompt_tokens_tensor(
        prompt_token_ids, VOCAB_SIZE, device
    )

    sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size,), 0.0),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=prompt_tokens_tensor,
        output_token_ids=output_token_ids,
        spec_token_ids=[[] for _ in range(batch_size)],
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        allowed_token_ids_mask=mask,
        bad_words_token_ids=bad_words_token_ids,
        logitsprocs=LogitsProcessors(),
    )

    sampler = Sampler()
    sampler_output = sampler.forward(fake_logits, sampling_metadata)

    no_valid_token_mask = sampler_output.no_valid_token_mask
    assert no_valid_token_mask is not None
    mask_cpu = no_valid_token_mask.cpu()

    # Row 0: valid → False
    assert not mask_cpu[0].item(), "Row 0 should be valid"
    # Row 1: contradiction → True
    assert mask_cpu[1].item(), "Row 1 should have no valid token"
    # Row 2: valid → False
    assert not mask_cpu[2].item(), "Row 2 should be valid"

    sampled = sampler_output.sampled_token_ids.cpu()
    # Rows 0 and 2 produce a valid token
    assert sampled[0, 0].item() >= 0
    assert sampled[2, 0].item() >= 0
    # Row 1 produces the neutralized token (0) which will be discarded
    assert sampled[1, 0].item() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_no_valid_token_nan_not_detected(device: str):
    """NaN rows must NOT be detected as no-valid-token.
    torch.isneginf does not match NaNs, so they are treated as
    non--inf values and left alone.
    """
    torch.set_default_device(device)
    batch_size = 3
    fake_logits = torch.full((batch_size, VOCAB_SIZE), 1e-2, dtype=torch.float)
    fake_logits[1, :] = float("nan")

    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )

    sampler = Sampler()
    sampler_output = sampler.forward(fake_logits, sampling_metadata)

    mask = sampler_output.no_valid_token_mask
    assert mask is not None
    mask_cpu = mask.cpu()
    assert not mask_cpu[0].item(), "Row 0 should not be invalid"
    assert not mask_cpu[1].item(), "NaN row should not be detected as invalid"
    assert not mask_cpu[2].item(), "Row 2 should not be invalid"


@pytest.mark.parametrize("device", DEVICES)
def test_no_valid_token_posinf_not_detected(device: str):
    """+inf rows must NOT be detected as no-valid-token."""
    torch.set_default_device(device)
    batch_size = 3
    fake_logits = torch.full((batch_size, VOCAB_SIZE), 1e-2, dtype=torch.float)
    fake_logits[1, :] = float("inf")

    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )

    sampler = Sampler()
    sampler_output = sampler.forward(fake_logits, sampling_metadata)

    mask = sampler_output.no_valid_token_mask
    assert mask is not None
    mask_cpu = mask.cpu()
    assert not mask_cpu[0].item()
    assert not mask_cpu[1].item(), "+inf row should not be detected as invalid"
    assert not mask_cpu[2].item()


@pytest.mark.parametrize("device", DEVICES)
def test_no_valid_token_mixed_finite_and_neginf(device: str):
    """Rows with a mix of -inf and finite values should NOT be detected."""
    torch.set_default_device(device)
    batch_size = 1
    fake_logits = torch.full((batch_size, VOCAB_SIZE), 1e-2, dtype=torch.float)
    fake_logits[0, :] = float("-inf")
    fake_logits[0, 5] = 1.0

    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )

    sampler = Sampler()
    sampler_output = sampler.forward(fake_logits, sampling_metadata)

    mask = sampler_output.no_valid_token_mask
    assert mask is not None
    assert not mask[0].item()


@pytest.mark.parametrize("device", DEVICES)
def test_no_valid_token_all_neginf_detected(device: str):
    """Only rows where EVERY logit is -inf should be detected."""
    torch.set_default_device(device)
    batch_size = 1
    fake_logits = torch.full((batch_size, VOCAB_SIZE), float("-inf"))

    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device)
    )

    sampler = Sampler()
    sampler_output = sampler.forward(fake_logits, sampling_metadata)

    mask = sampler_output.no_valid_token_mask
    assert mask is not None
    assert mask[0].item(), "All -inf row must be detected as invalid"
