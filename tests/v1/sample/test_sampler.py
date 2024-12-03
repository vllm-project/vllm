from typing import List, Set, Tuple

import numpy as np
import pytest
import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
MAX_NUM_PROMPT_TOKENS = 64


def _create_fake_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    fake_logits = torch.full((batch_size, vocab_size), 1e-2, dtype=torch.float)
    return fake_logits


def _create_default_sampling_metadata(
    num_output_tokens: int,
    batch_size: int,
    vocab_size: int,
) -> SamplingMetadata:
    output_token_ids: List[List[int]] = []
    prompt_token_ids: List[List[int]] = []
    for _ in range(batch_size):
        output_token_ids.append(
            np.random.randint(0, vocab_size, size=num_output_tokens).tolist())
        prompt_token_ids.append(
            np.random.randint(0,
                              vocab_size,
                              size=np.random.randint(
                                  1, MAX_NUM_PROMPT_TOKENS)).tolist())
    fake_sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size, ), 0.0),
        all_greedy=True,
        all_random=False,
        top_p=torch.empty(batch_size, ),
        top_k=torch.empty(batch_size, ),
        no_top_p=True,
        no_top_k=True,
        generators={},
        max_num_logprobs=VOCAB_SIZE,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        frequency_penalties=[0.0 for _ in range(batch_size)],
        presence_penalties=[0.0 for _ in range(batch_size)],
        repetition_penalties=[1.0 for _ in range(batch_size)],
        min_tokens=[],
        stop_token_ids=[],
    )
    return fake_sampling_metadata


def _create_min_token_penalty_dataset(
    num_output_tokens: int, batch_size: int, vocab_size: int,
    batch_indices_for_min_token_penalty: List[int]
) -> Tuple[List[int], List[Set[int]]]:
    """

    """
    stop_token_ids: List[Set[int]] = []
    min_tokens: List[int] = []
    for index in range(batch_size):
        if index in batch_indices_for_min_token_penalty:
            min_tokens.append(
                np.random.randint(num_output_tokens + 1,
                                  2 * num_output_tokens))
            stop_token_ids.append(
                set(
                    np.random.randint(0, vocab_size - 1)
                    for _ in range(np.random.randint(0, vocab_size))))

        else:
            min_tokens.append(np.random.randint(0, num_output_tokens))
            stop_token_ids.append(set())
    return (min_tokens, stop_token_ids)


def _create_weighted_output_token_list(
        batch_size: int,
        vocab_size: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Creates an output token list where each token occurs a distinct 
    number of times.

    For each batch, a random subset of token IDs is selected from the
    vocabulary. The selected tokens are then added to the output token
    list, each with a different frequency.

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            - The first element is the output token list, where each sublist 
              corresponds to a batch and contains tokens with weighted 
              frequencies.
            - The second element is a list of distinct token IDs for each
              batch, ordered by their frequency in the corresponding output
              list.
    """
    output_token_ids: List[List[int]] = []
    sorted_token_ids_in_output: List[List[int]] = []
    for _ in range(batch_size):
        distinct_token_ids = np.random.choice(vocab_size,
                                              size=np.random.randint(1, 10),
                                              replace=False).tolist()
        sorted_token_ids_in_output.append(distinct_token_ids)
        output_token_ids_for_batch = []
        for index, token_id in enumerate(distinct_token_ids):
            output_token_ids_for_batch.extend(
                [token_id for _ in range(index + 1)])
        output_token_ids.append(output_token_ids_for_batch)
    return (output_token_ids, sorted_token_ids_in_output)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
def test_sampler_min_tokens_penalty(device: str, batch_size: int):
    """
    Tests that if the number of output tokens is less than 
    SamplingParams.min_tokens then we will set the logits for
    the stop token ids to -inf.
    """
    torch.set_default_device(device)
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE)
    batch_indices_for_min_token_penalty = np.random.randint(
        0, batch_size - 1, size=np.random.randint(0, batch_size)).tolist()
    min_tokens, stop_token_ids = _create_min_token_penalty_dataset(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE,
        batch_indices_for_min_token_penalty)
    sampling_metadata.min_tokens = min_tokens
    sampling_metadata.stop_token_ids = stop_token_ids
    sampler = Sampler()
    sampler_output = sampler(fake_logits, sampling_metadata)
    for batch_idx in range(batch_size):
        for vocab in range(VOCAB_SIZE):
            # Verify that the logprobs for stop token ids is set
            # to -inf.
            logprob_index = torch.where(
                sampler_output.logprob_token_ids[batch_idx] ==
                vocab)[0].item()
            if vocab in stop_token_ids[batch_idx]:
                assert sampler_output.logprobs[batch_idx][
                    logprob_index] == -float("inf")
            else:
                assert sampler_output.logprobs[batch_idx][
                    logprob_index] != -float("inf")


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("presence_penalty", [-2.0, 2.0])
def test_sampler_presence_penalty(device: str, batch_size: int,
                                  presence_penalty: float):
    """
    Test to verify that if presence penalty is enabled then tokens
    are penalized as per their presence in the existing output.
    """
    torch.set_default_device(device)
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE)
    output_token_ids = sampling_metadata.output_token_ids
    sampling_metadata.presence_penalties = [
        presence_penalty for _ in range(batch_size)
    ]
    sampler = Sampler()
    sampler_output = sampler(fake_logits, sampling_metadata)
    for batch_idx in range(batch_size):
        # The logprobs in the SamplerOutput are arranged in descending order.
        # Since all tokens initially have the same logprobs, the non-penalized
        # tokens will appear at the beginning, while the penalized tokens
        #  will appear at the end of the list.
        penalized_token_id = sampler_output.logprob_token_ids[batch_idx][
            VOCAB_SIZE - 1]
        penalized_log_prod = sampler_output.logprobs[batch_idx][VOCAB_SIZE - 1]
        non_penalized_token_id = sampler_output.logprob_token_ids[batch_idx][0]
        non_penalized_log_prod = sampler_output.logprobs[batch_idx][0]
        assert non_penalized_log_prod > penalized_log_prod
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
def test_sampler_frequency_penalty(device: str, batch_size: int,
                                   frequency_penalty: float):
    """
    Test to verify that if frequency penalty is enabled then tokens are
    penalized as per their frequency of occurrence.
    """
    torch.set_default_device(device)
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE)
    sampling_metadata.frequency_penalties = [
        frequency_penalty for _ in range(batch_size)
    ]
    output_token_ids, sorted_token_ids_in_output = \
        _create_weighted_output_token_list(batch_size, VOCAB_SIZE)
    sampling_metadata.output_token_ids = output_token_ids
    sampler = Sampler()
    sampler_output = sampler(fake_logits, sampling_metadata)
    for batch_idx in range(batch_size):
        logprobs_token_ids = sampler_output.logprob_token_ids[batch_idx]
        non_penalized_token_id = logprobs_token_ids[0]
        penalized_token_id = logprobs_token_ids[VOCAB_SIZE - 1]
        distinct_sorted_token_ids_in_output = \
            sorted_token_ids_in_output[batch_idx]
        most_frequent_token_id = distinct_sorted_token_ids_in_output[
            len(distinct_sorted_token_ids_in_output) - 1]
        if frequency_penalty > 0:
            # If `frequency_penalty` is set to > 0, it indicates
            # a preference for new tokens over existing ones. Verify that the
            # non-penalized token ID is not present in the output, while the
            # most penalized token is the one that occurs most frequently in
            # the output.
            assert non_penalized_token_id \
                not in distinct_sorted_token_ids_in_output
            assert penalized_token_id == most_frequent_token_id
        elif frequency_penalty < 0:
            # If `frequency_penalty` is set to < 0, it indicates
            # a preference for existing tokens over new ones. Verify that the
            # non-penalized token ID is the one that occurs most frequently
            # in the output, while the penalized token ID is one that has not
            # yet appeared.
            assert non_penalized_token_id == most_frequent_token_id
            assert penalized_token_id \
                not in distinct_sorted_token_ids_in_output


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("repetition_penalty", [0.1, 1.9])
def test_sampler_repetition_penalty(device: str, batch_size: int,
                                    repetition_penalty: float):
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
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE)
    sampling_metadata.repetition_penalties = [
        repetition_penalty for _ in range(batch_size)
    ]
    sampler = Sampler()
    sampler_output = sampler(fake_logits, sampling_metadata)
    for batch_idx in range(batch_size):
        logprobs_token_ids = sampler_output.logprob_token_ids[batch_idx]
        non_penalized_token_id = logprobs_token_ids[0]
        penalized_token_id = logprobs_token_ids[VOCAB_SIZE - 1]
        prompt_tokens = sampling_metadata.prompt_token_ids[batch_idx]
        output_tokens = sampling_metadata.output_token_ids[batch_idx]
        if repetition_penalty > 1.0:
            # If `repetition_penalty` > 1.0, verify that the non-penalized
            # token ID has not been seen before, while the penalized token ID
            # exists either in the prompt or the output.
            assert (non_penalized_token_id not in prompt_tokens and \
                non_penalized_token_id not in output_tokens)
            assert (penalized_token_id  in prompt_tokens or \
                penalized_token_id in output_tokens)
        elif repetition_penalty < 1.0:
            # If `repetition_penalty` < 1.0, verify that the penalized
            # token ID has not been seen before, while the non-penalized
            # token ID exists either in the prompt or the output.
            assert (penalized_token_id not in prompt_tokens and \
                penalized_token_id not in output_tokens)
            assert (non_penalized_token_id  in prompt_tokens or \
                non_penalized_token_id in output_tokens)
