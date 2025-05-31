# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.v1.sample.logits_processor import (BatchUpdate,
                                             LogitBiasLogitsProcessor,
                                             LogitsProcessor,
                                             MinPLogitsProcessor,
                                             MinTokensLogitsProcessor)
from vllm.v1.sample.metadata import SamplingMetadata

BatchAddType = Sequence[tuple[int, SamplingParams, list[int]]]

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


def _create_penalty_tensor(batch_size: int, penalty_value: float,
                           device: torch.device) -> torch.Tensor:
    return torch.full((batch_size, ),
                      fill_value=penalty_value,
                      dtype=torch.float,
                      device=device)


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


def _create_logit_bias(
    batch_size: int,
    vocab_size: int,
    bias_value: float,
) -> list[Optional[dict[int, float]]]:
    res: list[Optional[dict[int, float]]] = []
    for i in range(batch_size):
        logit_bias = {min(i, vocab_size - 1): bias_value}
        res.append(logit_bias)
    return res


def _create_allowed_token_ids(
    batch_size: int,
    vocab_size: int,
    num_allowed_token_ids: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    mask: Optional[torch.Tensor] = None
    for i in range(batch_size):
        if i % 2 == 1:
            continue
        if mask is None:
            mask = torch.zeros((batch_size, vocab_size),
                               dtype=torch.bool,
                               device=device)
        start = min(i, vocab_size - 1)
        end = min(i + num_allowed_token_ids, vocab_size - 1)
        mask[i, start:end] = True
    return mask


def _create_bad_words_token_ids(
        batch_size: int, vocab_size: int,
        bad_words_lengths: list[tuple[int]]) -> dict[int, list[list[int]]]:
    bad_words_token_ids = {}
    for batch_idx in range(batch_size):
        token_ids_single_batch = []
        for bad_words_length in bad_words_lengths:
            token_ids = np.random.choice(vocab_size,
                                         size=bad_words_length,
                                         replace=True).tolist()
            token_ids_single_batch.append(token_ids)
        bad_words_token_ids[batch_idx] = token_ids_single_batch
    if batch_size >= 2:
        # Test no bad_words for some batch
        no_bad_words_batch_idx = np.random.choice(batch_size)
        bad_words_token_ids.pop(no_bad_words_batch_idx, None)
    return bad_words_token_ids


def _update_output_token_ids_for_bad_words(
        metadata: SamplingMetadata, vocab_size: int) -> dict[int, list[int]]:
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
                    output_token_ids[-prefix_length:] = bad_word_token_ids[:-1]
                    bad_words_last_token.append(bad_word_token_ids[-1])
                    break  # Maximum one update to output_token_ids
                else:  # Make sure no accidental match to bad words
                    output_token_ids[-1] = (bad_word_token_ids[-2] +
                                            1) % vocab_size
        bad_words_last_tokens[batch_idx] = bad_words_last_token
    return bad_words_last_tokens


def _create_default_sampling_metadata(
    num_output_tokens: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[SamplingMetadata, dict[str, LogitsProcessor]]:
    output_token_ids: list[list[int]] = []
    prompt_token_ids: list[list[int]] = []
    for _ in range(batch_size):
        output_token_ids.append(
            np.random.randint(0, vocab_size, size=num_output_tokens).tolist())
        prompt_token_ids.append(
            np.random.randint(0,
                              vocab_size,
                              size=np.random.randint(
                                  1, MAX_NUM_PROMPT_TOKENS)).tolist())
    min_tokens_logitproc = MinTokensLogitsProcessor(
        pin_memory=PIN_MEMORY_AVAILABLE, device=device)
    logit_bias_logitproc = LogitBiasLogitsProcessor(
        pin_memory=PIN_MEMORY_AVAILABLE, device=device)
    min_p_logitproc = MinPLogitsProcessor(
        pin_memory=PIN_MEMORY_AVAILABLE,
        device=device,
        # +1 for temporary swap space
        max_num_reqs=MAX_NUM_REQS + 1)
    fake_sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size, ), 0.0),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=_create_prompt_tokens_tensor(prompt_token_ids,
                                                      vocab_size, device),
        output_token_ids=output_token_ids,
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logits_procs=[
            min_tokens_logitproc,
            logit_bias_logitproc,
        ],
        nongreedy_logits_procs=[min_p_logitproc])
    return fake_sampling_metadata, {
        "min_tokens": min_tokens_logitproc,
        "logit_bias": logit_bias_logitproc,
        "min_p": min_p_logitproc
    }


def _fake_apply_greedy_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Imitate greedy logit processor application in engine
    core"""
    for processor in sampling_metadata.logits_procs:
        logits = processor.apply(logits)
    return logits


def _fake_apply_nongreedy_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Imitate non-greedy logit processoed application in engine
    core"""
    for processor in sampling_metadata.nongreedy_logits_procs:
        logits = processor.apply(logits)
    return logits


def _generate_min_token_penalties_and_stop_tokens(
    num_output_tokens: int, batch_size: int, vocab_size: int,
    batch_indices_for_min_token_penalty: list[int]
) -> dict[int, tuple[int, set[int]]]:
    """
    Generates and returns a dict of minimum token penalties and
    corresponding stop token IDs (`min_tokens`, `stop_token_ids`) for each
    batch.

    If a batch index is included in `batch_indices_for_min_token_penalty`,
    a higher `min_tokens` value is assigned (within a randomized range),
    and a random set of stop token IDs is created. Otherwise, a lower
    `min_tokens` value is assigned, and the stop token IDs set is empty.
    """
    min_tokens: dict[int, tuple[int, set[int]]] = {}
    for index in range(batch_size):
        if index in batch_indices_for_min_token_penalty:
            min_tokens[index] = (
                np.random.randint(num_output_tokens + 1,
                                  2 * num_output_tokens),
                set(
                    np.random.randint(0, vocab_size - 1)
                    for _ in range(np.random.randint(0, vocab_size))))
        else:
            min_tokens[index] = (np.random.randint(0,
                                                   num_output_tokens), set())
    return min_tokens


def _create_weighted_output_token_list(
        batch_size: int,
        vocab_size: int) -> tuple[list[list[int]], list[list[int]]]:
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
        distinct_token_ids = np.random.choice(vocab_size,
                                              size=np.random.randint(1, 10),
                                              replace=False).tolist()
        sorted_token_ids_in_output.append(distinct_token_ids)
        output_token_ids_for_batch = []
        for index, token_id in enumerate(distinct_token_ids):
            output_token_ids_for_batch.extend(
                [token_id for _ in range(index + 1)])
        output_token_ids.append(output_token_ids_for_batch)
    return output_token_ids, sorted_token_ids_in_output


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("bias_value", [-0.1, 1.2])
def test_logit_bias(device: str, batch_size: int, bias_value: float):
    """
    Test to verify that when the repetition penalty is enabled, tokens
    are penalized based on their presence in the prompt or the existing
    output.
    """
    torch.set_default_device(device)
    # Create fake logits where each token is assigned the same
    # logit value.
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata, logitproc_dict = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    logit_bias_logitproc = logitproc_dict["logit_bias"]
    # Create batch update where each request demands a
    # different logit bias
    logit_bias_list = _create_logit_bias(
        batch_size=batch_size,
        vocab_size=VOCAB_SIZE,
        bias_value=bias_value,
    )
    added: BatchAddType = [
        (rdx, SamplingParams(logit_bias=logit_bias_list[rdx]), [])
        for rdx in range(batch_size)
    ]
    batch_update = BatchUpdate(
        removed=[],
        moved=[],
        added=added,
        batch_size=batch_size,
    )
    # Register batch update with logit processor
    logit_bias_logitproc.update_states(batch_update)
    # Emulate application of greedy logits processors in engine
    logits = _fake_apply_greedy_logits_processors(fake_logits,
                                                  sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        logits_for_req = logits[batch_idx]
        biased_index = min(batch_idx, VOCAB_SIZE - 1)
        for token_id in range(VOCAB_SIZE):
            if biased_index == token_id:
                assert logits_for_req[token_id] == pytest.approx(bias_value +
                                                                 1e-2)
            else:
                assert logits_for_req[token_id] == pytest.approx(1e-2)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("min_p", [0.0, 0.1])
def test_min_p(device: str, batch_size: int, min_p: float):
    """
    Tests that when min_p is applied, tokens with probability below 
    min_p * max_prob are masked with -inf.
    """
    torch.set_default_device(device)
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)

    # Create one dominant token per batch
    for i in range(batch_size):
        fake_logits[i, 0] = 10.0  # High logit for first token
        fake_logits[i, 1:] = 1e-2  # Others remain low

    sampling_metadata, logitproc_dict = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))

    min_p_logitproc = logitproc_dict["min_p"]
    # Create batch update where each request demands
    # the same min_p value
    added: BatchAddType = [(rdx, SamplingParams(min_p=min_p), [])
                           for rdx in range(batch_size)]
    batch_update = BatchUpdate(
        removed=[],
        moved=[],
        added=added,
        batch_size=batch_size,
    )
    # Register batch update with logit processor
    min_p_logitproc.update_states(batch_update)
    # Emulate application of non-greedy logits processors in engine
    logits = _fake_apply_nongreedy_logits_processors(fake_logits,
                                                     sampling_metadata)
    logits = logits.cpu()

    for batch_idx in range(batch_size):
        for token_id in range(VOCAB_SIZE):
            if token_id == 0:
                # Dominant token should always be unmasked
                assert logits[batch_idx][token_id] != -float("inf")
            else:
                if min_p > 0.0:
                    # Non-dominant tokens should be masked when min_p > 0
                    assert logits[batch_idx][token_id] == -float("inf")
                else:
                    # No masking when min_p is 0
                    assert logits[batch_idx][token_id] != -float("inf")


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
def test_min_tokens_penalty(device: str, batch_size: int):
    """
    Tests that if the number of output tokens is less than
    SamplingParams.min_tokens then we will set the logits for
    the stop token ids to -inf.
    """
    torch.set_default_device(device)
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    sampling_metadata, logitproc_dict = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    min_tokens_logitproc = logitproc_dict["min_tokens"]
    batch_indices_for_min_token_penalty = np.random.randint(
        0, batch_size - 1, size=np.random.randint(0, batch_size)).tolist()
    min_tokens_dict = _generate_min_token_penalties_and_stop_tokens(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE,
        batch_indices_for_min_token_penalty)

    # Create batch update where each request demands
    # a different min_tokens value
    added: BatchAddType = [(rdx,
                            SamplingParams(min_tokens=min_tokens_dict[rdx][0],
                                           max_tokens=None), [])
                           for rdx in range(batch_size)]
    batch_update = BatchUpdate(
        removed=[],
        moved=[],
        added=added,
        batch_size=batch_size,
    )
    # Register batch update with logit processor
    min_tokens_logitproc.update_states(batch_update)
    # Emulate application of greedy logits processors in engine
    logits = _fake_apply_greedy_logits_processors(fake_logits,
                                                  sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        for token_id in range(VOCAB_SIZE):
            _, stop_token_ids = min_tokens_dict.get(batch_idx, (0, set()))
            if token_id in stop_token_ids:
                assert logits[batch_idx][token_id] == -float("inf")
            else:
                assert logits[batch_idx][token_id] != -float("inf")
