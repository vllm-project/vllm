# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest
import torch

from vllm.utils import make_tensor_with_pad
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
MAX_NUM_PROMPT_TOKENS = 64

DEFAULT_LOGIT_VALUE = 1e-2


def _create_fake_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    fake_logits = torch.full((batch_size, vocab_size),
                             DEFAULT_LOGIT_VALUE,
                             dtype=torch.float)
    return fake_logits


def _create_penalty_tensor(batch_size: int, penalty_value: float,
                           device: torch.device) -> torch.Tensor:
    return torch.full((batch_size, ),
                      fill_value=penalty_value,
                      dtype=torch.float,
                      device=device)


def _create_prompt_tokens_tensor(
    prompt_token_ids: List[List[int]],
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


def _create_default_sampling_metadata(
    num_output_tokens: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
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
        max_num_logprobs=0,
        prompt_token_ids=_create_prompt_tokens_tensor(prompt_token_ids,
                                                      vocab_size, device),
        output_token_ids=output_token_ids,
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        min_tokens=[],
        stop_token_ids=[],
        logits_processors=[None] * batch_size,
        prompt_token_ids_cpu=prompt_token_ids[:],
    )
    return fake_sampling_metadata


class IncreaseLogitProcessor:

    def __init__(self, target_token_id: int, incr_value: float):
        self.target_token_id = target_token_id
        self.incr_value = incr_value

    def __call__(self, output_token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        logits[self.target_token_id] += self.incr_value
        return logits


class IncreaseLogitProcessorWithPromptParams:

    def __init__(self, target_token_id: int, incr_value: float):
        self.target_token_id = target_token_id
        self.incr_value = incr_value

    def __call__(self, promot_token_ids: List[int],
                 output_token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        logits[self.target_token_id] += self.incr_value
        return logits


def validate_logits_min_max(logits: torch.Tensor, expected_min: float,
                            expected_max: float):
    assert abs(logits.min().item() - expected_min) < 1e-5, \
        f"Expected min logit to be {expected_min}, got {logits.min().item()}"

    assert abs(logits.max().item() - expected_max) < 1e-5, \
        f"Expected max logit to be {expected_max}, got {logits.max().item()}"


UNTOUCHED_VALIDATOR = lambda logits: validate_logits_min_max(
    logits, expected_min=DEFAULT_LOGIT_VALUE, expected_max=DEFAULT_LOGIT_VALUE)

LogitsProcessorType = Callable[[List[int], torch.Tensor], torch.Tensor]


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("processors_and_validator", [
    (
        None,
        UNTOUCHED_VALIDATOR,
    ),
    (
        [],
        UNTOUCHED_VALIDATOR,
    ),
    (
        [IncreaseLogitProcessor(target_token_id=1, incr_value=1.0)],
        lambda logits: validate_logits_min_max(
            logits,
            expected_min=DEFAULT_LOGIT_VALUE,
            expected_max=DEFAULT_LOGIT_VALUE + 1.0,
        ),
    ),
    (
        [
            IncreaseLogitProcessor(target_token_id=1, incr_value=1.0),
            IncreaseLogitProcessor(target_token_id=2, incr_value=2.0),
        ],
        lambda logits: validate_logits_min_max(
            logits,
            expected_min=DEFAULT_LOGIT_VALUE,
            expected_max=DEFAULT_LOGIT_VALUE + 2.0,
        ),
    ),
    (
        [
            IncreaseLogitProcessorWithPromptParams(
                target_token_id=1,
                incr_value=1.0,
            ),
            IncreaseLogitProcessorWithPromptParams(
                target_token_id=2,
                incr_value=2.0,
            ),
        ],
        lambda logits: validate_logits_min_max(
            logits,
            expected_min=DEFAULT_LOGIT_VALUE,
            expected_max=DEFAULT_LOGIT_VALUE + 2.0,
        ),
    ),
])
def test_sampler_logits_processors(
    device: str,
    batch_size: int,
    processors_and_validator: Tuple[
        # logits_processors
        Optional[List[LogitsProcessorType]],
        # validator
        Callable[[torch.Tensor], None],
    ],
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
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))

    processors = processors_and_validator[0]
    sampling_metadata.logits_processors = [processors] * batch_size

    # leave the last but non-first seq untouched
    if batch_size > 1:
        sampling_metadata.logits_processors[-1] = None

    sampler = Sampler()
    logits = sampler.apply_logits_processors(fake_logits, sampling_metadata)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        validator = processors_and_validator[1] \
            if batch_idx == 0 or batch_idx != batch_size - 1 \
            else UNTOUCHED_VALIDATOR
        validator(logits[batch_idx])
