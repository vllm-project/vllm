# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Callable, Sequence
from typing import NamedTuple, Optional

import numpy as np
import pytest
import torch

from tests.v1.sample.utils import (LogitsprocsTestFakes, create_fake_logits,
                                   create_penalty_tensor,
                                   create_prompt_tokens_tensor,
                                   fake_apply_logits_processors,
                                   fake_update_logitsprocs_state)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available
from vllm.v1.sample.logits_processor import AddedRequestType, BatchUpdate
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.utils import (STR_LOGITS_BIAS_LOGITPROC_ID,
                                  STR_MIN_P_LOGITPROC_ID,
                                  STR_MIN_TOKENS_LOGITPROC_ID,
                                  STR_NO_LOGITPROC,
                                  init_hard_coded_logitsprocs)

PIN_MEMORY_AVAILABLE = is_pin_memory_available()
MAX_NUM_REQS = 256
VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
CUDA_DEVICES = [
    f"{current_platform.device_type}:{i}"
    for i in range(1 if current_platform.device_count() == 1 else 2)
]
MAX_NUM_PROMPT_TOKENS = 64
MIN_TOKENS_LEN_THRESHOLD = 5
REQS_PER_LOGITPROC = 10


class LogitsProcsRequestParams:
    """Encapsulates key params for a single request in a batch.
    
    Params can be customized based on the enabled logitproc
    """
    batch_index: int
    logitproc_id: str  # Logitproc enabled, specified by str id
    out_tokens: list[int]  # Output tokens required for min tokens test
    params: SamplingParams  # Settings customized for logitproc

    def __init__(self, batch_index: int, logitproc_id: str):
        self.batch_index = batch_index
        self.logitproc_id = logitproc_id
        # Number of output tokens is randomly 0 or twice the min-tokens
        # threshold which will be used in testing. Output token values
        # don't matter *for these tests* so use 0 as a dummy value
        self.out_tokens = ([0] *
                           (MIN_TOKENS_LEN_THRESHOLD * random.randint(0, 2)))
        self.params = _sampling_params_from_logitproc(logitproc_id)

    def __str__(self):
        """For debugging"""
        summ = ', '.join(f'{k}={v}' for k, v in vars(self).items())
        return f"MyClass({summ})"


def _generate_fake_sampling_metadata(
    num_output_tokens: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> SamplingMetadata:
    """Generate fake sampling metadata with fake logitsprocs"""
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
    logitsprocs = init_hard_coded_logitsprocs(
        pin_memory_available=PIN_MEMORY_AVAILABLE,
        max_num_reqs=MAX_NUM_REQS + 1,
        device=device)

    fake_sampling_metadata = SamplingMetadata(
        temperature=torch.full((batch_size, ), 0.0),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=create_prompt_tokens_tensor(prompt_token_ids,
                                                     vocab_size, device),
        output_token_ids=output_token_ids,
        frequency_penalties=create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=logitsprocs)
    return fake_sampling_metadata


def _generate_test_fakes(batch_size: int, device: str) -> LogitsprocsTestFakes:
    """Generate fake logits and sampling metadata"""
    fake_logits = create_fake_logits(batch_size, VOCAB_SIZE)
    # Create one dominant token per batch, to support min-p test
    for i in range(batch_size):
        fake_logits[i, 0] = 10.0  # High logit for first token
        fake_logits[i, 1:] = 1e-2  # Others remain low
    sampling_metadata = _generate_fake_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    return LogitsprocsTestFakes(
        logits=fake_logits,
        sampling_metadata=sampling_metadata,
    )


def _sampling_params_from_logitproc(logitproc_id: str) -> SamplingParams:
    """Customize request SamplingParams for a specified logitproc"""
    # SamplingParams for req with no logitproc
    kwargs = {"min_p": 0, "logit_bias": None, "min_tokens": 0}
    if fxn := logitsprocs_test_mapping[logitproc_id].gen_request_fxn:
        fxn(kwargs)
    return SamplingParams(**kwargs)


def _generate_mixed_logitsprocs_batch_params(
    reqs_per_logitproc: int,
    logitsprocs_ids: list[str],
) -> list[LogitsProcsRequestParams]:
    """Define key params for a batch of requests with a different
    logitproc enabled per request.
    
    The batch will have `reqs_per_logitproc` repeats for all
    `logitsprocs_ids` under test, including the case where
    no logitsproc is enabled. The batch is randomly shuffled. The
    size of the batch is `reqs_per_logitproc` times
    `n = len(logitsprocs_ids)`

    Args:
      reqs_per_logitproc: number of requests using each logitproc
      logitsprocs_ids: logitsprocs under test

    Returns:
      List of per-request params which configure the engine for that request's
      enabled logitproc
    """
    batch_size = len(logitsprocs_ids) * reqs_per_logitproc
    # Generate multiple repeats of key params for each logitproc;
    # apply random inverse permutation to the iteration
    # over logitsprocs, such that logitsprocs are shuffled.
    batch_perm = random.sample(range(batch_size), k=batch_size)
    return [
        LogitsProcsRequestParams(
            batch_index=idx,
            logitproc_id=logitsprocs_ids[pdx // reqs_per_logitproc])
        for idx, pdx in enumerate(batch_perm)
    ]


def _logit_bias_params(kwargs: dict) -> None:
    """Logit bias config"""
    kwargs["logit_bias"] = {
        random.randint(0, VOCAB_SIZE - 1): random.choice([-0.1, 0.2])
    }


def _logit_bias_validate(
    test_fakes: LogitsprocsTestFakes,
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
) -> bool:
    """Validate logit bias logitproc applied correctly"""
    logit_bias = request_params.params.logit_bias
    logits_old = test_fakes.logits[batch_index].cpu()
    logits_new = logits_new[batch_index].cpu()
    for token_id in range(VOCAB_SIZE):
        logit_old_value = logits_old[token_id]
        logit_new_value = logits_new[token_id]
        if token_id in logit_bias:
            bias_value = logit_bias[token_id]
            exp_value = bias_value + logit_old_value
            if logit_new_value != pytest.approx(exp_value):
                print(f"Biased token {token_id} logit value {logit_new_value} "
                      f"does not match expected value {exp_value} "
                      f"given bias {bias_value}")
                return False

        else:
            if logit_new_value != pytest.approx(logit_old_value):
                print(
                    f"Unbiased token {token_id} logit value {logit_new_value} "
                    f"does not match expected value {logit_old_value}")
                return False

    return True


def _min_p_params(kwargs: dict) -> None:
    """Min-p logitproc config"""
    kwargs["min_p"] = 0.1


def _min_p_validate(
    test_fakes: LogitsprocsTestFakes,
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
) -> bool:
    """Validate min-p logitproc applied correctly"""
    for token_id in range(VOCAB_SIZE):
        logits_for_token = logits_new[batch_index][token_id]
        if token_id == 0:
            # Dominant token should always be unmasked
            if logits_for_token == -float("inf"):
                print("Invalid: dominant token 0 masked (-inf)")
                return False
        else:
            if request_params.params.min_p > 0.0:
                # Non-dominant tokens should be masked when min_p > 0
                if logits_for_token != -float("inf"):
                    print(f"Invalid: non-dominant token {token_id} not masked")
                    return False
            else:
                # No masking when min_p is 0
                if logits_for_token == -float("inf"):
                    print(f"Invalid: token {token_id} masked when min_p=0.0")
                    return False
    return True


def _min_tokens_params(kwargs: dict) -> None:
    """Min-tokens logitproc config"""
    kwargs["min_tokens"] = MIN_TOKENS_LEN_THRESHOLD
    kwargs["stop_token_ids"] = [
        np.random.randint(0, VOCAB_SIZE - 1)
        for _ in range(np.random.randint(0, VOCAB_SIZE))
    ]


def _min_tokens_validate(
    test_fakes: LogitsprocsTestFakes,
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
) -> bool:
    """Validate min-tokens logitsproc applied correctly"""
    num_out_tokens = len(request_params.out_tokens)
    min_reached = num_out_tokens >= MIN_TOKENS_LEN_THRESHOLD
    stop_token_ids = request_params.params.stop_token_ids
    for token_id in range(VOCAB_SIZE):
        logits_for_token = logits_new[batch_index][token_id]
        if token_id in stop_token_ids and not min_reached:
            if logits_for_token != -float("inf"):
                print(f"Token {token_id} is a stop token and "
                      "the sequence has not reached min length, "
                      "but the token is not masked "
                      f"(logit={logits_for_token})")
                return False
        else:
            if logits_for_token == -float("inf"):
                print(f"Token {token_id} should not be masked but "
                      f"is (output len={len(num_out_tokens)})")
                return False

    return True


def _none_validate(
    test_fakes: LogitsprocsTestFakes,
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: LogitsProcsRequestParams,
) -> bool:
    """Validate that no logits processors are applied"""
    return torch.all(
        logits_new[batch_index] == test_fakes.logits.cpu()[batch_index])


class LogitsprocTestHelpers(NamedTuple):
    """Supports setting up and validating logitsprocs unit tests."""
    eval_fxn: Callable
    gen_request_fxn: Optional[Callable] = None


logitsprocs_test_mapping = {
    STR_NO_LOGITPROC:
    LogitsprocTestHelpers(eval_fxn=_none_validate),
    STR_LOGITS_BIAS_LOGITPROC_ID:
    LogitsprocTestHelpers(gen_request_fxn=_logit_bias_params,
                          eval_fxn=_logit_bias_validate),
    STR_MIN_P_LOGITPROC_ID:
    LogitsprocTestHelpers(gen_request_fxn=_min_p_params,
                          eval_fxn=_min_p_validate),
    STR_MIN_TOKENS_LOGITPROC_ID:
    LogitsprocTestHelpers(gen_request_fxn=_min_tokens_params,
                          eval_fxn=_min_tokens_validate),
}


def _get_test_cases() -> list[str]:
    """Each test case is a set of logitsprocs"""
    logitsprocs_ids = list(logitsprocs_test_mapping.keys())
    return [[logitproc_id]
            for logitproc_id in logitsprocs_ids] + [logitsprocs_ids]


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("reqs_per_logitproc", [REQS_PER_LOGITPROC])
@pytest.mark.parametrize("logitsprocs_under_test", _get_test_cases())
def test_logitsprocs(device: str, reqs_per_logitproc: int,
                     logitsprocs_under_test: list[str]):
    random.seed(42)
    torch.set_default_device(device)

    # Define a shuffled batch of requests which individually use a different
    # logitproc, or no logitproc at all
    batch_params = _generate_mixed_logitsprocs_batch_params(
        reqs_per_logitproc=reqs_per_logitproc,
        logitsprocs_ids=logitsprocs_under_test)
    batch_size = len(batch_params)

    # Create fake test data structures for testing.
    test_fakes = _generate_test_fakes(batch_size, device)

    # Construct logitsprocs batch update
    added: Sequence[AddedRequestType] = [
        (req_params.batch_index, req_params.params, req_params.out_tokens)
        for req_params in batch_params
    ]
    fake_batch_update = BatchUpdate(
        removed=[],
        moved=[],
        added=added,
        batch_size=batch_size,
    )

    # Apply fake batch update to logitsprocs
    fake_update_logitsprocs_state(test_fakes, fake_batch_update)

    # Emulate application of greedy logits processors in engine
    logits_w_lp = fake_apply_logits_processors(test_fakes)
    logits_w_lp = logits_w_lp.cpu()

    # Validate logits for each fake request
    for batch_index in range(batch_size):
        request_params = batch_params[batch_index]
        fxn = logitsprocs_test_mapping[request_params.logitproc_id].eval_fxn
        assert fxn(test_fakes=test_fakes,
                   logits_new=logits_w_lp,
                   batch_index=batch_index,
                   request_params=request_params), (
                       f"Validation failed for batch_index={batch_index}, "
                       f"req_params={request_params}")
