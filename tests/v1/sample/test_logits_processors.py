# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Callable, Sequence
from typing import NamedTuple, Optional

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.v1.sample.logits_processor import (AddedRequestType, BatchUpdate,
                                             LogitsProcessor)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.utils import (STR_LOGITS_BIAS_LOGITPROC_ID,
                                  STR_MIN_P_LOGITPROC_ID,
                                  STR_MIN_TOKENS_LOGITPROC_ID,
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
STR_NO_LOGITPROC = "none"
MIN_TOKENS_LEN_THRESHOLD = 5
REQS_PER_LOGITPROC = 10


class TestFakes(NamedTuple):
    """Wraps fake data structures to support testing"""
    logits: torch.Tensor
    sampling_metadata: SamplingMetadata

    def get_logitsproc_by_id(self, id: str) -> LogitsProcessor:
        """Shorthand for getting a specific logitproc from SamplingMetadata"""
        return self.sampling_metadata.logitsprocs.get_logitsproc_by_id(id)

    def get_logitsprocs(self) -> list[LogitsProcessor]:
        return self.sampling_metadata.logitsprocs.all_list


class RequestParams:
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
) -> list[dict[int, float]]:
    res: list[dict[int, float]] = []
    for i in range(batch_size):
        logit_bias = {min(i, vocab_size - 1): bias_value}
        res.append(logit_bias)
    return res


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
        prompt_token_ids=_create_prompt_tokens_tensor(prompt_token_ids,
                                                      vocab_size, device),
        output_token_ids=output_token_ids,
        frequency_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        presence_penalties=_create_penalty_tensor(batch_size, 0.0, device),
        repetition_penalties=_create_penalty_tensor(batch_size, 1.0, device),
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=logitsprocs)
    return fake_sampling_metadata


def _fake_apply_greedy_logits_processors(
        test_fakes: TestFakes) -> torch.Tensor:
    """Imitate greedy-compatible logit processor application
    in engine"""
    logits = test_fakes.logits.clone()
    for processor in test_fakes.sampling_metadata.logitsprocs.greedy_list:
        logits = processor.apply(logits)
    return logits


def _fake_apply_nongreedy_logits_processors(
        test_fakes: TestFakes) -> torch.Tensor:
    """Imitate non-greedy-only logit processor application in engine
    core"""
    logits = test_fakes.logits.clone()
    for processor in test_fakes.sampling_metadata.logitsprocs.nongreedy_list:
        logits = processor.apply(logits)
    return logits


def _fake_apply_all_logits_processors(test_fakes: TestFakes) -> torch.Tensor:
    """Imitate application of logits processors in engine core"""
    logits = test_fakes.logits.clone()
    for processor in test_fakes.sampling_metadata.logitsprocs.all_list:
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
            min_tokens[index] = (np.random.randint(
                num_output_tokens + 1, 2 * num_output_tokens), [
                    np.random.randint(0, vocab_size - 1)
                    for _ in range(np.random.randint(0, vocab_size))
                ])
        else:
            min_tokens[index] = (np.random.randint(0,
                                                   num_output_tokens), set())
    return min_tokens


def _test_setup(batch_size: int, device: str) -> TestFakes:
    fake_logits = _create_fake_logits(batch_size, VOCAB_SIZE)
    # Create one dominant token per batch, to support min-p test
    for i in range(batch_size):
        fake_logits[i, 0] = 10.0  # High logit for first token
        fake_logits[i, 1:] = 1e-2  # Others remain low
    sampling_metadata = _create_default_sampling_metadata(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE, torch.device(device))
    return TestFakes(
        logits=fake_logits,
        sampling_metadata=sampling_metadata,
    )


def _logit_bias_params(kwargs: dict) -> None:
    kwargs["logit_bias"] = _create_logit_bias(
        batch_size=1,
        vocab_size=VOCAB_SIZE,
        bias_value=random.choice([-0.1, 1.2]),
    )[0]


def _min_p_params(kwargs: dict) -> None:
    kwargs["min_p"] = 0.1


def _min_tokens_params(kwargs: dict) -> None:
    (
        _,
        kwargs["stop_token_ids"],
    ) = _generate_min_token_penalties_and_stop_tokens(NUM_OUTPUT_TOKENS, 1,
                                                      VOCAB_SIZE, [0])[0]
    kwargs["min_tokens"] = MIN_TOKENS_LEN_THRESHOLD


def _logit_bias_validate(
    test_fakes: TestFakes,
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: RequestParams,
) -> bool:
    logit_bias = request_params.params.logit_bias
    logits_old = test_fakes.logits[batch_index].cpu()
    logits_new = logits_new[batch_index].cpu()
    biased_index = 0
    for token_id in range(VOCAB_SIZE):
        logit_old_value = logits_old[token_id]
        logit_new_value = logits_new[token_id]
        if biased_index == token_id:
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


def _min_p_validate(
    test_fakes: TestFakes,
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: RequestParams,
) -> bool:
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


def _min_tokens_validate(
    test_fakes: TestFakes,
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: RequestParams,
) -> bool:
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
    test_fakes: TestFakes,
    logits_new: torch.Tensor,
    batch_index: int,
    request_params: RequestParams,
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


def _get_logitsprocs_under_test() -> list[str]:
    return list(logitsprocs_test_mapping.keys())


def _sampling_params_from_logitproc(logitproc_id: str) -> SamplingParams:
    """Customize SamplingParams for a specified logitproc"""
    # SamplingParams for req with no logitproc
    kwargs = {"min_p": 0, "logit_bias": None, "min_tokens": 0}
    if fxn := logitsprocs_test_mapping[logitproc_id].gen_request_fxn:
        fxn(kwargs)
    return SamplingParams(**kwargs)


def _generate_mixed_logitsprocs_batch_params(
    reqs_per_logitproc: int,
    logitsprocs_ids: list[str],
) -> list[RequestParams]:
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
        RequestParams(batch_index=idx,
                      logitproc_id=logitsprocs_ids[pdx // reqs_per_logitproc])
        for idx, pdx in enumerate(batch_perm)
    ]


def _fake_update_logitsprocs_state(
    test_fakes: TestFakes,
    batch_update: BatchUpdate,
) -> None:
    for logitproc in test_fakes.get_logitsprocs():
        logitproc.update_state(batch_update)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("reqs_per_logitproc", [REQS_PER_LOGITPROC])
@pytest.mark.parametrize("logitsprocs_under_test",
                         [_get_logitsprocs_under_test()])
def test_mixed_batch_with_reordering(device: str, reqs_per_logitproc: int,
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
    test_fakes = _test_setup(batch_size, device)

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
    _fake_update_logitsprocs_state(test_fakes, fake_batch_update)

    # Emulate application of greedy logits processors in engine
    logits_w_lp = _fake_apply_all_logits_processors(test_fakes)
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


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("bias_value", [-0.1, 1.2])
def test_logit_bias(device: str, batch_size: int, bias_value: float):
    """
    Test to verify logit bias logits processor
    """
    torch.set_default_device(device)

    # Create fake test data structures for testing
    test_fakes = _test_setup(batch_size, device)
    logit_bias_logitproc = test_fakes.get_logitsproc_by_id(
        STR_LOGITS_BIAS_LOGITPROC_ID)
    # Create batch update where each request demands a
    # different logit bias
    logit_bias_list = _create_logit_bias(
        batch_size=batch_size,
        vocab_size=VOCAB_SIZE,
        bias_value=bias_value,
    )
    added: Sequence[AddedRequestType] = [
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
    logit_bias_logitproc.update_state(batch_update)
    # Emulate application of greedy logits processors in engine
    logits = _fake_apply_greedy_logits_processors(test_fakes)
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

    # Create fake logits where each token is assigned the same
    # logit value.
    test_fakes = _test_setup(batch_size, device)

    min_p_logitproc = test_fakes.get_logitsproc_by_id(STR_MIN_P_LOGITPROC_ID)
    # Create batch update where each request demands
    # the same min_p value
    added: Sequence[AddedRequestType] = [(rdx, SamplingParams(min_p=min_p), [])
                                         for rdx in range(batch_size)]
    batch_update = BatchUpdate(
        removed=[],
        moved=[],
        added=added,
        batch_size=batch_size,
    )
    # Register batch update with logit processor
    min_p_logitproc.update_state(batch_update)
    # Emulate application of non-greedy logits processors in engine
    logits = _fake_apply_nongreedy_logits_processors(test_fakes)
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
    test_fakes = _test_setup(batch_size, device)
    min_tokens_logitproc = test_fakes.get_logitsproc_by_id(
        STR_MIN_TOKENS_LOGITPROC_ID)
    batch_indices_for_min_token_penalty = (
        [0] if batch_size == 1 else np.random.randint(
            0, batch_size - 1, size=np.random.randint(1, batch_size)).tolist())
    min_tokens_dict = _generate_min_token_penalties_and_stop_tokens(
        NUM_OUTPUT_TOKENS, batch_size, VOCAB_SIZE,
        batch_indices_for_min_token_penalty)

    # Create batch update where each request demands
    # a different min_tokens value
    added: Sequence[AddedRequestType] = [
        (rdx,
         SamplingParams(min_tokens=min_tokens_dict[rdx][0],
                        max_tokens=None,
                        stop_token_ids=list(min_tokens_dict[rdx][1])), [])
        for rdx in range(batch_size)
    ]
    batch_update = BatchUpdate(
        removed=[],
        moved=[],
        added=added,
        batch_size=batch_size,
    )
    # Register batch update with logit processor
    min_tokens_logitproc.update_state(batch_update)
    # Emulate application of greedy logits processors in engine
    logits = _fake_apply_greedy_logits_processors(test_fakes)
    logits = logits.cpu()
    for batch_idx in range(batch_size):
        _, stop_token_ids = min_tokens_dict.get(batch_idx, (0, set()))
        for token_id in range(VOCAB_SIZE):
            if token_id in stop_token_ids:
                assert logits[batch_idx][token_id] == -float("inf")
            else:
                assert logits[batch_idx][token_id] != -float("inf")
