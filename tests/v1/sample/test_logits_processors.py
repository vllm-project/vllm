# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Callable, Sequence
from itertools import combinations
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
from vllm.v1.worker.utils import (STR_LOGITS_BIAS_LOGITSPROC_ID,
                                  STR_MIN_P_LOGITSPROC_ID,
                                  STR_MIN_TOKENS_LOGITSPROC_ID,
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
STR_NO_LOGITSPROC = "none"
MIN_TOKENS_LEN_THRESHOLD = 5
REQS_PER_COMBO = 10


class TestFakes(NamedTuple):
    """Wraps fake data structures to support testing"""
    logits: torch.Tensor
    sampling_metadata: SamplingMetadata

    def get_logitsproc_by_id(self, id: str) -> LogitsProcessor:
        """Shorthand for getting a specific logitproc from SamplingMetadata"""
        return self.sampling_metadata.logitsprocs.get_logitsproc_by_id(id)


class RequestParams:
    """Encapsulates key params for a single request in a batch.
    
    Params can be customized based on the enabled logitsprocs
    """
    batch_index: int
    combo: set[str]  # Logitsprocs enabled, specified by str id
    out_tokens: list[int]  # Output tokens required for min tokens test
    params: SamplingParams  # Settings customized for logitsprocs combo

    def __init__(self, batch_index: int, combo: set[str]):
        self.batch_index = batch_index
        self.combo = combo
        self.out_tokens = ([0] *
                           (MIN_TOKENS_LEN_THRESHOLD * random.randint(0, 1)))
        self.params = _sampling_params_from_combo(combo)


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
    logits = test_fakes.logits
    for processor in test_fakes.sampling_metadata.logitsprocs.greedy_list:
        logits = processor.apply(logits)
    return logits


def _fake_apply_nongreedy_logits_processors(
        test_fakes: TestFakes) -> torch.Tensor:
    """Imitate non-greedy-only logit processor application in engine
    core"""
    logits = test_fakes.logits
    for processor in test_fakes.sampling_metadata.logitsprocs.nongreedy_list:
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


def _min_p_params(kwargs: dict) -> None:
    kwargs["min_p"] = 0.1


def _logit_bias_params(kwargs: dict) -> None:
    pass


def _min_tokens_params(kwargs: dict) -> None:
    pass


class LogitsprocTestHelpers(NamedTuple):
    """Supports setting up and validating logitsprocs unit tests."""
    gen_request_fxn: Optional[Callable] = None
    eval_fxn: Optional[Callable] = None


logitsprocs_test_mapping = {
    STR_LOGITS_BIAS_LOGITSPROC_ID:
    LogitsprocTestHelpers(gen_request_fxn=_min_p_params),
    STR_MIN_P_LOGITSPROC_ID:
    LogitsprocTestHelpers(gen_request_fxn=_min_p_params),
    STR_MIN_TOKENS_LOGITSPROC_ID:
    LogitsprocTestHelpers(gen_request_fxn=_min_p_params),
}


def _sampling_params_from_combo(combo: set[str]) -> SamplingParams:
    """Customize SamplingParams for a specified combo of logitsprocs"""
    # SamplingParams for req with no logitsprocs
    kwargs = {"min_p": 0, "logit_bias": None, "min_tokens": 0}
    for id in combo:
        # Update SamplingParams for each enabled logitproc
        fxn = logitsprocs_test_mapping[id].gen_request_fxn
        fxn(kwargs)
    return SamplingParams(**kwargs)


def _generate_mixed_logitsprocs_batch_params(
    reqs_per_combo: int,
    logitsprocs_ids: set[str],
) -> list[RequestParams]:
    """Define key params for a batch of requests with different
    combinations of logitsprocs enabled per request.
    
    The batch will have `reqs_per_combo` repeats for all possible
    combinations of the `logitsprocs_ids`, including the case where
    no logitsprocs are enabled. The batch is randomly shuffled. The
    size of the batch is `reqs_per_combo`
    $\times \sum_{k=0}^n \binom{n}{k}$ where `n = len(logitsprocs_ids)`

    Args:
      reqs_per_combo: number of repeats of each combo of enabled logitsprocs
      logitsprocs_ids: available logitsprocs, to enable in different
                       combinations

    Returns:
      List of per-request params which configure the engine for that request's
      enabled logitsprocs
    """
    # List of $\binom{n}{k}$ for $k \in [0,n]$ where `n = len(logitsprocs_ids)`
    all_combos = [
        set(combo) for k in range(len(logitsprocs_ids) + 1)
        for combo in combinations(logitsprocs_ids, k)
    ]
    batch_size = len(all_combos) * reqs_per_combo
    # Generate multiple repeats of key params for each combo of
    # logits procs; apply random inverse permutation to the iteration
    # over logitsprocs combos, yielding shuffled batch
    batch_perm = random.sample(range(batch_size), k=batch_size)
    return [
        RequestParams(batch_index=idx, combo=all_combos[pdx // reqs_per_combo])
        for idx, pdx in enumerate(batch_perm)
    ]


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("reqs_per_combo", [REQS_PER_COMBO])
@pytest.mark.parametrize("logitsprocs_under_test",
                         [set(logitsprocs_test_mapping.keys())])
def test_mixed_batch_with_reordering(device: str, reqs_per_combo: int,
                                     logitsprocs_under_test: set[str]):
    batch_spec = _generate_mixed_logitsprocs_batch_params(
        reqs_per_combo=reqs_per_combo, logitsprocs_ids=logitsprocs_under_test)
    print(batch_spec)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("bias_value", [-0.1, 1.2])
def test_logit_bias(device: str, batch_size: int, bias_value: float):
    """
    Test to verify logit bias logits processor
    """
    torch.set_default_device(device)

    # Create fake logits where each token is assigned the same
    # logit value.
    test_fakes = _test_setup(batch_size, device)
    logit_bias_logitproc = test_fakes.get_logitsproc_by_id(
        STR_LOGITS_BIAS_LOGITSPROC_ID)
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

    min_p_logitproc = test_fakes.get_logitsproc_by_id(STR_MIN_P_LOGITSPROC_ID)
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
        STR_MIN_TOKENS_LOGITSPROC_ID)
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
