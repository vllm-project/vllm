# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
MAX_PROMPT_SIZE = 100
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
MAX_NUM_PROMPT_TOKENS = 64


def _remove_requests(
        input_batch: InputBatch, batch_size: int,
        reqs: list[CachedRequestState]) -> tuple[set[str], list[int]]:
    """
    Remove some requests randomly from the batch and returns a tuple
    of 1) set of request removed 2) indices of the requests removed
    ordered in descending order
    """

    num_reqs_to_remove = np.random.randint(0, batch_size)
    req_indices_to_remove: set[int] = set()
    for _ in range(num_reqs_to_remove):
        req_index_to_remove = np.random.randint(0, batch_size)
        req_indices_to_remove.add(req_index_to_remove)

    req_indices_to_remove_list = list(req_indices_to_remove)
    req_indices_to_remove_list.sort(reverse=True)
    req_ids_to_remove: set[str] = set()
    for index in req_indices_to_remove:
        input_batch.remove_request(reqs[index].req_id)
        req_ids_to_remove.add(reqs[index].req_id)
    return req_ids_to_remove, req_indices_to_remove_list


def _construct_expected_sampling_metadata(
    reqs: list[CachedRequestState],
    req_ids_retained: set[int],
    req_id_index_in_input_batch: dict[str, int],
    device: torch.device,
) -> SamplingMetadata:
    """
    Constructs and returns the expected SamplingMetadata for this
    batch.
    """
    num_reqs = len(req_ids_retained)
    output_token_ids: list[list[int]] = [list() for _ in range(num_reqs)]
    prompt_token_ids: list[list[int]] = [list() for _ in range(num_reqs)]
    presence_penalties = [0.0 for _ in range(num_reqs)]
    frequency_penalties = [0.0 for _ in range(num_reqs)]
    repetition_penalties = [1.0 for _ in range(num_reqs)]
    top_k = [0 for _ in range(num_reqs)]
    top_p = [0.0 for _ in range(num_reqs)]
    min_p = [0.0 for _ in range(num_reqs)]
    temperature = [0.0 for _ in range(num_reqs)]
    min_tokens = {}
    logit_bias = [None] * num_reqs
    allowed_token_ids_mask = torch.zeros(num_reqs,
                                         VOCAB_SIZE,
                                         dtype=torch.bool,
                                         device=device)
    for req in reqs:
        if req.req_id not in req_ids_retained:
            continue
        index_in_input_batch = req_id_index_in_input_batch[req.req_id]
        output_token_ids[index_in_input_batch] = req.output_token_ids
        prompt_token_ids[index_in_input_batch] = req.prompt_token_ids
        presence_penalties[
            index_in_input_batch] = req.sampling_params.presence_penalty
        frequency_penalties[index_in_input_batch] = (
            req.sampling_params.frequency_penalty)
        repetition_penalties[index_in_input_batch] = (
            req.sampling_params.repetition_penalty)
        top_k[index_in_input_batch] = req.sampling_params.top_k
        top_p[index_in_input_batch] = req.sampling_params.top_p
        min_p[index_in_input_batch] = req.sampling_params.min_p
        temperature[index_in_input_batch] = req.sampling_params.temperature
        min_tokens[index_in_input_batch] = (
            req.sampling_params.min_tokens,
            req.sampling_params.all_stop_token_ids)
        logit_bias[index_in_input_batch] = req.sampling_params.logit_bias
        if req.sampling_params.allowed_token_ids:
            allowed_token_ids_mask[index_in_input_batch][
                req.sampling_params.allowed_token_ids] = True

    return SamplingMetadata(
        temperature=torch.tensor(temperature, dtype=torch.float,
                                 device=device),
        all_greedy=False,
        all_random=True,
        top_p=None if all(x == 1.0 for x in top_p) else torch.tensor(
            top_p, dtype=torch.float, device=device),
        top_k=None if all(x == 0 for x in top_k) else torch.tensor(
            top_k, dtype=torch.int, device=device),
        min_p=None if all(x == 0.0 for x in min_p) else torch.tensor(
            min_p, dtype=torch.float, device=device),
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=make_tensor_with_pad(
            prompt_token_ids,
            pad=VOCAB_SIZE,
            device=torch.device(device),
            dtype=torch.int64,
        ),
        frequency_penalties=torch.tensor(frequency_penalties,
                                         dtype=torch.float,
                                         device=device),
        presence_penalties=torch.tensor(presence_penalties,
                                        dtype=torch.float,
                                        device=device),
        repetition_penalties=torch.tensor(repetition_penalties,
                                          dtype=torch.float,
                                          device=device),
        output_token_ids=output_token_ids,
        min_tokens=min_tokens,
        no_penalties=(all(x == 0 for x in presence_penalties)
                      and all(x == 0 for x in frequency_penalties)
                      and all(x == 1 for x in repetition_penalties)),
        logit_bias=logit_bias,
        allowed_token_ids_mask=allowed_token_ids_mask,
    )


def _create_sampling_params():
    return SamplingParams(
        top_k=np.random.randint(1, 10),
        top_p=np.random.uniform(0.0, 1.0),
        presence_penalty=np.random.uniform(-2.0, 2.0),
        repetition_penalty=np.random.uniform(0.0, 2.0),
        frequency_penalty=np.random.uniform(-2.0, 2.0),
        min_tokens=np.random.randint(1, 10),
        stop_token_ids=[
            np.random.randint(0, VOCAB_SIZE)
            for _ in range(np.random.randint(10))
        ],
        logit_bias={0: np.random.uniform(-3.0, 3.0)},
    )


def _construct_cached_request_state(req_id_suffix: int):
    prompt_token_ids = [
        np.random.randint(0, VOCAB_SIZE)
        for _ in range(np.random.randint(0, MAX_PROMPT_SIZE))
    ]
    output_token_ids = [
        np.random.randint(0, VOCAB_SIZE)
        for _ in range(np.random.randint(0, NUM_OUTPUT_TOKENS))
    ]
    return CachedRequestState(
        req_id=f"req_id_{req_id_suffix}",
        prompt_token_ids=prompt_token_ids,
        prompt=None,
        sampling_params=_create_sampling_params(),
        mm_inputs=[],
        mm_positions=[],
        block_ids=[],
        generator=None,
        num_computed_tokens=len(output_token_ids),
        output_token_ids=output_token_ids,
    )


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [1, 2, 32, 64])
def test_sampling_metadata_in_input_batch(device: str, batch_size: int):
    """
    Tests the logic for managing sampling metadata in the InputBatch.

    This test involves adding a set of requests to the InputBatch,
    followed by removing a subset of them. Afterward, the batch is compacted,
    and the `make_sampling_metadata` method is invoked on the batch. The
    output of `make_sampling_metadata` is then compared against the expected
    results to ensure correctness.
    """
    input_batch: InputBatch = InputBatch(
        max_num_reqs=batch_size,
        max_model_len=1024,
        max_num_blocks_per_req=10,
        device=torch.device(device),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
    )
    reqs: list[CachedRequestState] = []
    req_id_reqs = {}
    req_id_output_token_ids = {}
    # Add requests
    for req_index in range(batch_size):
        req: CachedRequestState = _construct_cached_request_state(req_index)
        input_batch.add_request(req, req_index)
        reqs.append(req)
        req_id_reqs[req.req_id] = req
        req_id_output_token_ids[req.req_id] = req.output_token_ids

    # Remove some requests
    req_ids_to_remove, req_indices_to_remove = _remove_requests(
        input_batch, batch_size, reqs)
    req_ids_retained = set(req_id_reqs.keys()) - req_ids_to_remove

    # Compact the input batch
    input_batch.condense(req_indices_to_remove)

    # Generate the sampling metadata
    sampling_metadata = input_batch._make_sampling_metadata()

    # Create expected output.
    expected_sampling_metadata = _construct_expected_sampling_metadata(
        reqs,
        req_ids_retained,
        input_batch.req_id_to_index,
        device=torch.device(device))

    def same(t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]) -> bool:
        return (t1 is None
                and t2 is None) or (t1 is not None and t2 is not None
                                    and torch.allclose(t1, t2))

    # Assert the actual and expected output.
    assert torch.allclose(expected_sampling_metadata.temperature,
                          sampling_metadata.temperature)
    assert same(expected_sampling_metadata.top_p, sampling_metadata.top_p)
    assert same(expected_sampling_metadata.top_k, sampling_metadata.top_k)
    assert torch.allclose(
        expected_sampling_metadata.frequency_penalties,
        sampling_metadata.frequency_penalties,
    )
    assert torch.allclose(
        expected_sampling_metadata.presence_penalties,
        sampling_metadata.presence_penalties,
    )
    assert torch.allclose(
        expected_sampling_metadata.repetition_penalties,
        sampling_metadata.repetition_penalties,
    )
    assert torch.allclose(expected_sampling_metadata.prompt_token_ids,
                          sampling_metadata.prompt_token_ids)
    assert (expected_sampling_metadata.output_token_ids ==
            sampling_metadata.output_token_ids)
    assert expected_sampling_metadata.min_tokens == sampling_metadata.min_tokens
    assert expected_sampling_metadata.no_penalties == \
           sampling_metadata.no_penalties
    assert expected_sampling_metadata.logit_bias == sampling_metadata.logit_bias
    if sampling_metadata.allowed_token_ids_mask:
        assert torch.allclose(
            expected_sampling_metadata.allowed_token_ids_mask,
            sampling_metadata.allowed_token_ids_mask)
