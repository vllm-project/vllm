# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from collections.abc import Sequence

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

VOCAB_SIZE = 1024
NUM_OUTPUT_TOKENS = 20
MAX_PROMPT_SIZE = 100
CUDA_DEVICES = [
    f"{current_platform.device_type}:{i}"
    for i in range(min(current_platform.device_count(), 2))
]
MAX_NUM_PROMPT_TOKENS = 64


def _compare_objs(obj1, obj2, skip: Sequence = ("logitsprocs", "batch_update_builder")):
    attrs = inspect.getmembers(obj1, lambda a: not (inspect.isroutine(a)))
    attr_names = set(
        [a[0] for a in attrs if not (a[0].startswith("__") and a[0].endswith("__"))]
    )
    for attr_name in attr_names:
        if attr_name in skip:
            continue

        a = getattr(obj1, attr_name)
        b = getattr(obj2, attr_name)

        is_same = False
        if isinstance(a, torch.Tensor):
            if a.numel() == 0 or b.numel() == 0:
                is_same = a.numel() == 0 and b.numel() == 0
            elif torch.allclose(a, b):
                is_same = True
        elif isinstance(a, np.ndarray):
            if np.allclose(a, b):
                is_same = True
        elif isinstance(a, MultiGroupBlockTable):
            for a_i, b_i in zip(a.block_tables, b.block_tables):
                _compare_objs(a_i, b_i)
            is_same = True
        elif isinstance(a, (BlockTable, SamplingMetadata, PoolingMetadata)):
            _compare_objs(a, b)
            is_same = True  # if we make it here must be same
        elif a == b:
            is_same = True
        elif isinstance(a, CpuGpuBuffer):
            is_same = np.allclose(a.np, b.np) and torch.allclose(a.gpu, b.gpu)
        assert is_same, (
            f"Attribute {attr_name} is different in {obj1} and {obj2}: {a} != {b}"
        )


def _remove_requests(
    input_batch: InputBatch, batch_size: int, reqs: list[CachedRequestState]
) -> set[str]:
    """
    Remove some requests randomly from the batch and returns
    set of request removed
    """

    num_reqs_to_remove = np.random.randint(0, batch_size)
    req_indices_to_remove: set[int] = set()
    for _ in range(num_reqs_to_remove):
        req_index_to_remove = np.random.randint(0, batch_size)
        req_indices_to_remove.add(req_index_to_remove)

    req_ids_to_remove: set[str] = set()
    for index in req_indices_to_remove:
        input_batch.remove_request(reqs[index].req_id)
        req_ids_to_remove.add(reqs[index].req_id)
    return req_ids_to_remove


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
    temperature = [0.0 for _ in range(num_reqs)]
    min_tokens = {}
    logit_bias = [None] * num_reqs
    allowed_token_ids_mask = torch.zeros(
        num_reqs, VOCAB_SIZE, dtype=torch.bool, device=device
    )
    bad_words_token_ids = {}
    for req in reqs:
        if req.req_id not in req_ids_retained:
            continue
        index_in_input_batch = req_id_index_in_input_batch[req.req_id]
        output_token_ids[index_in_input_batch] = req.output_token_ids
        prompt_token_ids[index_in_input_batch] = req.prompt_token_ids
        presence_penalties[index_in_input_batch] = req.sampling_params.presence_penalty
        frequency_penalties[index_in_input_batch] = (
            req.sampling_params.frequency_penalty
        )
        repetition_penalties[index_in_input_batch] = (
            req.sampling_params.repetition_penalty
        )
        top_k[index_in_input_batch] = req.sampling_params.top_k
        top_p[index_in_input_batch] = req.sampling_params.top_p
        temperature[index_in_input_batch] = req.sampling_params.temperature
        min_tokens[index_in_input_batch] = (
            req.sampling_params.min_tokens,
            req.sampling_params.all_stop_token_ids,
        )
        logit_bias[index_in_input_batch] = req.sampling_params.logit_bias
        if req.sampling_params.allowed_token_ids:
            allowed_token_ids_mask[index_in_input_batch][
                req.sampling_params.allowed_token_ids
            ] = True
        if req.sampling_params.bad_words_token_ids:
            bad_words_token_ids[index_in_input_batch] = (
                req.sampling_params.bad_words_token_ids
            )

    return SamplingMetadata(
        temperature=torch.tensor(temperature, dtype=torch.float, device=device),
        all_greedy=False,
        all_random=True,
        top_p=None
        if all(x == 1.0 for x in top_p)
        else torch.tensor(top_p, dtype=torch.float, device=device),
        top_k=None
        if all(x == 0 for x in top_k)
        else torch.tensor(top_k, dtype=torch.int, device=device),
        generators={},
        max_num_logprobs=0,
        prompt_token_ids=make_tensor_with_pad(
            prompt_token_ids,
            pad=VOCAB_SIZE,
            device=torch.device(device),
            dtype=torch.int64,
        ),
        frequency_penalties=torch.tensor(
            frequency_penalties, dtype=torch.float, device=device
        ),
        presence_penalties=torch.tensor(
            presence_penalties, dtype=torch.float, device=device
        ),
        repetition_penalties=torch.tensor(
            repetition_penalties, dtype=torch.float, device=device
        ),
        output_token_ids=output_token_ids,
        spec_token_ids=[[] for _ in range(len(output_token_ids))],
        no_penalties=(
            all(x == 0 for x in presence_penalties)
            and all(x == 0 for x in frequency_penalties)
            and all(x == 1 for x in repetition_penalties)
        ),
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=bad_words_token_ids,
        logitsprocs=LogitsProcessors(),
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
            np.random.randint(0, VOCAB_SIZE) for _ in range(np.random.randint(10))
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
        sampling_params=_create_sampling_params(),
        pooling_params=None,
        mm_features=[],
        block_ids=([],),
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

    Note: Ignore logits processor logic, which is tested separately
    """
    input_batch: InputBatch = InputBatch(
        max_num_reqs=batch_size,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device=torch.device(device),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    reqs: list[CachedRequestState] = []
    req_id_reqs = {}
    req_id_output_token_ids = {}

    # Add requests
    for req_index in range(batch_size):
        req: CachedRequestState = _construct_cached_request_state(req_index)
        assigned_req_index = input_batch.add_request(req)
        assert req_index == assigned_req_index
        reqs.append(req)
        req_id_reqs[req.req_id] = req
        req_id_output_token_ids[req.req_id] = req.output_token_ids

    # Remove some requests
    req_ids_to_remove = _remove_requests(input_batch, batch_size, reqs)
    req_ids_retained = set(req_id_reqs.keys()) - req_ids_to_remove

    # Compact the input batch
    input_batch.condense()

    # Generate the sampling metadata
    sampling_metadata = input_batch._make_sampling_metadata()

    # Create expected output.
    expected_sampling_metadata = _construct_expected_sampling_metadata(
        reqs, req_ids_retained, input_batch.req_id_to_index, device=torch.device(device)
    )

    def same(t1: torch.Tensor | None, t2: torch.Tensor | None) -> bool:
        return (t1 is None and t2 is None) or (
            t1 is not None and t2 is not None and torch.allclose(t1, t2)
        )

    # Assert the actual and expected output.
    assert torch.allclose(
        expected_sampling_metadata.temperature, sampling_metadata.temperature
    )
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
    assert torch.allclose(
        expected_sampling_metadata.prompt_token_ids, sampling_metadata.prompt_token_ids
    )
    assert (
        expected_sampling_metadata.output_token_ids
        == sampling_metadata.output_token_ids
    )
    assert expected_sampling_metadata.no_penalties == sampling_metadata.no_penalties
    if sampling_metadata.allowed_token_ids_mask:
        assert torch.allclose(
            expected_sampling_metadata.allowed_token_ids_mask,
            sampling_metadata.allowed_token_ids_mask,
        )
    assert (
        expected_sampling_metadata.bad_words_token_ids
        == sampling_metadata.bad_words_token_ids
    )


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("swap_list", [((0, 1),)])
def test_swap_states_in_input_batch(device: str, batch_size: int, swap_list: list):
    """
    Tests the logic for managing sampling metadata in the InputBatch.

    This test involves adding a set of requests to the InputBatch,
    followed by removing a subset of them. Afterward, the batch is compacted,
    and the `make_sampling_metadata` method is invoked on the batch. The
    output of `make_sampling_metadata` is then compared against the expected
    results to ensure correctness.

    Note: Ignore logits processor logic, which is tested separately
    """
    input_batch: InputBatch = InputBatch(
        max_num_reqs=batch_size,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device=torch.device(device),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    ref_input_batch: InputBatch = InputBatch(
        max_num_reqs=batch_size,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device=torch.device(device),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )

    reqs: list[CachedRequestState] = []
    req_id_reqs = {}
    req_id_output_token_ids = {}
    # Add requests
    for req_index in range(batch_size):
        req: CachedRequestState = _construct_cached_request_state(req_index)
        assigned_req_index = input_batch.add_request(req)
        assert assigned_req_index == req_index
        reqs.append(req)
        req_id_reqs[req.req_id] = req
        req_id_output_token_ids[req.req_id] = req.output_token_ids

    reordered_reqs = reqs.copy()
    for swap_pair in swap_list:
        reordered_reqs[swap_pair[0]], reordered_reqs[swap_pair[1]] = (
            reordered_reqs[swap_pair[1]],
            reordered_reqs[swap_pair[0]],
        )
        input_batch.swap_states(swap_pair[0], swap_pair[1])

    for req_index in range(batch_size):
        req = reordered_reqs[req_index]
        assigned_req_index = ref_input_batch.add_request(req)
        assert assigned_req_index == req_index

    input_batch.refresh_metadata()
    ref_input_batch.refresh_metadata()

    _compare_objs(input_batch, ref_input_batch)


# ==============================================================================
# Tests for merged_track_token_ids
# ==============================================================================


def _create_input_batch(device: str, batch_size: int = 32) -> InputBatch:
    """Helper to create an InputBatch for testing."""
    return InputBatch(
        max_num_reqs=batch_size,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device=torch.device(device),
        pin_memory=is_pin_memory_available(),
        vocab_size=VOCAB_SIZE,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )


def _create_cached_request_with_track_ids(
    req_id_suffix: int,
    track_token_ids: list[int] | None = None,
) -> CachedRequestState:
    """Create a CachedRequestState with optional track_token_ids."""
    prompt_token_ids = [
        np.random.randint(0, VOCAB_SIZE)
        for _ in range(np.random.randint(1, MAX_PROMPT_SIZE))
    ]
    output_token_ids = [
        np.random.randint(0, VOCAB_SIZE)
        for _ in range(np.random.randint(0, NUM_OUTPUT_TOKENS))
    ]
    sampling_params = SamplingParams(
        top_k=np.random.randint(1, 10),
        top_p=np.random.uniform(0.0, 1.0),
        track_token_ids=track_token_ids,
    )
    return CachedRequestState(
        req_id=f"req_id_{req_id_suffix}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=[],
        block_ids=([],),
        generator=None,
        num_computed_tokens=len(output_token_ids),
        output_token_ids=output_token_ids,
    )


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_empty(device: str):
    """
    Test that merged_track_token_ids returns None
    when no requests have track_token_ids.
    """
    input_batch = _create_input_batch(device)

    # Add requests without track_token_ids
    for i in range(3):
        req = _create_cached_request_with_track_ids(i, track_token_ids=None)
        input_batch.add_request(req)

    # Should return None
    assert input_batch.merged_track_token_ids is None


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_single_request(device: str):
    """Test merged_track_token_ids with a single request having track_token_ids."""
    input_batch = _create_input_batch(device)

    track_ids = [100, 200, 300]
    req = _create_cached_request_with_track_ids(0, track_token_ids=track_ids)
    input_batch.add_request(req)

    merged = input_batch.merged_track_token_ids

    assert merged is not None
    assert merged.device == torch.device(device)
    assert merged.tolist() == sorted(track_ids)


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_multiple_requests_disjoint(device: str):
    """
    Test merged_track_token_ids with multiple requests
    having disjoint track_token_ids.
    """
    input_batch = _create_input_batch(device)

    # Request 0: track [100, 200]
    req0 = _create_cached_request_with_track_ids(0, track_token_ids=[100, 200])
    input_batch.add_request(req0)

    # Request 1: track [300, 400]
    req1 = _create_cached_request_with_track_ids(1, track_token_ids=[300, 400])
    input_batch.add_request(req1)

    merged = input_batch.merged_track_token_ids

    assert merged is not None
    # Should contain union of all track_token_ids, sorted
    assert merged.tolist() == [100, 200, 300, 400]


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_multiple_requests_overlapping(device: str):
    """Test merged_track_token_ids with overlapping track_token_ids (deduplication)."""
    input_batch = _create_input_batch(device)

    # Request 0: track [100, 200, 300]
    req0 = _create_cached_request_with_track_ids(0, track_token_ids=[100, 200, 300])
    input_batch.add_request(req0)

    # Request 1: track [200, 300, 400] (overlaps with req0)
    req1 = _create_cached_request_with_track_ids(1, track_token_ids=[200, 300, 400])
    input_batch.add_request(req1)

    merged = input_batch.merged_track_token_ids

    assert merged is not None
    # Should deduplicate and sort
    assert merged.tolist() == [100, 200, 300, 400]


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_mixed_requests(device: str):
    """
    Test merged_track_token_ids with some requests
    having track_token_ids and others not.
    """
    input_batch = _create_input_batch(device)

    # Request 0: no track_token_ids
    req0 = _create_cached_request_with_track_ids(0, track_token_ids=None)
    input_batch.add_request(req0)

    # Request 1: track [100, 200]
    req1 = _create_cached_request_with_track_ids(1, track_token_ids=[100, 200])
    input_batch.add_request(req1)

    # Request 2: no track_token_ids
    req2 = _create_cached_request_with_track_ids(2, track_token_ids=None)
    input_batch.add_request(req2)

    merged = input_batch.merged_track_token_ids

    assert merged is not None
    assert merged.tolist() == [100, 200]


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_empty_list(device: str):
    """Test merged_track_token_ids when a request has track_token_ids=[]."""
    input_batch = _create_input_batch(device)

    # Request 0: empty list
    req0 = _create_cached_request_with_track_ids(0, track_token_ids=[])
    input_batch.add_request(req0)

    # Should return None since no tokens to track
    assert input_batch.merged_track_token_ids is None


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_many_tokens(device: str):
    """Test merged_track_token_ids with many tokens (classification use case)."""
    input_batch = _create_input_batch(device)

    # Track 100 tokens for classification
    track_ids = list(range(100, 200))
    req = _create_cached_request_with_track_ids(0, track_token_ids=track_ids)
    input_batch.add_request(req)

    merged = input_batch.merged_track_token_ids

    assert merged is not None
    assert merged.tolist() == track_ids
    assert len(merged) == 100


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_after_remove_request(device: str):
    """Test that merged_track_token_ids updates after removing a request."""
    input_batch = _create_input_batch(device)

    # Request 0: track [100, 200]
    req0 = _create_cached_request_with_track_ids(0, track_token_ids=[100, 200])
    input_batch.add_request(req0)

    # Request 1: track [300, 400]
    req1 = _create_cached_request_with_track_ids(1, track_token_ids=[300, 400])
    input_batch.add_request(req1)

    # Initial merged should have all 4 tokens
    assert input_batch.merged_track_token_ids.tolist() == [100, 200, 300, 400]

    # Remove request 0
    input_batch.remove_request(req0.req_id)

    # After removal, should only have tokens from request 1
    assert input_batch.merged_track_token_ids.tolist() == [300, 400]


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_merged_track_token_ids_tensor_properties(device: str):
    """Test that the returned tensor has correct properties."""
    input_batch = _create_input_batch(device)

    track_ids = [500, 100, 300, 200]  # Unsorted input
    req = _create_cached_request_with_track_ids(0, track_token_ids=track_ids)
    input_batch.add_request(req)

    merged = input_batch.merged_track_token_ids

    assert merged is not None
    assert merged.dtype == torch.int64
    assert merged.device == torch.device(device)
    # Should be sorted
    assert merged.tolist() == sorted(track_ids)


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_track_token_ids_validation_exceeds_vocab(device: str):
    """Test that track_token_ids exceeding vocab_size raises ValueError."""
    input_batch = _create_input_batch(device)

    # Try to add request with token ID >= vocab_size
    invalid_track_ids = [100, VOCAB_SIZE + 100]  # VOCAB_SIZE is 1024
    req = _create_cached_request_with_track_ids(0, track_token_ids=invalid_track_ids)

    with pytest.raises(ValueError) as exc_info:
        input_batch.add_request(req)

    assert "exceed vocab_size" in str(exc_info.value)
