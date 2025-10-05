# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.cp_utils import (cp_get_shard_size,
                                                 prepare_inputs_for_cp)
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState


@pytest.fixture(autouse=True)
def patch_parallel_state(monkeypatch):
    # Patch get_context_parallel_world_size and get_context_parallel_rank
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_context_parallel_world_size",
        lambda: 2)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_context_parallel_rank", lambda: 0)

    # Patch get_cp_group to return a mock object
    class MockCPGroup:
        world_size = 2
        rank = 0
        rank_in_group = 0

    monkeypatch.setattr("vllm.distributed.parallel_state.get_cp_group",
                        lambda: MockCPGroup())


def make_cached_request_state(id: int, prefill_len: int, decode_len: int,
                              num_computed_tokens: list[int]):
    assert prefill_len + decode_len == sum(num_computed_tokens)
    return CachedRequestState(
        req_id="req" + str(id),
        prompt_token_ids=list(range(prefill_len)),
        prompt_embeds=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        generator=None,
        block_ids=([0], ),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=list(range(decode_len)),
        lora_request=None,
    )


def create_block_table():
    return MultiGroupBlockTable(max_num_reqs=32,
                                max_model_len=2048,
                                max_num_batched_tokens=512,
                                pin_memory=False,
                                device=torch.device("cpu"),
                                block_sizes=[16],
                                num_speculative_tokens=0)


def test_prepare_inputs_for_cp_prefill(monkeypatch):
    # Setup
    id = 0
    prefill_len = 8
    decode_len = 0
    num_computed_tokens = [0]
    num_scheduled_tokens_ = prefill_len

    req_state = CachedRequestState(
        req_id="req" + str(id),
        prompt_token_ids=list(range(prefill_len)),
        prompt_embeds=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        generator=None,
        block_ids=([0], ),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=list(range(decode_len)),
        lora_request=None,
    )
    num_scheduled_tokens = {req_state.req_id: num_scheduled_tokens_}
    req_ids = [req_state.req_id]
    requests = {req_state.req_id: req_state}

    block_table = MultiGroupBlockTable(max_num_reqs=32,
                                       max_model_len=2048,
                                       max_num_batched_tokens=512,
                                       pin_memory=False,
                                       device=torch.device("cpu"),
                                       block_sizes=[16],
                                       num_speculative_tokens=0)

    positions_np = np.zeros(64, dtype=np.int64)
    computed_positions_np = np.zeros(64, dtype=np.int64)
    arange_np = np.arange(64, dtype=np.int64)
    padding_loc = -1

    # Run
    num_sched_local, num_comp_local, q_seqlens_sharded = prepare_inputs_for_cp(
        num_scheduled_tokens, requests, req_ids, block_table, positions_np,
        computed_positions_np, arange_np, padding_loc)

    # Check
    cp_shard_size, _ = cp_get_shard_size(num_scheduled_tokens_)
    assert num_sched_local == [2 * cp_shard_size]
    assert num_comp_local == [0]
    assert q_seqlens_sharded == [[cp_shard_size, cp_shard_size]]
    assert np.all(
        positions_np[:sum(num_sched_local)] == np.array([0, 1, 6, 7]))
    if sum(num_comp_local) > 0:
        assert np.all(computed_positions_np[:sum(num_comp_local)] == np.arange(
            2 * cp_shard_size))


def test_prepare_inputs_for_cp_decode(monkeypatch):
    # Setup
    id = 0
    prefill_len = 8
    decode_len = 2
    num_computed_tokens = [0, 4, 8, 9, 10]
    num_scheduled_tokens_ = 1

    req_state = CachedRequestState(
        req_id="req" + str(id),
        prompt_token_ids=list(range(prefill_len)),
        prompt_embeds=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        generator=None,
        block_ids=([0], ),
        num_computed_tokens=num_computed_tokens,
        output_token_ids=list(range(decode_len)),
        lora_request=None,
    )
    num_scheduled_tokens = {req_state.req_id: num_scheduled_tokens_}
    req_ids = [req_state.req_id]
    requests = {req_state.req_id: req_state}

    block_table = MultiGroupBlockTable(max_num_reqs=32,
                                       max_model_len=2048,
                                       max_num_batched_tokens=512,
                                       pin_memory=False,
                                       device=torch.device("cpu"),
                                       block_sizes=[16],
                                       num_speculative_tokens=0)

    positions_np = np.zeros(64, dtype=np.int64)
    computed_positions_np = np.zeros(64, dtype=np.int64)
    arange_np = np.arange(64, dtype=np.int64)
    padding_loc = -1

    # Run
    num_sched_local, num_comp_local, q_seqlens_sharded = prepare_inputs_for_cp(
        num_scheduled_tokens, requests, req_ids, block_table, positions_np,
        computed_positions_np, arange_np, padding_loc)

    # Check
    assert num_sched_local == [1]
    assert num_comp_local == [num_computed_tokens[-1] // 2]
    assert q_seqlens_sharded == [[1]]
    assert np.all(positions_np[:num_sched_local[0]] == np.array([10]))
    if sum(num_comp_local) > 0:
        assert np.all(computed_positions_np[:num_comp_local[0]] == np.array(
            [0, 3, 4, 7, 8]))


def test_prepare_inputs_for_cp_multiple_requests(monkeypatch):
    # Setup
    prefill_lens = [8, 16]
    decode_lens = [2, 0]
    num_computed_tokens = [[0, 4, 8, 9, 10], [0, 8]]
    num_scheduled_tokens_ = [1, 8]

    num_scheduled_tokens = {}
    requests = {}
    req_ids = []
    for i in range(2):
        req_state = CachedRequestState(
            req_id="req" + str(i),
            prompt_token_ids=list(range(prefill_lens[i])),
            prompt_embeds=None,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            generator=None,
            block_ids=([0], ),
            num_computed_tokens=num_computed_tokens[i],
            output_token_ids=list(range(decode_lens[i])),
            lora_request=None,
        )
        num_scheduled_tokens[req_state.req_id] = num_scheduled_tokens_[i]
        req_ids.append(req_state.req_id)
        requests[req_state.req_id] = req_state

    block_table = MultiGroupBlockTable(max_num_reqs=32,
                                       max_model_len=2048,
                                       max_num_batched_tokens=512,
                                       pin_memory=False,
                                       device=torch.device("cpu"),
                                       block_sizes=[16],
                                       num_speculative_tokens=0)

    positions_np = np.zeros(64, dtype=np.int64)
    computed_positions_np = np.zeros(64, dtype=np.int64)
    arange_np = np.arange(64, dtype=np.int64)
    padding_loc = -1

    # Run
    num_sched_local, num_comp_local, q_seqlens_sharded = prepare_inputs_for_cp(
        num_scheduled_tokens, requests, req_ids, block_table, positions_np,
        computed_positions_np, arange_np, padding_loc)

    # Check
    assert num_sched_local == [1, 4]
    assert num_comp_local == [
        num_computed_tokens[0][-1] // 2, [num_computed_tokens[1][-1] // 2]
    ]
    assert q_seqlens_sharded == [[1], [2, 2]]
    assert np.all(positions_np[:num_sched_local[0]] == np.array([10]))
    assert np.all(positions_np[num_sched_local[0]:num_sched_local[0] +
                               num_sched_local[1]] == np.array([8, 9, 14, 15]))
    if sum(num_comp_local) > 0:
        assert np.all(computed_positions_np[:num_comp_local[0]] == np.array(
            [0, 3, 4, 7, 8]))
        assert np.all(
            computed_positions_np[num_comp_local[0]:num_comp_local[0] +
                                  num_comp_local[1]] == np.array([0, 1, 6, 7]))
