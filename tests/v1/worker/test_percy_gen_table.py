# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy G1: the device generation table.

It mirrors the scheduler's GrantTable gens exactly (and the W1/W5 CPU mirror),
costs 4 bytes per block, never changes the null block, keeps fixed addresses
for CUDA graphs and is updated in stream order before the step that reads it.
"""

import random
from collections import deque

import numpy as np
import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from tests.v1.worker.test_percy_rows import (
    BLOCK_SIZE,
    _model_output,
    _sliding_window_scheduler,
)
from vllm.v1.core.grant_table import NULL_BLOCK_ID, NULL_GEN, normalize_gen_updates
from vllm.v1.worker.gpu.block_table import PercyGenCheck
from vllm.v1.worker.gpu.percy_rows import RowVerifier


def _gen_check(num_blocks: int, device: str = "cpu") -> PercyGenCheck:
    return PercyGenCheck(
        num_kv_blocks=num_blocks,
        block_sizes=[BLOCK_SIZE],
        max_num_reqs=8,
        max_num_batched_tokens=256,
        device=torch.device(device),
    )


def _device_gens(gen_check: PercyGenCheck) -> np.ndarray:
    return gen_check.table.cpu().numpy().view(np.uint32)


def _drive_and_compare(
    scheduler, arrivals: dict[int, list], num_steps: int, depth: int = 1
) -> int:
    """Run the scheduler like the engine; after every dispatched output,
    apply its gen_updates to a device table and a RowVerifier mirror, and
    require device table == GrantTable gens == CPU mirror for every block.
    Returns the number of gen updates seen."""
    rng = random.Random(0)
    manager = scheduler.kv_cache_manager
    num_blocks = manager.block_pool.num_gpu_blocks
    num_groups = len(manager.coordinator.single_type_managers)
    gen_check = _gen_check(num_blocks)
    verifier = RowVerifier(num_groups, num_blocks)
    all_ids = np.arange(num_blocks)
    num_updates = 0
    in_flight: deque = deque()
    for step in range(num_steps):
        for request in arrivals.get(step, ()):
            scheduler.add_request(request)
        if scheduler.has_requests():
            so = scheduler.schedule()
            if so.gen_updates:
                num_updates += len(so.gen_updates[0])
            gen_check.apply_gen_updates(so.gen_updates)
            verifier.begin_step(so)
            scheduler_gens = manager.grant_table.gens_of(all_ids)
            np.testing.assert_array_equal(_device_gens(gen_check), scheduler_gens)
            np.testing.assert_array_equal(verifier.gen_mirror, scheduler_gens)
            assert gen_check.mismatched_blocks(verifier.gen_mirror).size == 0
            in_flight.append(so)
        while in_flight and (len(in_flight) >= depth or not scheduler.has_requests()):
            done = in_flight.popleft()
            scheduler.update_from_output(done, _model_output(done, rng))
    return num_updates


# ------------------------------------------- (a) end-to-end mirror invariant
def test_table_tracks_grant_table_through_prefix_hits_and_preemption():
    scheduler = create_scheduler(
        enable_prefix_caching=True, num_blocks=16, block_size=BLOCK_SIZE
    )
    reqs = create_requests(
        4, num_tokens=60, max_tokens=30, same_prompt=True, block_size=BLOCK_SIZE
    )
    num_updates = _drive_and_compare(scheduler, {0: reqs[:2], 5: reqs[2:]}, 120)
    assert num_updates, "the scenario must revoke blocks"
    assert all(r.is_finished() for r in reqs)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_table_tracks_grant_table_with_sliding_window(async_scheduling: bool):
    scheduler = _sliding_window_scheduler(async_scheduling=async_scheduling)
    reqs = create_requests(2, num_tokens=100, max_tokens=40, block_size=BLOCK_SIZE)
    num_updates = _drive_and_compare(
        scheduler, {0: reqs}, 80, depth=2 if async_scheduling else 1
    )
    assert num_updates
    assert all(r.is_finished() for r in reqs)


def test_table_tracks_grant_table_under_async_prefix_caching():
    scheduler = create_scheduler(
        async_scheduling=True,
        enable_prefix_caching=True,
        num_blocks=20,
        block_size=BLOCK_SIZE,
    )
    reqs = create_requests(
        4, num_tokens=70, max_tokens=20, same_prompt=True, block_size=BLOCK_SIZE
    )
    assert _drive_and_compare(scheduler, {0: reqs[:1], 3: reqs[1:]}, 120, depth=2)
    assert all(r.is_finished() for r in reqs)


# ------------------------------------------------------------ (b) memory
def test_table_is_four_bytes_per_block_and_the_only_per_block_buffer():
    small, large = _gen_check(100), _gen_check(100_000)
    assert small.table.dtype == torch.int32
    assert small.table_bytes_per_block == 4
    assert large.table.numel() == 100_000
    # Every other device buffer is sized by requests/tokens, not by blocks:
    # no gen sits beside block-table entries on the GPU.
    for name in ("meta", "gens"):
        assert getattr(small, name).numel() == getattr(large, name).numel()


def test_table_initial_state():
    gen_check = _gen_check(64)
    gens = _device_gens(gen_check)
    assert gens[NULL_BLOCK_ID] == NULL_GEN
    assert np.all(gens[1:] == 1)


# ------------------------------------------------------- (c) null block
def test_null_block_update_is_dropped_on_both_mirrors():
    gen_check = _gen_check(16)
    verifier = RowVerifier(1, 16)
    updates = ([NULL_BLOCK_ID, 3, 99], [7, 5, 6])  # null and out-of-range
    gen_check.apply_gen_updates(updates)
    verifier.begin_step(_step(updates))
    for gens in (_device_gens(gen_check), verifier.gen_mirror):
        assert gens[NULL_BLOCK_ID] == NULL_GEN
        assert gens[3] == 5
    np.testing.assert_array_equal(_device_gens(gen_check), verifier.gen_mirror)


# ------------------------------------------------ (d) repeated block ids
def test_repeated_block_in_one_message_last_gen_wins_on_both_mirrors():
    gen_check = _gen_check(16)
    verifier = RowVerifier(1, 16)
    updates = ([4, 9, 4, 4], [2, 3, 7, 0xFFFFFFFE])
    gen_check.apply_gen_updates(updates)
    verifier.begin_step(_step(updates))
    for gens in (_device_gens(gen_check), verifier.gen_mirror):
        assert gens[4] == 0xFFFFFFFE and gens[9] == 3
    np.testing.assert_array_equal(_device_gens(gen_check), verifier.gen_mirror)


def test_normalize_gen_updates_keeps_order_of_last_occurrence():
    ids, gens, dropped = normalize_gen_updates([5, 2, 5, 0, 3], [1, 2, 3, 4, -1], 8)
    assert dropped == 1
    assert ids.tolist() == [2, 5, 3]
    assert gens.tolist() == [2, 3, NULL_GEN]


# ------------------------------------------- fixed addresses (CUDA graphs)
def test_updates_and_windows_keep_buffer_addresses():
    gen_check = _gen_check(32)
    ptrs = [t.data_ptr() for t in (gen_check.table, gen_check.meta, gen_check.gens)]
    for step in range(3):
        gen_check.apply_gen_updates(([1 + step, 2 + step], [5, 6]))
        gen_check.begin_windows()
        gen_check.set_windows(0, [(0, 1, 1)])
        gen_check.commit_windows()
    assert ptrs == [
        t.data_ptr() for t in (gen_check.table, gen_check.meta, gen_check.gens)
    ]


def _step(gen_updates) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        msg_seq=0,
        gen_updates=gen_updates,
        finished_req_ids=set(),
        preempted_req_ids=set(),
        new_block_ids_to_zero=None,
        new_block_gens_to_zero=None,
        kv_cache_block_copies=None,
        kv_cache_block_copy_gens=None,
    )


# ---------------------------------------------- (e) GPU-only: I10 ordering
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@requires_cuda
def test_update_is_stream_ordered_after_earlier_reads():
    gen_check = _gen_check(1024, "cuda")
    before = torch.empty_like(gen_check.table)
    after = torch.empty_like(gen_check.table)
    torch.cuda._sleep(50_000_000)  # keep the stream busy past the host calls
    before.copy_(gen_check.table)  # a read enqueued before the update
    gen_check.apply_gen_updates(([5], [9]))
    after.copy_(gen_check.table)  # a read enqueued after it
    torch.cuda.synchronize()
    assert before[5].item() == 1
    assert after[5].item() == 9


@requires_cuda
def test_update_between_graph_replays_is_seen_by_the_next_replay():
    gen_check = _gen_check(1024, "cuda")
    idx = torch.tensor([5, 6], device="cuda")
    out = torch.empty(2, dtype=torch.int32, device="cuda")
    torch.index_select(gen_check.table, 0, idx, out=out)  # warm up
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.index_select(gen_check.table, 0, idx, out=out)
    graph.replay()
    assert out.tolist() == [1, 1]
    gen_check.apply_gen_updates(([5, 6], [3, 4]))
    graph.replay()
    torch.cuda.synchronize()
    assert out.tolist() == [3, 4]
