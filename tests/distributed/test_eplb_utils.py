# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.parallel import EPLBConfig
from vllm.distributed.eplb import eplb_state as eplb_state_module
from vllm.distributed.eplb.eplb_communicator import ring_p2p_batches
from vllm.distributed.eplb.eplb_state import (
    EplbState,
    _commit_eplb_maps,
    _commit_eplb_maps_for_layer,
    rank_load_imbalance,
)


def _make_model_state(
    phy2log: torch.Tensor,
    log2phy: torch.Tensor,
    logcnt: torch.Tensor,
    phy2log_storage: torch.Tensor | None = None,
) -> MagicMock:
    """Build a minimal EplbModelState mock with its map tensors."""
    state = MagicMock()
    state.physical_to_logical_map = phy2log
    state.physical_to_logical_map_buffer = (
        phy2log if phy2log_storage is None else phy2log_storage
    )
    state.logical_to_physical_map = log2phy
    state.logical_replica_count = logcnt
    return state


def test_commit_eplb_maps_shape_change():
    """When the number of physical experts changes, resize the active map within its
    preallocated storage.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    # Build current state tensors
    storage = torch.full((num_layers, num_physical + 2), -1, dtype=torch.long)
    model_state = _make_model_state(
        phy2log=storage[:, :num_physical],
        phy2log_storage=storage,
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    # The new map has two more physical experts. These new physical experts will
    # automatically map to the first two logical experts
    new_phy2log_larger = (
        (torch.arange(num_physical + 2, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
    )
    _commit_eplb_maps(model_state, new_phy2log_larger)

    # Check that the number of physical experts has been updated and that the values
    # match
    assert model_state.physical_to_logical_map.shape[1] == num_physical + 2
    assert model_state.physical_to_logical_map.data_ptr() == storage.data_ptr()
    assert torch.equal(model_state.physical_to_logical_map, new_phy2log_larger)


def test_commit_eplb_maps_for_layer_logical_padding():
    """Test that logical_to_physical_map is padded with -1 to fill the
    pre-allocated slots when the new map has fewer replicas than the max.
    """
    num_layers, num_logical, num_physical = 2, 4, 6
    max_replicas = 3

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = (
        (torch.arange(num_physical, dtype=torch.long) % num_logical)
        .unsqueeze(0)
        .expand(num_layers, -1)
        .contiguous()
    )
    layer = 0
    _commit_eplb_maps_for_layer(model_state, new_phy2log[layer], layer)

    assert torch.all(model_state.logical_to_physical_map[layer, :, 2] == -1)


def test_commit_eplb_maps_for_layer_shape_assert():
    """Test that a mismatched number of physical experts triggers an assertion error."""
    num_layers, num_logical, num_physical = 2, 4, 6

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full((num_layers, num_logical, 2), -1, dtype=torch.long),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )
    bad_phy2log = torch.zeros(num_layers, num_physical + 1, dtype=torch.long)
    with pytest.raises(AssertionError):
        _commit_eplb_maps_for_layer(model_state, bad_phy2log, layer=0)


def test_commit_eplb_maps():
    """Test that all values are copied correctly into model_state."""
    num_layers, num_logical, num_physical, max_replicas = 2, 3, 4, 2

    model_state = _make_model_state(
        phy2log=torch.zeros(num_layers, num_physical, dtype=torch.long),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps(model_state, new_phy2log)

    assert torch.equal(model_state.physical_to_logical_map, new_phy2log)
    assert torch.equal(model_state.logical_to_physical_map, new_log2phy)
    assert torch.equal(model_state.logical_replica_count, new_logcnt)


def test_commit_eplb_maps_for_layer():
    """Test that only the target layer is updated"""
    num_layers, num_logical, max_replicas = 2, 3, 2

    original_phy2log = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    model_state = _make_model_state(
        phy2log=original_phy2log.clone(),
        log2phy=torch.full(
            (num_layers, num_logical, max_replicas), -1, dtype=torch.long
        ),
        logcnt=torch.zeros(num_layers, num_logical, dtype=torch.long),
    )

    new_phy2log = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=torch.long)
    new_log2phy = torch.tensor(
        [[[0, 3], [1, -1], [2, -1]], [[2, -1], [0, 3], [1, -1]]], dtype=torch.long
    )
    new_logcnt = torch.tensor([[2, 1, 1], [1, 2, 1]], dtype=torch.long)

    _commit_eplb_maps_for_layer(model_state, new_phy2log[0], layer=0)

    # Layer 0 updated
    assert torch.equal(model_state.physical_to_logical_map[0], new_phy2log[0])
    assert torch.equal(model_state.logical_to_physical_map[0], new_log2phy[0])
    assert torch.equal(model_state.logical_replica_count[0], new_logcnt[0])

    # Layer 1 untouched
    assert torch.equal(model_state.physical_to_logical_map[1], original_phy2log[1])


@pytest.mark.parametrize("max_rebalances, expected", [(0, 5), (2, 2)])
def test_eplb_max_rebalances_stops_rearranging(monkeypatch, max_rebalances, expected):
    """With max_rebalances, rearrangement and load recording stop for good."""
    monkeypatch.setattr(
        eplb_state_module,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(rank=lambda: 0, size=lambda: 1)
        ),
    )
    eplb_config = EPLBConfig(
        window_size=2,
        step_interval=4,
        max_rebalances=max_rebalances,
        use_async=False,
        communicator="torch_gloo",
    )
    state = EplbState(SimpleNamespace(eplb_config=eplb_config), torch.device("cpu"))
    state.expert_load_window_size = 2
    state.expert_rearrangement_step_interval = 4
    # add_model() starts the counter at 3/4 of the interval.
    state.expert_rearrangement_step = 3
    state.should_record_tensor = torch.ones((), dtype=torch.bool)
    calls = []
    monkeypatch.setattr(state, "rearrange", lambda **kwargs: calls.append(kwargs))

    for _ in range(20):
        state.step()

    # Rearrangement points at steps 1, 5, 9, 13 and 17 without a limit.
    assert len(calls) == expected
    assert state.num_rebalances == expected
    assert bool(state.should_record_tensor) == (max_rebalances == 0)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("world_size", [2, 3, 8])
def test_ring_p2p_batches_pair_every_send_with_its_recv(seed, world_size):
    """Each ring batch has one peer per direction, and a send from A to B
    lands in the same batch as B's receive from A, in the same order."""
    experts_per_rank = 6
    num_experts = world_size * experts_per_rank
    new_slot = torch.randperm(
        num_experts, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    owner_before = {e: e // experts_per_rank for e in range(num_experts)}
    owner_after = {e: slot // experts_per_rank for slot, e in enumerate(new_slot)}

    # Ops in the order move_to_buffer adds them: sends then receives, each
    # sorted by expert, one op per weight tensor.
    # rank -> [(is_send, peer, expert)]
    ops: dict[int, list[tuple[bool, int, int]]] = {r: [] for r in range(world_size)}
    for is_send in (True, False):
        for e in range(num_experts):
            src, dst = owner_before[e], owner_after[e]
            if src == dst:
                continue
            rank, peer = (src, dst) if is_send else (dst, src)
            ops[rank] += [(is_send, peer, e)] * 2

    batches: dict[tuple[int, int], list[tuple[bool, int, int]]] = {}
    for rank, rank_ops in ops.items():
        result = ring_p2p_batches(
            [(is_send, peer) for is_send, peer, _ in rank_ops], rank, world_size
        )
        assert [d for d, _ in result] == sorted(d for d, _ in result)
        covered = sorted(i for _, indices in result for i in indices)
        assert covered == list(range(len(rank_ops)))
        for d, indices in result:
            batch = [rank_ops[i] for i in indices]
            batches[(rank, d)] = batch
            assert {p for is_send, p, _ in batch if is_send} <= {
                (rank + d) % world_size
            }
            assert {p for is_send, p, _ in batch if not is_send} <= {
                (rank - d) % world_size
            }

    for (rank, d), batch in batches.items():
        dst = (rank + d) % world_size
        sent = [e for is_send, p, e in batch if is_send and p == dst]
        received = [
            e
            for is_send, p, e in batches.get((dst, d), [])
            if not is_send and p == rank
        ]
        assert sent == received


def test_rank_load_imbalance_is_per_layer():
    """Skews of different layers must not cancel out: each layer waits for its
    busiest rank."""
    # 2 layers x 4 experts on 2 ranks (2 experts each). Contiguous placement:
    # layer 0 overloads rank 0 and layer 1 overloads rank 1.
    loads = torch.tensor([[4.0, 4.0, 1.0, 1.0], [1.0, 1.0, 4.0, 4.0]])
    contiguous = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    balanced = torch.tensor([[0, 2, 1, 3], [0, 2, 1, 3]])

    assert rank_load_imbalance(contiguous, loads, ep_size=2) == pytest.approx(1.6)
    assert rank_load_imbalance(balanced, loads, ep_size=2) == pytest.approx(1.0)


def test_rank_load_imbalance_splits_replica_load():
    loads = torch.tensor([[6.0, 2.0, 2.0]])
    # Expert 0 has two replicas, one per rank: 3 + 2 on rank 0, 3 + 2 on rank 1.
    mapping = torch.tensor([[0, 1, 0, 2]])
    assert rank_load_imbalance(mapping, loads, ep_size=2) == pytest.approx(1.0)
