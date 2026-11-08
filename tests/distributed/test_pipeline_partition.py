# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.distributed import parallel_state
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.utils import get_pp_indices


def test_custom_layer_partition(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:

        def _verify(partition_str, num_layers, pp_size, goldens):
            bak = os.environ.get("VLLM_PP_LAYER_PARTITION", None)
            m.setenv("VLLM_PP_LAYER_PARTITION", partition_str)
            for pp_rank, golden in enumerate(goldens):
                assert get_pp_indices(num_layers, pp_rank, pp_size) == golden
            if bak is not None:
                m.setenv("VLLM_PP_LAYER_PARTITION", bak)

        # Even partition
        _verify("5,5,5,5", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Balanced partition
        _verify("4,6,6,4", 20, 4, [(0, 4), (4, 10), (10, 16), (16, 20)])
        # Put reminder somewhere
        _verify("5,6,5,6", 22, 4, [(0, 5), (5, 11), (11, 16), (16, 22)])
        # A single stage, such as a draft model under PP, holds every layer
        _verify("5,5,5,5", 6, 1, [(0, 6)])
        # Invalid partition strings
        with pytest.raises(ValueError):
            _verify("5,5,5,5,", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        with pytest.raises(ValueError):
            _verify("5,5,5,a", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Wrong number of partitions
        with pytest.raises(ValueError):
            _verify("5,5,5", 20, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Wrong number of layers
        with pytest.raises(ValueError):
            _verify("5,5,5,5", 21, 4, [(0, 5), (5, 10), (10, 15), (15, 20)])
        # Negative layer count
        with pytest.raises(ValueError):
            _verify("-1,5", 4, 2, [(0, -1), (-1, 4)])


@pytest.mark.parametrize(
    "num_hidden_layers,pp_size,pp_rank,indices",
    [
        # pp_size 2
        (2, 2, 0, (0, 1)),
        (2, 2, 1, (1, 2)),
        (3, 2, 0, (0, 2)),
        (3, 2, 1, (2, 3)),
        # pp_size 3
        (3, 3, 0, (0, 1)),
        (3, 3, 1, (1, 2)),
        (3, 3, 2, (2, 3)),
        (4, 3, 0, (0, 1)),
        (4, 3, 1, (1, 3)),
        (4, 3, 2, (3, 4)),
        (5, 3, 0, (0, 2)),
        (5, 3, 1, (2, 4)),
        (5, 3, 2, (4, 5)),
    ],
)
def test_uneven_auto_partition(
    num_hidden_layers: int,
    pp_size: int,
    pp_rank: int,
    indices: tuple[int, int],
):
    assert indices == get_pp_indices(num_hidden_layers, pp_rank, pp_size)


def _pp_group(rank_in_group: int, world_size: int) -> GroupCoordinator:
    # Only the rank fields that define the stage; no process groups.
    group = object.__new__(GroupCoordinator)
    group.rank = rank_in_group
    group.ranks = list(range(world_size))
    group.rank_in_group = rank_in_group
    group.world_size = world_size
    return group


def test_single_pp_stage_runs_the_last_rank_as_the_only_stage(
    monkeypatch: pytest.MonkeyPatch,
):
    """A draft model runs whole on the last PP rank, as at PP=1."""
    last_rank = _pp_group(rank_in_group=1, world_size=2)
    monkeypatch.setattr(parallel_state, "_PP", last_rank)
    with parallel_state.single_pp_stage():
        stage = parallel_state.get_pp_group()
        assert stage.is_first_rank and stage.is_last_rank
        assert (stage.rank_in_group, stage.world_size) == (0, 1)
        assert parallel_state.in_single_pp_stage()
    assert parallel_state.get_pp_group() is last_rank
    assert not parallel_state.in_single_pp_stage()

    only_rank = _pp_group(rank_in_group=0, world_size=1)
    monkeypatch.setattr(parallel_state, "_PP", only_rank)
    with parallel_state.single_pp_stage():
        assert parallel_state.get_pp_group() is only_rank
        assert not parallel_state.in_single_pp_stage()
