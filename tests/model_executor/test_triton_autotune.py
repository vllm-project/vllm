# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.warmup.triton_autotune import (
    TritonAutotuneRegistry,
    TunableConfigTable,
    TuningItem,
    _merge_pending_items,
    _merge_results,
    _shard_items,
)

pytestmark = pytest.mark.cpu_test


class _FakeTable(TunableConfigTable):
    name = "fake"

    def __init__(self, pending=()):
        self.pending = list(pending)
        self.tuned = []
        self.committed = None

    def pending_items(self, worker):
        return self.pending

    def tune(self, item):
        self.tuned.append(item)
        return {"BLOCK_SIZE": item.bucket}

    def commit(self, results):
        self.committed = dict(results)


def _item(bucket, shape=(64, 128)):
    return TuningItem(table="fake", shape=shape, bucket=bucket)


def test_merge_pending_items_deduplicates_and_sorts():
    assert _merge_pending_items([[_item(8), _item(1)], [_item(4), _item(8)]]) == (
        _item(1),
        _item(4),
        _item(8),
    )


def test_shard_items_assigns_every_item_once():
    items = tuple(_item(bucket) for bucket in range(7))
    shards = [_shard_items(items, rank, 3) for rank in range(3)]

    assert set().union(*map(set, shards)) == set(items)
    assert sum(map(len, shards)) == len(items)


def test_merge_results_rejects_conflicts():
    item = _item(1)
    with pytest.raises(ValueError, match="Conflicting autotune results"):
        _merge_results([{item: {"BLOCK_SIZE": 16}}, {item: {"BLOCK_SIZE": 32}}])


def test_registry_rejects_duplicate_table_names():
    registry = TritonAutotuneRegistry()
    registry.register(_FakeTable())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_FakeTable())


def test_empty_registry_does_not_initialize_distributed_state():
    registry = TritonAutotuneRegistry()

    with patch("vllm.distributed.parallel_state.get_world_group") as get_world_group:
        registry.run(object())

    get_world_group.assert_not_called()


def test_distributed_run_tunes_rank_slice_and_commits_union():
    items = tuple(_item(bucket) for bucket in (1, 4, 8))
    remote_item = _item(2)
    remote_result = {remote_item: {"BLOCK_SIZE": 2}}
    table = _FakeTable(items)
    registry = TritonAutotuneRegistry()
    registry.register(table)
    world = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=object())
    gather_calls = 0

    def all_gather_object(output, value, group):
        nonlocal gather_calls
        assert group is world.cpu_group
        if gather_calls == 0:
            output[:] = [list(items) + [remote_item], list(items)]
        else:
            output[:] = [remote_result, value]
        gather_calls += 1

    with (
        patch("vllm.distributed.parallel_state.get_world_group", return_value=world),
        patch(
            "vllm.model_executor.warmup.triton_autotune."
            "torch.distributed.all_gather_object",
            side_effect=all_gather_object,
        ),
    ):
        registry.run(object())

    # Globally sorted buckets are 1, 2, 4, 8; rank 1 owns 2 and 8.
    assert table.tuned == [remote_item, _item(8)]
    assert table.committed == {
        remote_item: {"BLOCK_SIZE": 2},
        _item(8): {"BLOCK_SIZE": 8},
    }
    assert gather_calls == 2


def test_discovery_failure_still_reaches_both_collectives():
    class _FailingDiscoveryTable(_FakeTable):
        def pending_items(self, worker):
            raise RuntimeError("bad model shape")

    table = _FailingDiscoveryTable()
    registry = TritonAutotuneRegistry()
    registry.register(table)
    world = SimpleNamespace(world_size=1, rank_in_group=0, cpu_group=object())
    gathered_values = []

    def all_gather_object(output, value, group):
        gathered_values.append(value)
        output[0] = value

    with (
        patch("vllm.distributed.parallel_state.get_world_group", return_value=world),
        patch(
            "vllm.model_executor.warmup.triton_autotune."
            "torch.distributed.all_gather_object",
            side_effect=all_gather_object,
        ),
    ):
        registry.run(object())

    assert gathered_values == [[], {}]
    assert table.committed == {}


def test_tuning_failure_still_reaches_result_collective():
    item = _item(1)

    class _FailingTable(_FakeTable):
        def tune(self, item):
            raise RuntimeError("bad candidate")

    table = _FailingTable([item])
    registry = TritonAutotuneRegistry()
    registry.register(table)
    world = SimpleNamespace(world_size=1, rank_in_group=0, cpu_group=object())
    gathered_values = []

    def all_gather_object(output, value, group):
        gathered_values.append(value)
        output[0] = value

    with (
        patch("vllm.distributed.parallel_state.get_world_group", return_value=world),
        patch(
            "vllm.model_executor.warmup.triton_autotune."
            "torch.distributed.all_gather_object",
            side_effect=all_gather_object,
        ),
    ):
        registry.run(object())

    assert gathered_values == [[item], {}]
    assert table.committed == {}
