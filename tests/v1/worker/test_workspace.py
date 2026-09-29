# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.workspace as workspace
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import _num_workspace_lanes


class _SpecConfig:
    def __init__(self, dspark: bool) -> None:
        self._dspark = dspark

    def use_dspark(self) -> bool:
        return self._dspark


class _VllmConfig:
    def __init__(self, spec_config: _SpecConfig | None) -> None:
        self.speculative_config = spec_config


@pytest.mark.parametrize(
    ("use_v2_model_runner", "spec_config", "expected"),
    [
        (True, _SpecConfig(True), 2),
        (False, _SpecConfig(True), 1),
        (True, _SpecConfig(False), 1),
        (True, None, 1),
    ],
)
def test_workspace_lane_count_is_dspark_only(
    use_v2_model_runner: bool,
    spec_config: _SpecConfig | None,
    expected: int,
) -> None:
    config = cast(VllmConfig, _VllmConfig(spec_config))
    assert _num_workspace_lanes(config, use_v2_model_runner) == expected


def test_workspace_lanes_do_not_alias_and_restore_context(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    assert manager._current_workspaces == [None, None, None, None]

    (target,) = manager.get_simultaneous(((512,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft,) = manager.get_simultaneous(((256,), torch.uint8))
        (draft_reused,) = manager.get_simultaneous(((8,), torch.uint8))
    (target_reused,) = manager.get_simultaneous(((8,), torch.uint8))

    assert manager._current_workspaces[0].numel() == 512  # type: ignore[union-attr]
    assert manager._current_workspaces[1].numel() == 256  # type: ignore[union-attr]
    assert manager._current_workspaces[2:] == [None, None]
    assert target.data_ptr() != draft.data_ptr()
    assert draft.data_ptr() == draft_reused.data_ptr()
    assert target.data_ptr() == target_reused.data_ptr()


def test_workspace_lanes_compose_with_ubatches(monkeypatch) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    pointers = set()
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                (buffer,) = manager.get_simultaneous(((16,), torch.uint8))
                pointers.add(buffer.data_ptr())

    assert len(pointers) == 4


def test_workspace_lock_blocks_growth_and_unlock_restores(monkeypatch) -> None:
    """Once locked, oversized requests fail loudly instead of reallocating the
    buffer that captured CUDA graphs point at; unlock restores growth."""
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    (buf,) = manager.get_simultaneous(((256,), torch.uint8))
    manager.lock()
    assert manager.is_locked()

    # Requests within the reserved size still reuse the same buffer.
    (same,) = manager.get_simultaneous(((256,), torch.uint8))
    (smaller,) = manager.get_simultaneous(((8,), torch.uint8))
    assert same.data_ptr() == buf.data_ptr()
    assert smaller.data_ptr() == buf.data_ptr()

    with pytest.raises(AssertionError, match="Workspace is locked"):
        manager.get_simultaneous(((512,), torch.uint8))

    manager.unlock()
    (grown,) = manager.get_simultaneous(((512,), torch.uint8))
    assert grown.numel() == 512


def test_reserve_simultaneous_sizes_every_ubatch_in_current_lane(
    monkeypatch,
) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    manager._reserve_simultaneous(
        ((64,), torch.float32),
        ((16,), torch.float32),
    )
    assert manager._current_workspaces[0] is not None
    assert manager._current_workspaces[2] is not None
    assert manager._current_workspaces[1] is None
    assert manager._current_workspaces[3] is None

    manager.lock()
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        first, second = manager.get_simultaneous(
            ((64,), torch.float32),
            ((16,), torch.float32),
        )
        assert first.numel() == 64
        assert second.numel() == 16

    with pytest.raises(RuntimeError, match="initialization-only"):
        manager._reserve_simultaneous(((64,), torch.float32))


def test_workspace_lane_validation(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    with (
        pytest.raises(ValueError, match="non-negative"),
        workspace.use_workspace_lane(-1),
    ):
        pass

    with (
        workspace.use_workspace_lane(1),
        pytest.raises(RuntimeError, match="is not configured"),
    ):
        manager.get_simultaneous(((1,), torch.uint8))

    with pytest.raises(ValueError, match="at least one"):
        workspace.WorkspaceManager(torch.device("cpu"), num_lanes=0)


@pytest.mark.parametrize("dim,hq,hk", [(128, 16, 1), (256, 12, 2)])
@pytest.mark.parametrize("fp8", [False, True])
def test_segmented_attention_reservation_covers_ragged_query_caps(
    monkeypatch, dim, hq, hk, fp8
):
    """Every supported query bucket fits the startup buffer after locking."""
    from vllm.v1.attention.ops import segmented_attention as segmented

    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)
    monkeypatch.setattr(segmented, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(segmented, "current_workspace_manager", lambda: manager)
    segmented.reserve_segmented_attention_workspace(
        32,
        hq,
        hk,
        dim,
        65536,
        fp8=fp8,
    )
    manager.lock()
    pointers = set()
    query_lengths = {
        query_len
        for capacity in segmented._query_capacity_buckets()
        for query_len in (max(1, capacity // 2 + 1), capacity)
    }
    for batch in range(1, 33):
        for query_len in query_lengths:
            qcap = segmented.segmented_query_capacity(query_len)
            cfg = segmented.select_segmented_config(
                batch, query_len, 65536, hq, hk, dim, fp8
            )
            shapes = segmented.segmented_workspace_shapes(
                batch, qcap, hq, hk, dim, cfg["splits"]
            )
            if shapes is not None:
                partial, lse = manager.get_simultaneous(
                    (shapes[0], torch.float32), (shapes[1], torch.float32)
                )
                pointers.add(partial.untyped_storage().data_ptr())
                assert partial.data_ptr() != lse.data_ptr()
    assert len(pointers) == 1


def test_segmented_attention_reservation_respects_scheduler_token_limit(
    monkeypatch,
) -> None:
    """Workspace planning excludes batch/query pairs the scheduler cannot form."""
    from vllm.v1.attention.ops import segmented_attention as segmented

    calls = []

    def record_config(batch, query_len, *_args):
        calls.append((batch, query_len))
        return {"splits": 1}

    monkeypatch.setattr(segmented, "is_workspace_manager_initialized", lambda: True)
    monkeypatch.setattr(segmented, "select_segmented_config", record_config)
    segmented.reserve_segmented_attention_workspace(
        32,
        16,
        2,
        128,
        65536,
        max_tokens=8,
    )

    assert calls
    assert all(batch + query_len - 1 <= 8 for batch, query_len in calls)
    assert (8, 1) in calls
    assert (1, 8) in calls


def test_segmented_attention_query_capacity_buckets() -> None:
    from vllm.v1.attention.ops.segmented_attention import (
        MAX_QUERY_LEN,
        segmented_query_capacity,
    )

    assert [
        segmented_query_capacity(query_len)
        for query_len in (1, 2, 3, 33, 129, 1025, 2049, MAX_QUERY_LEN)
    ] == [1, 2, 4, 64, 256, 2048, 4096, 8192]
    with pytest.raises(ValueError, match="Query length"):
        segmented_query_capacity(MAX_QUERY_LEN + 1)


def test_persistent_resources_are_lazy_and_isolated(monkeypatch) -> None:
    """Only the requesting slot allocates; each ubatch/lane owns its resource."""
    active_ubatch = 0
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )
    factory = Mock(side_effect=lambda: {"buffer": torch.zeros(8)})
    resources = []
    for ubatch in range(2):
        active_ubatch = ubatch
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                resource = manager.get_persistent_resource("scratch", factory)
                resources.append(resource)
                resource["buffer"].fill_(len(resources))
                assert manager.get_persistent_resource("scratch", factory) is resource
                assert factory.call_count == len(resources)

    assert len({r["buffer"].data_ptr() for r in resources}) == 4
    for i, resource in enumerate(resources, 1):
        assert torch.all(resource["buffer"] == i)


def test_persistent_resource_lock_applies_to_each_slot(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=2)
    factory = Mock(side_effect=object)
    first = manager.get_persistent_resource("scratch", factory)
    manager.lock()
    assert manager.get_persistent_resource("scratch", factory) is first
    with pytest.raises(AssertionError, match="was not allocated during warmup"):
        manager.get_persistent_resource("new", factory)
    with workspace.use_workspace_lane(1):
        with pytest.raises(AssertionError, match="was not allocated during warmup"):
            manager.get_persistent_resource("scratch", factory)
        assert factory.call_count == 1
        manager.unlock()
        assert manager.get_persistent_resource("scratch", factory) is not first
    assert manager.get_persistent_resource("scratch", factory) is first


def test_persistent_tensor_preserves_contents_and_rejects_changes() -> None:
    """Initialization happens once and transient scratch cannot overwrite locks."""
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    first = manager.get_persistent("locks", (8,), torch.int32, zero_init=True)
    assert torch.count_nonzero(first) == 0
    first.fill_(1)
    (scratch,) = manager.get_simultaneous(((1024,), torch.uint8))
    scratch.zero_()
    assert manager.get_persistent("locks", (8,), torch.int32, zero_init=True) is first
    assert torch.all(first == 1)
    with pytest.raises(ValueError, match="requested shape"):
        manager.get_persistent("locks", (16,), torch.int32)
    with pytest.raises(ValueError, match="requested shape"):
        manager.get_persistent("locks", (8,), torch.int64)
    manager.lock()
    assert manager.get_persistent("locks", (8,), torch.int32) is first
    with pytest.raises(AssertionError, match="was not allocated during warmup"):
        manager.get_persistent("new", (8,), torch.int32)
