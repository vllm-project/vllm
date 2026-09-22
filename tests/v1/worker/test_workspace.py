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


def test_persistent_resources_follow_the_ubatch_override():
    """The scratch and the persistent cache have to land in the same slot.

    Reservation runs under ``use_workspace_ubatch_id`` rather than inside a real
    ubatch, so a persistent resource created there must be cached where the run
    will look for it. Resolving the ubatch from ``dbo_current_ubatch_id()``
    alone puts it in slot 0, and the lookup then misses once the manager is
    locked.
    """
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    for ubatch in range(2):
        for lane in range(2):
            with (
                workspace.use_workspace_ubatch_id(ubatch),
                workspace.use_workspace_lane(lane),
            ):
                assert manager._get_workspace_id() == manager._resolve_workspace_id()

    with workspace.use_workspace_ubatch_id(1):
        manager.get_persistent_resource("k", lambda: "made-for-ubatch-1")
    manager.lock()

    # The run reaches the same slot through the override the reservation used.
    with workspace.use_workspace_ubatch_id(1):
        assert manager.get_persistent_resource("k", lambda: "rebuilt") == (
            "made-for-ubatch-1"
        )

    # And ubatch 0 never saw it, so a locked lookup there still fails.
    with workspace.use_workspace_ubatch_id(0), pytest.raises(AssertionError):
        manager.get_persistent_resource("k", lambda: "rebuilt")
