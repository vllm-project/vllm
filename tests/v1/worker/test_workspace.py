# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.workspace import WorkspaceManager


def _workspace_sizes(manager: WorkspaceManager) -> list[int]:
    return [
        manager._workspace_size_bytes(workspace)
        for workspace in manager._current_workspaces
    ]


def test_get_simultaneous_keeps_all_ubatch_slots_in_sync():
    # Outside an active DBO region growing one ubatch slot grows them all,
    # so every slot is sized for the largest workspace before lock().
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)

    # round_up(300, 256) == 512
    manager.get_simultaneous(((300,), torch.uint8))
    assert _workspace_sizes(manager) == [512, 512]

    # A larger request grows every slot together as well.
    # round_up(600, 256) == 768
    manager.get_simultaneous(((600,), torch.uint8))
    assert _workspace_sizes(manager) == [768, 768]


def test_get_simultaneous_only_grows_active_slot_inside_dbo(monkeypatch):
    # Inside an active DBO region a sibling ubatch may still hold live views
    # into its slot, so only the requesting ubatch's slot is resized to avoid
    # orphaning that tensor (DBO leak). Siblings grow lazily later.
    monkeypatch.setattr("vllm.v1.worker.workspace.dbo_enabled", lambda: True)
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)

    # round_up(300, 256) == 512; active ubatch id defaults to 0.
    manager.get_simultaneous(((300,), torch.uint8))
    assert _workspace_sizes(manager) == [512, 0]


def test_lock_does_not_reallocate_buffers():
    # lock() must only flip the flag: relocating a buffer here could break a
    # CUDA graph captured before locking that already points at it.
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)
    manager._current_workspaces[0] = torch.empty(
        (512,), dtype=torch.uint8, device="cpu"
    )
    slot0 = manager._current_workspaces[0]

    manager.lock()

    assert manager.is_locked()
    assert manager._current_workspaces[0] is slot0
    assert manager._current_workspaces[1] is None


def test_workspace_lock_still_rejects_larger_allocations():
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)
    manager._current_workspaces[0] = torch.empty((16,), dtype=torch.uint8, device="cpu")

    manager.lock()

    with pytest.raises(AssertionError, match="Workspace is locked"):
        manager.get_simultaneous(((17,), torch.uint8))


def test_locked_execution_can_rotate_into_warmup_synced_slot(monkeypatch):
    # Regression for the warmup-touches-one-slot bug: warmup runs outside a
    # DBO region and sizes every slot in sync, so after locking, execution
    # rotating into a different ubatch slot still fits without triggering
    # (now forbidden) growth.
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)

    # Warmup on the default ubatch (id 0) sizes all slots together.
    manager.get_simultaneous(((300,), torch.uint8))
    manager.lock()
    assert _workspace_sizes(manager) == [512, 512]

    # Execution rotates into ubatch 1; a same-size request is served from the
    # pre-synced slot without raising the locked-growth assertion.
    monkeypatch.setattr("vllm.v1.worker.workspace.dbo_current_ubatch_id", lambda: 1)
    (view,) = manager.get_simultaneous(((300,), torch.uint8))
    assert view.numel() == 300
