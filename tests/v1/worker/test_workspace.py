# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker import workspace as workspace_module


def test_persistent_workspace_is_stable_and_isolated(monkeypatch):
    ubatch_id = 0
    monkeypatch.setattr(workspace_module, "dbo_current_ubatch_id", lambda: ubatch_id)
    manager = workspace_module.WorkspaceManager(torch.device("cpu"), num_ubatches=2)

    key = ("locks", "main")
    first = manager.get_persistent(key, (8,), torch.int32, zero_init=True)
    assert torch.count_nonzero(first) == 0
    first.fill_(1)
    assert manager.get_persistent(key, (8,), torch.int32) is first

    manager.get_simultaneous(((1024,), torch.uint8))
    assert manager.get_persistent(key, (8,), torch.int32) is first

    other = manager.get_persistent(("locks", "aux"), (8,), torch.int32)
    assert other is not first

    ubatch_id = 1
    second = manager.get_persistent(key, (8,), torch.int32, zero_init=True)
    assert second is not first
    assert torch.count_nonzero(second) == 0


def test_persistent_workspace_rejects_changes_after_allocation():
    manager = workspace_module.WorkspaceManager(torch.device("cpu"))
    workspace = manager.get_persistent("locks", (8,), torch.int32)

    with pytest.raises(ValueError, match="requested shape"):
        manager.get_persistent("locks", (16,), torch.int32)
    with pytest.raises(ValueError, match="requested shape"):
        manager.get_persistent("locks", (8,), torch.int64)

    manager.lock()
    assert manager.get_persistent("locks", (8,), torch.int32) is workspace
    with pytest.raises(AssertionError, match="was not allocated during warmup"):
        manager.get_persistent("new", (8,), torch.int32)
