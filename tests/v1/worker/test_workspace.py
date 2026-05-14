# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for WorkspaceManager.

Focused on the lock semantics that turboquant_attn._decode_attention
depends on: a locked workspace that is already large enough must still
hand out tensor views, while a locked workspace that would need to grow
must return None (via try_get_simultaneous) instead of raising — so the
caller can fall back to per-call allocation. Repros the failure mode in
vllm-project/vllm#42544.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.v1.worker.workspace import WorkspaceManager


@pytest.fixture
def manager() -> WorkspaceManager:
    return WorkspaceManager(device=torch.device("cpu"), num_ubatches=1)


@pytest.fixture
def dbo_manager() -> WorkspaceManager:
    return WorkspaceManager(device=torch.device("cpu"), num_ubatches=2)


def test_get_simultaneous_grows_when_unlocked(manager: WorkspaceManager) -> None:
    a, b = manager.get_simultaneous(
        ((16,), torch.float32),
        ((8,), torch.float32),
    )
    assert a.shape == (16,)
    assert b.shape == (8,)
    assert a.dtype == torch.float32


def test_get_simultaneous_raises_on_locked_undersized(
    manager: WorkspaceManager,
) -> None:
    manager.lock()
    with pytest.raises(AssertionError, match="Workspace is locked"):
        manager.get_simultaneous(((1024,), torch.float32))


def test_get_simultaneous_succeeds_on_locked_when_fits(
    manager: WorkspaceManager,
) -> None:
    manager.get_simultaneous(((1024,), torch.float32))
    manager.lock()
    (t,) = manager.get_simultaneous(((128,), torch.float32))
    assert t.shape == (128,)


def test_try_get_simultaneous_returns_none_when_locked_undersized(
    manager: WorkspaceManager,
) -> None:
    """The failure mode from issue #42544: workspace locked at 0 bytes,
    decode-time allocation requested. Must not raise."""
    manager.lock()
    result = manager.try_get_simultaneous(
        ((24, 4, 8, 257), torch.float32),
        ((24, 4, 256), torch.bfloat16),
        ((24, 4), torch.float32),
    )
    assert result is None


def test_try_get_simultaneous_returns_views_when_locked_and_fits(
    manager: WorkspaceManager,
) -> None:
    manager.get_simultaneous(
        ((24, 4, 8, 257), torch.float32),
        ((24, 4, 256), torch.bfloat16),
        ((24, 4), torch.float32),
    )
    manager.lock()
    result = manager.try_get_simultaneous(
        ((24, 4, 8, 257), torch.float32),
        ((24, 4, 256), torch.bfloat16),
        ((24, 4), torch.float32),
    )
    assert result is not None
    mid, out, lse = result
    assert mid.shape == (24, 4, 8, 257)
    assert out.shape == (24, 4, 256)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (24, 4)


def test_try_get_simultaneous_grows_when_unlocked(
    manager: WorkspaceManager,
) -> None:
    result = manager.try_get_simultaneous(((1024,), torch.float32))
    assert result is not None
    (t,) = result
    assert t.shape == (1024,)


def test_reserve_sizes_every_ubatch_slot(dbo_manager: WorkspaceManager) -> None:
    """In DBO setups, reservation at init must hit every ubatch slot —
    otherwise lock_workspace() snapshots sibling ubatches at 0 bytes and
    they hit the slow fallback on every forward."""
    shapes = (
        ((24, 4, 8, 257), torch.float32),
        ((24, 4, 256), torch.bfloat16),
        ((24, 4), torch.float32),
    )
    dbo_manager.reserve(*shapes)
    dbo_manager.lock()
    for ubatch_id in range(2):
        with patch(
            "vllm.v1.worker.workspace.dbo_current_ubatch_id", return_value=ubatch_id
        ):
            result = dbo_manager.try_get_simultaneous(*shapes)
            assert result is not None, f"ubatch {ubatch_id} fell back to None"
            mid, out, lse = result
            assert mid.shape == (24, 4, 8, 257)
            assert out.shape == (24, 4, 256)
            assert lse.shape == (24, 4)


def test_get_simultaneous_does_not_size_sibling_ubatch(
    dbo_manager: WorkspaceManager,
) -> None:
    """Conversely, get_simultaneous on the active ubatch must NOT
    proactively size siblings — that would orphan their views mid-run
    (the DBO leak warned about in _ensure_workspace_size)."""
    with patch("vllm.v1.worker.workspace.dbo_current_ubatch_id", return_value=0):
        dbo_manager.get_simultaneous(((1024,), torch.float32))
    dbo_manager.lock()
    with patch("vllm.v1.worker.workspace.dbo_current_ubatch_id", return_value=1):
        assert dbo_manager.try_get_simultaneous(((128,), torch.float32)) is None
