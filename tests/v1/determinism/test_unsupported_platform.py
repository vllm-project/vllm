# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant mode must refuse platforms it cannot make invariant.

Only CUDA and XPU install the matmul overrides (or pin cuBLAS split-k) that
batch invariance depends on. Any other platform reaching
`enable_batch_invariant_mode` would previously register the softmax/mean/bmm
overrides on its own dispatch key and report success, leaving matmul
batch-variant.
"""

import pytest
import torch

from vllm.model_executor.determinism import batch_invariant

pytestmark = pytest.mark.cpu_test


class _StubPlatform:
    """Stands in for a platform that has no batch-invariant implementation."""

    def __init__(self, *, cuda: bool = False, xpu: bool = False) -> None:
        self._cuda = cuda
        self._xpu = xpu
        self.dispatch_key = "CPU"

    def is_cuda(self) -> bool:
        return self._cuda

    def is_xpu(self) -> bool:
        return self._xpu


@pytest.fixture(autouse=True)
def _reset_mode(monkeypatch: pytest.MonkeyPatch):
    """Keep the module's one-way mode flag and torch.bmm patch out of other
    tests."""
    monkeypatch.setattr(batch_invariant, "_batch_invariant_MODE", False)
    monkeypatch.setattr(batch_invariant, "_batch_invariant_LIB", None)
    monkeypatch.setattr(torch, "bmm", torch.bmm)


def test_unsupported_platform_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(batch_invariant, "current_platform", _StubPlatform())

    with pytest.raises(NotImplementedError, match="_StubPlatform"):
        batch_invariant.enable_batch_invariant_mode()


def test_unsupported_platform_leaves_no_side_effects(
    monkeypatch: pytest.MonkeyPatch,
):
    """The refusal must happen before any global state is mutated, so a caller
    that handles the error is not left with a half-installed mode."""
    monkeypatch.setattr(batch_invariant, "current_platform", _StubPlatform())
    original_bmm = torch.bmm

    with pytest.raises(NotImplementedError):
        batch_invariant.enable_batch_invariant_mode()

    assert batch_invariant._batch_invariant_MODE is False
    assert batch_invariant._batch_invariant_LIB is None
    assert torch.bmm is original_bmm


def test_already_enabled_returns_before_the_platform_check(
    monkeypatch: pytest.MonkeyPatch,
):
    """The existing idempotence guard still short-circuits, so an enabled mode
    is never turned into an error by this check."""
    monkeypatch.setattr(batch_invariant, "current_platform", _StubPlatform())
    monkeypatch.setattr(batch_invariant, "_batch_invariant_MODE", True)

    batch_invariant.enable_batch_invariant_mode()
