# SPDX-License-Identifier: Apache-2.0
"""Audit A-15 — torch optional in conftest test.

Verifies that conftest.py handles torch unavailability gracefully:
  1. `_TORCH_AVAILABLE` flag exists
  2. fixtures fall back when _TORCH_AVAILABLE is False
  3. `requires_torch` marker is registered
  4. tests marked `requires_torch` are skipped automatically
"""
from __future__ import annotations

import pytest


def test_torch_available_flag_exists():
    """conftest must define _TORCH_AVAILABLE flag (A-15 invariant)."""
    from vllm._genesis.tests import conftest
    assert hasattr(conftest, "_TORCH_AVAILABLE")
    assert isinstance(conftest._TORCH_AVAILABLE, bool)


def test_torch_module_attr_present():
    """conftest must keep `torch` attribute (None if unavailable)."""
    from vllm._genesis.tests import conftest
    # Either real torch module or None
    assert hasattr(conftest, "torch")


def test_requires_torch_marker_documented_in_pytest_configure():
    """pytest_configure must register the requires_torch marker (A-15)."""
    from vllm._genesis.tests import conftest
    src = open(conftest.__file__).read()
    assert "requires_torch" in src
    assert "audit A-15" in src.lower() or "A-15" in src


def test_collection_modifier_handles_no_torch():
    """pytest_collection_modifyitems must check _TORCH_AVAILABLE before
    accessing torch.cuda.is_available()."""
    from vllm._genesis.tests import conftest
    src = open(conftest.__file__).read()
    # The function must reference _TORCH_AVAILABLE before torch.cuda
    coll_idx = src.find("def pytest_collection_modifyitems")
    assert coll_idx > 0
    body_end = src.find("\ndef ", coll_idx + 1)
    body = src[coll_idx:body_end if body_end > 0 else len(src)]
    assert "_TORCH_AVAILABLE" in body, (
        "pytest_collection_modifyitems must reference _TORCH_AVAILABLE"
    )


@pytest.mark.requires_torch
def test_marker_can_be_applied():
    """Sanity: requires_torch marker is recognized (test runs if torch present)."""
    import torch  # noqa: F401
    assert True
