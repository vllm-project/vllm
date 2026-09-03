# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the chunked host-registration helpers."""

from __future__ import annotations

import mmap
from unittest.mock import MagicMock

import pytest

import vllm.utils.host_memory as host_memory

pytestmark = pytest.mark.cpu_test

PAGE = mmap.PAGESIZE


def _mock_cudart(monkeypatch, registered: list[tuple[int, int]]) -> None:
    cudart = MagicMock()

    def _register(ptr: int, size: int, _flags: int) -> MagicMock:
        registered.append((ptr, size))
        return MagicMock(value=0)

    cudart.cudaHostRegister.side_effect = _register
    monkeypatch.setattr(host_memory.torch.cuda, "cudart", lambda: cudart)
    monkeypatch.setattr(host_memory.time, "sleep", MagicMock())


def test_chunked_registration_cuts_on_page_boundaries(monkeypatch):
    """Interior chunk cuts must land on page boundaries even for an unaligned
    base: cudaHostRegister locks whole pages and rejects a range overlapping
    an already-registered page, so a mid-page cut would fail."""
    registered: list[tuple[int, int]] = []
    _mock_cudart(monkeypatch, registered)

    base = 100 * PAGE + 64  # torch CPU allocations are only 64-byte aligned
    nbytes = 4 * PAGE
    ptrs = host_memory.host_register_chunked(base, nbytes, chunk_size_bytes=2 * PAGE)

    assert ptrs == [ptr for ptr, _ in registered]
    assert len(registered) > 1
    # The ranges must tile [base, base + nbytes) with no gap or overlap.
    pos = base
    for ptr, size in registered:
        assert ptr == pos
        assert ptr == base or ptr % PAGE == 0
        pos = ptr + size
    assert pos == base + nbytes


def test_chunked_registration_rejects_non_page_multiple():
    with pytest.raises(ValueError, match="page multiple"):
        host_memory.host_register_chunked(0, PAGE, chunk_size_bytes=100)


def test_chunked_registration_cuts_respect_copy_units(monkeypatch):
    """Chunking creates multiple pinned regions and a CUDA copy spanning two
    is rejected, so with a copy unit every cut must land on both the page
    grid and the copy-unit grid."""
    registered: list[tuple[int, int]] = []
    _mock_cudart(monkeypatch, registered)

    copy_unit = 61 * 512  # 31232 B, not a page multiple
    base = 4096 * PAGE  # page-aligned, as copy_unit_bytes requires
    nbytes = 40 * copy_unit
    ptrs = host_memory.host_register_chunked(
        base, nbytes, chunk_size_bytes=8 * PAGE, copy_unit_bytes=copy_unit
    )

    assert ptrs == [ptr for ptr, _ in registered]
    assert len(registered) > 1
    pos = base
    for ptr, size in registered:
        assert ptr == pos
        assert ptr % PAGE == 0
        assert (ptr - base) % copy_unit == 0
        pos = ptr + size
    assert pos == base + nbytes


def test_chunked_registration_rolls_back_on_failure(monkeypatch):
    """A failed registration must unregister the batches pinned so far and
    surface as HostRegisterError."""
    registered: list[tuple[int, int]] = []
    unregistered: list[int] = []
    cudart = MagicMock()

    def _register(ptr: int, size: int, _flags: int) -> MagicMock:
        registered.append((ptr, size))
        return MagicMock(value=0 if len(registered) < 2 else 2)

    def _unregister(ptr: int) -> MagicMock:
        unregistered.append(ptr)
        return MagicMock(value=0)

    cudart.cudaHostRegister.side_effect = _register
    cudart.cudaHostUnregister.side_effect = _unregister
    monkeypatch.setattr(host_memory.torch.cuda, "cudart", lambda: cudart)
    monkeypatch.setattr(host_memory.time, "sleep", MagicMock())

    base = 100 * PAGE
    with pytest.raises(host_memory.HostRegisterError):
        host_memory.host_register_chunked(base, 4 * PAGE, chunk_size_bytes=PAGE)
    assert unregistered == [base]


def test_chunked_registration_rejects_unaligned_base_with_copy_unit():
    with pytest.raises(ValueError, match="page-aligned"):
        host_memory.host_register_chunked(
            64, 4 * PAGE, chunk_size_bytes=PAGE, copy_unit_bytes=512
        )


def test_chunked_registration_rejects_non_positive_copy_unit():
    with pytest.raises(ValueError, match="positive"):
        host_memory.host_register_chunked(
            0, 4 * PAGE, chunk_size_bytes=PAGE, copy_unit_bytes=0
        )


def test_alloc_page_aligned_zeros():
    tensor = host_memory.alloc_page_aligned_zeros(10_000)
    assert tensor.data_ptr() % PAGE == 0
    assert tensor.nbytes == 10_000
    assert tensor.count_nonzero().item() == 0
