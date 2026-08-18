# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for disk KV offload alignment and capacity configuration."""

from __future__ import annotations

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    _DISK_ONLY_KEYS,
    _get_direct_io_alignment,
)
from vllm.v1.simple_kv_offload.disk_backend import (
    _DEFAULT_DIRECT_IO_ALIGNMENT,
    _alloc_aligned,
    get_num_disk_slots,
    get_padded_slot_size,
)


def test_default_and_explicit_alignment():
    total_bpb = 4224

    assert _get_direct_io_alignment({}) == _DEFAULT_DIRECT_IO_ALIGNMENT
    assert _get_direct_io_alignment({"direct_io_alignment": 3000}) == 3000
    assert get_padded_slot_size(total_bpb) == 8192
    assert get_padded_slot_size(total_bpb, 3000) == 6000


def test_direct_io_alignment_is_disk_only():
    assert "direct_io_alignment" in _DISK_ONLY_KEYS


@pytest.mark.parametrize("alignment", [0, -1])
def test_direct_io_alignment_must_be_positive(alignment):
    with pytest.raises(ValueError, match="direct_io_alignment must be greater than 0"):
        _get_direct_io_alignment({"direct_io_alignment": alignment})


def test_staging_buffer_and_slot_offsets_use_selected_alignment():
    total_bpb = 4224
    alignment = 3000
    padded_slot_size = get_padded_slot_size(total_bpb, alignment)
    buffer = _alloc_aligned(2, padded_slot_size, alignment)

    assert buffer.data_ptr() % alignment == 0
    assert padded_slot_size % alignment == 0
    assert all((slot * padded_slot_size) % alignment == 0 for slot in range(2))


def test_slot_capacity_uses_padded_size():
    total_bpb = 4224
    capacity = 2 * total_bpb
    padded_slot_size = get_padded_slot_size(total_bpb)

    assert get_num_disk_slots(capacity, total_bpb) == 1
    assert get_num_disk_slots(capacity, total_bpb) * padded_slot_size <= capacity
    assert get_num_disk_slots(capacity, total_bpb, use_page_cache=True) == 2


def test_page_cache_mode_uses_raw_slot_size():
    total_bpb = 4224

    assert get_padded_slot_size(total_bpb, 4096, use_page_cache=True) == total_bpb
