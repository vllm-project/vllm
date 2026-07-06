# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.executor.vllm_net_devices import (
    _dp_adjusted_local_rank,
    normalize_pci,
)


def _make_vllm_config(**parallel_overrides):
    from types import SimpleNamespace

    defaults = {
        "assigned_physical_gpu_ids": None,
        "data_parallel_backend": "mp",
        "data_parallel_external_lb": False,
        "data_parallel_index": 0,
        "data_parallel_rank_local": None,
        "distributed_executor_backend": "mp",
        "nnodes_within_dp": 1,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
    }
    defaults.update(parallel_overrides)
    return SimpleNamespace(parallel_config=SimpleNamespace(**defaults))


@pytest.mark.parametrize(
    "addr, expected",
    [
        ("0000:3f:00.0", (0, 0x3F, 0, 0)),
        ("0001:00:00.0", (1, 0, 0, 0)),
        ("00000001:00:00.0", (1, 0, 0, 0)),
        ("0009:00:00.0", (9, 0, 0, 0)),
        ("0105:00:00.0", (0x105, 0, 0, 0)),
        ("0000:0a:1f.7", (0, 0x0A, 0x1F, 7)),
    ],
)
def test_normalize_pci_full_domain(addr, expected):
    assert normalize_pci(addr) == expected


@pytest.mark.parametrize(
    "addr, expected",
    [
        ("01:00.0", (0, 1, 0, 0)),
        ("3f:00.0", (0, 0x3F, 0, 0)),
        ("40:00.0", (0, 0x40, 0, 0)),
        ("ff:1f.7", (0, 0xFF, 0x1F, 7)),
    ],
)
def test_normalize_pci_short_form(addr, expected):
    assert normalize_pci(addr) == expected


def test_normalize_pci_case_insensitive():
    assert normalize_pci("0A:1F.7") == normalize_pci("0a:1f.7")


def test_normalize_pci_strips_whitespace():
    assert normalize_pci("  0001:00:00.0  ") == (1, 0, 0, 0)


def test_normalize_pci_strips_0x_prefix():
    assert normalize_pci("0x0001:00:00.0") == (1, 0, 0, 0)


def test_normalize_pci_missing_function_raises():
    with pytest.raises(ValueError, match="missing function suffix"):
        normalize_pci("0001:00:00")


def test_normalize_pci_invalid_function_char_raises():
    with pytest.raises(ValueError, match="invalid PCI function"):
        normalize_pci("0001:00:00.z")


def test_normalize_pci_too_many_segments_raises():
    with pytest.raises(ValueError, match="invalid PCI BDF"):
        normalize_pci("a:b:c:d.0")


def test_normalize_pci_bus_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        normalize_pci("0000:1ff:00.0")


def test_normalize_pci_device_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        normalize_pci("0000:00:20.0")


def test_normalize_pci_empty_string_raises():
    with pytest.raises(ValueError):
        normalize_pci("")


@pytest.mark.parametrize("tp_local_rank", [0, 1])
def test_dp_adjusted_local_rank_skips_external_lb(tp_local_rank):
    config = _make_vllm_config(
        assigned_physical_gpu_ids=[2, 3],
        data_parallel_external_lb=True,
        data_parallel_index=1,
        tensor_parallel_size=2,
    )

    assert _dp_adjusted_local_rank(tp_local_rank, config) == tp_local_rank


def test_dp_adjusted_local_rank_preserves_full_node_assigned_gpu_ids():
    config = _make_vllm_config(
        assigned_physical_gpu_ids=[0, 1, 2, 3],
        data_parallel_index=2,
    )

    assert _dp_adjusted_local_rank(0, config) == 2
