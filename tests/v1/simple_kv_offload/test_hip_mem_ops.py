# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HIP/ROCm paths in vllm.v1.simple_kv_offload.cuda_mem_ops.

Deliberately not skipped off ROCm: the version gate takes a plain version int
and the library loader is monkeypatched, so nothing here needs a HIP runtime
and the logic stays covered by the default (NVIDIA) CI that runs this
directory.
"""

import pytest

from vllm.v1.simple_kv_offload import cuda_mem_ops
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    _load_hip_runtime,
    _num_attrs_for_hip_version,
)


# hipMemcpyBatchAsync rejects numAttrs > 0 on ROCm 7.2.1-7.2.3 and accepts it
# on 7.13+. HIP encodes the version as major*10_000_000 + minor*100_000 + patch.
@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (0, 0),  # unknown (hipRuntimeGetVersion failed) -> conservative 0
        (70_100_000, 0),  # ROCm 7.1
        (70_200_000, 0),  # ROCm 7.2.0
        (70_253_211, 0),  # ROCm 7.2.2 (patch 53211)
        (70_300_000, 0),  # ROCm 7.3
        (71_200_000, 0),  # ROCm 7.12 -- just below the 7.13 cutoff
        (71_300_000, 1),  # ROCm 7.13 -- first release accepting numAttrs > 0
        (71_326_193, 1),  # ROCm 7.13 nightly (patch 26193)
        (71_400_000, 1),  # ROCm 7.14
        (71_500_000, 1),  # ROCm 7.15
        (80_000_000, 1),  # ROCm 8.0
    ],
)
def test_num_attrs_for_hip_version(version: int, expected: int):
    assert _num_attrs_for_hip_version(version) == expected


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        # Devel install: the unversioned symlink is present and preferred.
        ({"libamdhip64.so", "libamdhip64.so.7"}, "libamdhip64.so"),
        # Runtime-only / wheel-packaged ROCm: versioned soname only.
        ({"libamdhip64.so.7"}, "libamdhip64.so.7"),
        ({"libamdhip64.so.6"}, "libamdhip64.so.6"),
    ],
)
def test_load_hip_runtime_falls_back_to_versioned_soname(
    monkeypatch, available: set[str], expected: str
):
    loaded = []

    def fake_cdll(name, mode=0):
        if name not in available:
            raise OSError(f"{name}: cannot open shared object file")
        loaded.append(name)
        return name

    monkeypatch.setattr(cuda_mem_ops.ctypes, "CDLL", fake_cdll)
    assert _load_hip_runtime() == expected
    assert loaded == [expected]


def test_load_hip_runtime_reports_every_candidate_when_all_fail(monkeypatch):
    def fake_cdll(name, mode=0):
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(cuda_mem_ops.ctypes, "CDLL", fake_cdll)
    with pytest.raises(OSError, match="could not load the HIP runtime") as exc:
        _load_hip_runtime()
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        assert name in str(exc.value)
