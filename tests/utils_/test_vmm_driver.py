# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

import pytest
import torch

from vllm.utils.vmm_driver import _find_loaded_library

_CUDART = "/usr/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
_MAPS = [
    f"7f00-7f01 r-xp 00000000 08:01 1 {_CUDART}",
    "7f02-7f03 r-xp 00000000 08:01 2 [heap]",
    "7f04-7f05 r-xp 00000000 08:01 3 /usr/lib64/libcuda.so.580.65.06",
    "7f06-7f07 r-xp 00000000 08:01 4 /usr/lib64/libcuda.so.1",
]


def test_find_loaded_library_matches_the_file_name_not_a_substring():
    """Torch maps libcudart before the driver; the driver lookup must not pick
    it (its symbols differ), nor match a path component."""
    assert _find_loaded_library("libcuda", _MAPS) == "/usr/lib64/libcuda.so.580.65.06"
    assert _find_loaded_library("libcudart", _MAPS) == _CUDART
    assert _find_loaded_library("libamdhip64", _MAPS) is None
    assert _find_loaded_library("libcuda", ["garbage"]) is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA driver context handling",
)
def test_cuda_driver_context_usable_before_torch_initializes_the_device():
    """The VMM probe may run before torch touches the device; the driver must
    initialize itself rather than fail with a bare 'not initialized' error."""
    script = """
from vllm.utils.vmm_driver import get_vmm_driver
driver = get_vmm_driver()
driver.ensure_context(0)
driver.ensure_context(0)
print(driver.granularity(0))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert int(result.stdout.strip().splitlines()[-1]) > 0


class _SetAccessModel:
    """Models HIP's set-access size check over a reservation's allocations,
    kept in ``list_order`` (the runtime's internal order, which callers do not
    control)."""

    granule = 4096

    def __init__(self, fixed: bool, reverse_list_order: bool) -> None:
        self.fixed = fixed
        self.reverse = reverse_list_order
        self.chunks: list[tuple[int, int]] = []
        self.live: set[str] = set()

    def granularity(self, device_index: int) -> int:
        return self.granule

    def reserve(self, size: int) -> int:
        self.live.add("reservation")
        return 1 << 20

    def free_reserved(self, ptr: int, size: int) -> None:
        self.live.discard("reservation")

    def create(self, size: int, device_index: int) -> int:
        self.live.add(f"handle{size}")
        return size

    def release(self, handle: int) -> None:
        self.live.discard(f"handle{handle}")

    def map(self, ptr: int, size: int, handle: int) -> None:
        self.chunks.append((ptr, size))

    def unmap(self, ptr: int, size: int) -> None:
        self.chunks.remove((ptr, size))

    def set_access(self, ptr: int, size: int, device_index: int) -> None:
        if self.fixed:
            start = [p for p, _ in self.chunks].index(ptr)
            ordered = sorted(self.chunks)[start:]
        else:
            ordered = self.chunks[::-1] if self.reverse else list(self.chunks)
        covered = 0
        for _, chunk_size in ordered:
            covered += chunk_size
            if covered >= size:
                break
        if covered != size:
            raise RuntimeError("GPU driver error 1: invalid argument")


@pytest.mark.parametrize("reverse_list_order", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_set_access_probe_detects_the_hip_defect(fixed, reverse_list_order):
    """The probe flags runtimes that count a reservation's allocations from the
    first in their list (ROCm/rocm-systems#2516) whatever that list's order,
    passes fixed ones, and frees everything it maps either way."""
    from vllm.utils.vmm_driver import _set_access_fails_within_reservation

    model = _SetAccessModel(fixed, reverse_list_order)
    assert _set_access_fails_within_reservation(model, 0) is (not fixed)
    assert model.chunks == []
    assert model.live == set()


def test_driver_prefers_the_library_already_loaded(monkeypatch):
    """A ROCm torch wheel bundles its own HIP runtime; loading another copy by
    name would put two runtimes in the process, so the mapped one wins."""
    from types import SimpleNamespace

    from vllm.utils import vmm_driver

    opened: list[str] = []

    def fake_cdll(name: str) -> str:
        opened.append(name)
        return name

    monkeypatch.setattr(vmm_driver.ctypes, "CDLL", fake_cdll)
    driver = SimpleNamespace(
        _lib_search_name="libamdhip64",
        _lib_candidates=("libamdhip64.so.7", "libamdhip64.so"),
    )
    bundled = "/venv/lib/python3.12/site-packages/torch/lib/libamdhip64.so.7"
    monkeypatch.setattr(vmm_driver, "_find_loaded_library", lambda name: bundled)
    assert vmm_driver.VmmDriver._load_library(driver) == bundled

    monkeypatch.setattr(vmm_driver, "_find_loaded_library", lambda name: None)
    opened.clear()
    assert vmm_driver.VmmDriver._load_library(driver) == "libamdhip64.so.7"
    assert opened == ["libamdhip64.so.7"]
