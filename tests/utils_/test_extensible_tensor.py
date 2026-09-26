# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm.utils.extensible_tensor import ExtensibleTensor
from vllm.utils.vmm_driver import vmm_unavailable_reason

# Each test runs in its own process: touching the driver here would initialize
# CUDA in the pytest parent and break later fork-based tests in the shard.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _skip_unless_vmm() -> None:
    if (reason := vmm_unavailable_reason()) is not None:
        pytest.skip(f"VMM unavailable: {reason}")


@create_new_process_for_each_test("spawn")
def test_extensible_tensor_grows_without_moving() -> None:
    _skip_unless_vmm()
    buffer = ExtensibleTensor(4096, device="cuda")
    try:
        base_ptr = buffer.base_ptr
        buffer.resize_per_segment_(1024)
        first_view = buffer.segment_view(0)
        assert first_view.data_ptr() == base_ptr
        assert first_view.untyped_storage().nbytes() == 1024
        first_view.fill_(7)

        buffer.resize_per_segment_(2048)
        second_view = buffer.segment_view(0)
        assert second_view.data_ptr() == base_ptr
        assert second_view.untyped_storage().nbytes() == 2048
        assert torch.equal(second_view[:1024], torch.full_like(second_view[:1024], 7))

        full_view = buffer.full_view()
        assert full_view.data_ptr() == base_ptr
        assert full_view.numel() == 4096
    finally:
        buffer.free()


@create_new_process_for_each_test("spawn")
def test_extensible_tensor_rejects_shrink_and_overflow() -> None:
    _skip_unless_vmm()
    buffer = ExtensibleTensor(1024, device="cuda")
    try:
        buffer.resize_per_segment_(512)
        with pytest.raises(ValueError, match="grow-only"):
            buffer.resize_per_segment_(256)
        with pytest.raises(ValueError, match="exceeds the segment capacity"):
            buffer.resize_per_segment_(1025)
    finally:
        buffer.free()


@create_new_process_for_each_test("spawn")
def test_segments_grow_in_lockstep_and_zero_new() -> None:
    _skip_unless_vmm()
    """Each segment's committed prefix grows in lockstep.

    Data written to a segment's committed prefix survives a grow; the newly
    committed range of each segment is zeroed with `zero_new=True` while old
    bytes are preserved.
    """
    et = ExtensibleTensor(max_num_bytes=8192, device="cuda", num_segments=2)
    try:
        assert et.num_segments == 2
        assert et.segment_capacity_bytes == 4096

        et.resize_per_segment_(256, zero_new=True)
        assert et.bytes_per_segment == 256
        assert et.num_bytes == 512
        fv = et.full_view()
        assert fv.shape == (8192,)
        assert torch.count_nonzero(fv[:256]) == 0
        assert torch.count_nonzero(fv[4096 : 4096 + 256]) == 0

        pattern_a = torch.arange(256, device="cuda", dtype=torch.uint8)
        pattern_b = 255 - pattern_a
        fv[:256].copy_(pattern_a)
        fv[4096 : 4096 + 256].copy_(pattern_b)

        et.resize_per_segment_(1024, zero_new=True)
        fv2 = et.full_view()
        assert fv2.data_ptr() == fv.data_ptr()
        assert torch.equal(fv2[:256], pattern_a)
        assert torch.equal(fv2[4096 : 4096 + 256], pattern_b)
        assert torch.count_nonzero(fv2[256:1024]) == 0
        assert torch.count_nonzero(fv2[4096 + 256 : 4096 + 1024]) == 0

        seg1 = et.segment_view(1)
        assert seg1.data_ptr() == fv.data_ptr() + 4096
        assert seg1.untyped_storage().nbytes() == 1024
        assert torch.equal(seg1[:256], pattern_b)
    finally:
        et.free()


@create_new_process_for_each_test("spawn")
def test_segments_at_granularity_scale() -> None:
    _skip_unless_vmm()
    """Segments spanning multiple mapping granules commit correctly.

    Uses a segment capacity that is not a multiple of the allocation
    granularity, so a granule straddles the segment boundary and is shared by
    the first commit of one segment and a later commit of the other -- it must
    be mapped exactly once.
    """
    probe = ExtensibleTensor(max_num_bytes=1, device="cuda")
    granularity = probe.granularity
    assert probe.capacity_bytes == granularity
    probe.free()
    max_num_bytes = 3 * granularity
    et = ExtensibleTensor(max_num_bytes=max_num_bytes, device="cuda", num_segments=2)
    try:
        seg = et.segment_capacity_bytes
        assert seg == max_num_bytes // 2

        step = granularity // 2
        et.resize_per_segment_(step, zero_new=True)
        fv = et.full_view()
        fv[:step].fill_(1)
        fv[seg : seg + step].fill_(2)

        et.resize_per_segment_(seg, zero_new=True)
        assert et.physical_bytes == max_num_bytes
        fv2 = et.full_view()
        assert torch.all(fv2[:step] == 1)
        assert torch.all(fv2[seg : seg + step] == 2)
        assert torch.count_nonzero(fv2[step:seg]) == 0
        assert torch.count_nonzero(fv2[seg + step :]) == 0
    finally:
        et.free()


@create_new_process_for_each_test("spawn")
def test_unequal_segments_grow_independently() -> None:
    _skip_unless_vmm()
    """Segments of different capacities commit different byte counts, each a
    prefix of its own range, and expose their own views."""
    et = ExtensibleTensor(
        max_num_bytes=12288, device="cuda", segment_capacities=[4096, 8192]
    )
    try:
        assert et.num_segments == 2
        assert et.segment_offsets == [0, 4096]
        with pytest.raises(ValueError, match="differ"):
            _ = et.segment_capacity_bytes
        et.resize_segments_([1024, 2048], zero_new=True)
        assert et.num_bytes == 3072
        assert et.segment_view(0).untyped_storage().nbytes() == 1024
        assert et.segment_view(1).untyped_storage().nbytes() == 2048
        assert et.segment_view(1).data_ptr() == et.base_ptr + 4096
        with pytest.raises(ValueError, match="grow-only"):
            et.resize_segments_([512, 2048])
        with pytest.raises(ValueError, match="exceeds"):
            et.resize_segments_([4097, 2048])
        with pytest.raises(ValueError, match="Expected 2 sizes"):
            et.resize_segments_([4096])
    finally:
        et.free()

    with pytest.raises(ValueError, match="sum to"):
        ExtensibleTensor(max_num_bytes=100, device="cuda", segment_capacities=[50])


@create_new_process_for_each_test("spawn")
def test_invalid_usage_raises() -> None:
    _skip_unless_vmm()
    with pytest.raises(ValueError):
        ExtensibleTensor(max_num_bytes=100, device="cuda", num_segments=3)

    et = ExtensibleTensor(max_num_bytes=8192, device="cuda", num_segments=2)
    try:
        with pytest.raises(IndexError):
            et.segment_view(2)
        et.resize_per_segment_(256)
        with pytest.raises(ValueError):
            et.resize_per_segment_(128)
        with pytest.raises(ValueError):
            et.resize_per_segment_(et.segment_capacity_bytes + 1)
    finally:
        et.free()


@create_new_process_for_each_test("spawn")
def test_release_physical_keeps_addresses() -> None:
    _skip_unless_vmm()
    """Releasing physical pages keeps the reservation; recommitting maps fresh
    zeroed pages under the same addresses (the sleep/wake contract)."""
    et = ExtensibleTensor(max_num_bytes=8192, device="cuda", num_segments=2)
    try:
        et.resize_per_segment_(1024, zero_new=True)
        fv = et.full_view()
        fv[:1024].fill_(9)
        base = et.base_ptr

        et.release_physical()
        assert et.num_bytes == 0
        assert et.physical_bytes == 0
        assert et.base_ptr == base

        et.resize_per_segment_(1024, zero_new=True)
        assert et.full_view().data_ptr() == base
        assert torch.count_nonzero(et.full_view()[:1024]) == 0
    finally:
        et.free()


class _FailingDriver:
    """Delegates to the real driver, recording access grants and failing the
    n-th physical allocation with an error `_map_chunk_at` does not retry."""

    class Failure(Exception):
        pass

    def __init__(self, driver, fail_on_create: int) -> None:
        self._driver = driver
        self._fail_on_create = fail_on_create
        self.creates = 0
        self.access_grants: list[tuple[int, int]] = []

    def __getattr__(self, name):
        return getattr(self._driver, name)

    def create(self, size: int, device_index: int) -> int:
        self.creates += 1
        if self.creates == self._fail_on_create:
            raise self.Failure("simulated allocation failure")
        return self._driver.create(size, device_index)

    def set_access(self, ptr: int, size: int, device_index: int) -> None:
        self.access_grants.append((ptr, size))
        self._driver.set_access(ptr, size, device_index)


@create_new_process_for_each_test("spawn")
def test_failed_commit_leaves_no_inaccessible_granules() -> None:
    _skip_unless_vmm()
    """A commit that maps several runs and fails part-way must have granted
    access to every run it did map; later commits skip mapped granules, so an
    unmapped-but-recorded or mapped-but-inaccessible granule would fault."""
    et = ExtensibleTensor(max_num_bytes=1, device="cuda")
    g = et.granularity
    et.free()
    et = ExtensibleTensor(max_num_bytes=4 * g, device="cuda")
    try:
        buffer = et._buffer
        buffer.ensure_committed_range(g, 2 * g)
        driver = _FailingDriver(buffer._driver, fail_on_create=2)
        buffer._driver = driver
        # Runs [0, g) and [2g, 3g) are needed; the second allocation fails.
        with pytest.raises(_FailingDriver.Failure):
            buffer.ensure_committed_range(0, 3 * g)
        assert buffer._mapped_granules == {0, 1}
        base = buffer.base_ptr
        assert any(
            ptr <= base and base + g <= ptr + size for ptr, size in driver.access_grants
        ), driver.access_grants
        # The mapped prefix is usable and a retry completes the range.
        et.full_view()[: 2 * g].fill_(1)
        torch.accelerator.synchronize()
        buffer.ensure_committed_range(0, 3 * g)
        assert buffer._mapped_granules == {0, 1, 2}
        et.full_view()[: 3 * g].fill_(2)
        torch.accelerator.synchronize()
    finally:
        et.free()


@create_new_process_for_each_test("spawn")
def test_views_survive_interpreter_teardown() -> None:
    _skip_unless_vmm()
    """Views handed to torch through DLPack may be destroyed after this
    module's globals are cleared at exit; that must not call freed memory."""
    script = """
import sys, torch
from vllm.utils.extensible_tensor import ExtensibleTensor
et = ExtensibleTensor(4096, device="cuda")
et.resize_per_segment_(1024)
# Held by modules imported before the library, so destroyed after it.
torch._extkv_test_view = et.full_view()
sys._extkv_test_view = et.segment_view(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]


@create_new_process_for_each_test("spawn")
def test_vmm_probe_reports_usable_driver() -> None:
    reason = vmm_unavailable_reason()
    print(f"vmm_unavailable_reason reported as: {reason}")
    if torch.version.hip is None:
        assert reason is None
    else:
        # ROCm runtimes older than the hipMemSetAccess fix are rejected.
        assert reason is None or "HIP runtime" in reason
