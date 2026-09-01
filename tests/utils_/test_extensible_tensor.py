# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


@create_new_process_for_each_test("spawn")
def test_vmm_probe_reports_usable_driver() -> None:
    reason = vmm_unavailable_reason()
    print(f"vmm_unavailable_reason reported as: {reason}")
    if torch.version.hip is None:
        assert reason is None
    else:
        # ROCm runtimes older than the hipMemSetAccess fix are rejected.
        assert reason is None or "HIP runtime" in reason
