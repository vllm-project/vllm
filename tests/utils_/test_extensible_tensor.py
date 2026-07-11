# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils.extensible_tensor import ExtensibleTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_extensible_tensor_grows_without_moving() -> None:
    buffer = ExtensibleTensor(4096, device="cuda")
    try:
        base_ptr = buffer.base_ptr
        first_view = buffer.resize_(1024)
        assert first_view.data_ptr() == base_ptr
        first_view.fill_(7)

        second_view = buffer.resize_(2048)
        assert second_view.data_ptr() == base_ptr
        assert torch.equal(second_view[:1024], torch.full_like(second_view[:1024], 7))

        second_view[1024:].fill_(3)
        assert torch.equal(buffer.tensor, second_view)

        full_view = buffer.full_view()
        assert full_view.data_ptr() == base_ptr
        assert full_view.numel() == 4096
    finally:
        buffer.free()


def test_extensible_tensor_rejects_shrink_and_overflow() -> None:
    buffer = ExtensibleTensor(1024, device="cuda")
    try:
        buffer.resize_(512)
        with pytest.raises(ValueError, match="grow-only"):
            buffer.resize_(256)
        with pytest.raises(ValueError, match="exceeds the segment capacity"):
            buffer.resize_(1025)
    finally:
        buffer.free()


def test_segments_grow_in_lockstep_and_zero_new() -> None:
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
        # Committed prefixes start zeroed.
        assert torch.count_nonzero(fv[:256]) == 0
        assert torch.count_nonzero(fv[4096 : 4096 + 256]) == 0

        pattern_a = torch.arange(256, device="cuda", dtype=torch.uint8)
        pattern_b = 255 - pattern_a
        fv[:256].copy_(pattern_a)
        fv[4096 : 4096 + 256].copy_(pattern_b)

        et.resize_per_segment_(1024, zero_new=True)
        fv2 = et.full_view()
        assert fv2.data_ptr() == fv.data_ptr()
        # Old bytes of both segments preserved; freshly committed ranges zeroed.
        assert torch.equal(fv2[:256], pattern_a)
        assert torch.equal(fv2[4096 : 4096 + 256], pattern_b)
        assert torch.count_nonzero(fv2[256:1024]) == 0
        assert torch.count_nonzero(fv2[4096 + 256 : 4096 + 1024]) == 0
    finally:
        et.free()


def test_segments_at_granularity_scale() -> None:
    """Segments spanning multiple mapping granules commit correctly.

    Uses a segment capacity that is not a multiple of the allocation
    granularity, so a granule straddles the segment boundary and is shared by
    the first commit of one segment and a later commit of the other -- it must
    be mapped exactly once.
    """
    probe = ExtensibleTensor(max_num_bytes=1, device="cuda")
    granularity = probe.capacity_bytes
    probe.free()
    # Two segments of 1.5 granules each; the middle granule straddles the
    # boundary.
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

        # Grow to the full segment capacity: previously mapped granules
        # (including the boundary-straddling one) are reused, new ones are
        # committed and zeroed.
        et.resize_per_segment_(seg, zero_new=True)
        fv2 = et.full_view()
        assert torch.all(fv2[:step] == 1)
        assert torch.all(fv2[seg : seg + step] == 2)
        assert torch.count_nonzero(fv2[step:seg]) == 0
        assert torch.count_nonzero(fv2[seg + step :]) == 0
    finally:
        et.free()


def test_multi_segment_invalid_usage_raises() -> None:
    """Prefix-view APIs and invalid segment configs raise for multi-segment
    buffers."""
    with pytest.raises(ValueError):
        ExtensibleTensor(max_num_bytes=100, device="cuda", num_segments=3)

    et = ExtensibleTensor(max_num_bytes=8192, device="cuda", num_segments=2)
    try:
        with pytest.raises(ValueError):
            _ = et.tensor
        with pytest.raises(ValueError):
            et.resize_(256)

        et.resize_per_segment_(256)
        with pytest.raises(ValueError):
            et.resize_per_segment_(128)  # shrink
        with pytest.raises(ValueError):
            et.resize_per_segment_(et.segment_capacity_bytes + 1)  # over capacity
    finally:
        et.free()
