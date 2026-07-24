# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout regression for vllm-project/vllm#42718.

The combined fused MoE LoRA expand kernel computes the per-slice base as
``cur_a_ptr = a_ptr + slice_id * (numel // num_slices)``. After
``tensor_model_parallel_all_gather`` interleaves the rank dim into the
gathered tensor, the slice (dim 0) stride no longer equals
``numel // num_slices`` for small per-shard ranks (e.g. local rank == 1),
so the kernel reads the wrong slice.

These tests pin down the stride contract from the issue's minimal
reproducer (no GPU / no distributed required), and verify that the
per-slice view obtained by indexing + ``.contiguous()`` —
the construction the fix uses to dispatch each slice as an
independent ``num_slices=1`` expand — is addressable in the way the
kernel expects.
"""

import torch


def _simulate_fully_sharded_all_gather(
    world: int,
    num_slices: int,
    tokens: int,
    topk: int,
    local_rank: int,
) -> torch.Tensor:
    """Reproduces the layout that `tensor_model_parallel_all_gather`
    produces for the W13 intermediate cache before the expand call.
    Same construction as the issue body."""
    x = torch.empty(num_slices, tokens, topk, local_rank)
    out = torch.empty((world,) + tuple(x.shape))
    z = out.reshape((world,) + tuple(x.shape)).movedim(0, x.dim() - 1)
    z = z.reshape(tuple(x.shape)[:-1] + (x.shape[-1] * world,))
    return z


def test_local_rank_1_breaks_kernel_slice_offset_assumption():
    """The exact case the issue reports: 16-way TP, 2 slices, local rank 1.
    The kernel assumes slice 1 lives at byte offset
    `slice_id * (numel // num_slices)`; the actual layout puts it at
    `stride(0)`. They disagree."""
    z = _simulate_fully_sharded_all_gather(
        world=16, num_slices=2, tokens=3, topk=1, local_rank=1
    )

    actual_slice1_offset = z.stride(0)
    kernel_assumed_slice1_offset = z.numel() // 2

    assert actual_slice1_offset == 3
    assert kernel_assumed_slice1_offset == 48
    assert actual_slice1_offset != kernel_assumed_slice1_offset


def test_local_rank_4_happens_to_match_kernel_assumption():
    """At local rank >= 4 the final reshape forces a contiguous copy and
    the per-slice stride happens to equal `numel // num_slices`. This is
    why the bug only surfaces at local rank 1 — the corrupted-read path
    isn't reachable on the larger-rank configurations the regression
    tests typically cover."""
    z = _simulate_fully_sharded_all_gather(
        world=16, num_slices=2, tokens=3, topk=1, local_rank=4
    )

    actual_slice1_offset = z.stride(0)
    kernel_assumed_slice1_offset = z.numel() // 2

    assert actual_slice1_offset == kernel_assumed_slice1_offset == 192


def test_contiguous_copy_realigns_slice_dim_with_kernel_assumption():
    """A blanket ``.contiguous()`` on the gathered cache fixes the layout
    mismatch. The fix in #42718 chooses a per-slice path (smaller copies)
    instead, but this test documents that the underlying stride contract
    can be made to hold."""
    z = _simulate_fully_sharded_all_gather(
        world=16, num_slices=2, tokens=3, topk=1, local_rank=1
    )
    z_c = z.contiguous()

    assert z_c.stride(0) == z_c.numel() // 2


def test_per_slice_indexed_view_is_addressable():
    """The fix's per-slice path constructs each slice via
    ``z[slice_id : slice_id + 1].contiguous()``. Verify that view:
    1. has length 1 along the slice dim (so a downstream ``num_slices=1``
       expand call is correct),
    2. is contiguous (so the kernel's ``.view(-1, last_dim)`` doesn't
       fail), and
    3. round-trips the underlying values, i.e. we are addressing each
       slice's data exactly once."""
    num_slices = 2
    z = _simulate_fully_sharded_all_gather(
        world=16, num_slices=num_slices, tokens=3, topk=1, local_rank=1
    )
    # Fill with a per-slice marker so we can check we are addressing
    # the right region.
    z = z.contiguous()
    for slice_id in range(num_slices):
        z[slice_id].fill_(float(slice_id + 1))

    for slice_id in range(num_slices):
        single = z[slice_id : slice_id + 1].contiguous()
        assert single.shape[0] == 1
        assert single.is_contiguous()
        # Each per-slice view must contain exactly the marker for its slice.
        assert torch.all(single == float(slice_id + 1))


def test_per_slice_offset_in_output_advances_by_w1_output_dim_size():
    """The fix lays out the per-slice expand outputs at
    ``offset + slice_id * w1_output_dim_size`` in the combined output
    tensor. Check the simple integer-math invariant the loop relies on:
    consecutive slice writes don't overlap and they cover the same span
    the original combined call would have written."""
    base_offset = 7
    w1_output_dim_size = 64
    num_slices = 2

    per_slice_offsets = [
        base_offset + slice_id * w1_output_dim_size for slice_id in range(num_slices)
    ]
    spans = [
        (per_slice_offsets[i], per_slice_offsets[i] + w1_output_dim_size)
        for i in range(num_slices)
    ]

    # No overlap between consecutive slices.
    for i in range(num_slices - 1):
        assert spans[i][1] == spans[i + 1][0]

    # Total span equals what the combined call would have written.
    assert spans[-1][1] - spans[0][0] == num_slices * w1_output_dim_size


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
