# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for the draft-path host-dispatch reductions.

Two substitutions on the DFlash/DSpark draft path are pinned here. Both are
pure host arithmetic, so these tests need no GPU.

1. `seq_lens_cpu_upper_bound[:num_reqs].max().item()` became
   `int(seq_lens_cpu_upper_bound.numpy()[:num_reqs].max())`. Same reduction,
   read through a zero-copy view instead of a slice/max/item chain of torch
   dispatches -- which at concurrency 1 reduces a single element.

2. `_build_uniform_attn_metadata` now passes `max_query_len` instead of letting
   `_build_attn_metadata` recover it from the tensor. That is an algebraic
   claim, not a refactor: the path builds `query_start_loc` as a ramp of
   constant stride `num_query_per_req` and then fills the padded tail with the
   last value, so consecutive differences are `num_query_per_req` for real
   requests and 0 for padding. The claim is only safe if that holds for every
   batch shape, including the empty one, so it is checked exhaustively rather
   than argued.
"""

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------
# 1. the numpy-view reduction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [1, 8, 129])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_numpy_max_matches_torch_max(capacity, dtype):
    rng = np.random.default_rng(0)
    for _ in range(50):
        buf = torch.from_numpy(
            rng.integers(0, 1_000_000, size=capacity).astype(
                np.int32 if dtype is torch.int32 else np.int64
            )
        )
        for num_reqs in range(1, capacity + 1):
            want = buf[:num_reqs].max().item()
            got = int(buf.numpy()[:num_reqs].max())
            assert got == want, f"num_reqs={num_reqs}"


def test_numpy_view_is_not_a_copy():
    """If `.numpy()` copied, the substitution would cost more than it saves."""
    buf = torch.arange(16, dtype=torch.int32)
    view = buf.numpy()
    buf[3] = 999
    assert view[3] == 999


# --------------------------------------------------------------------------
# 2. the max_query_len identity
# --------------------------------------------------------------------------


def _build_query_start_loc(num_reqs, num_query_per_req, num_reqs_padded):
    """Exactly what `_build_uniform_attn_metadata` + `_build_attn_metadata` do."""
    arange_np = np.arange(num_reqs_padded + 1, dtype=np.int32)
    query_start_loc_np = arange_np[: num_reqs + 1] * num_query_per_req
    out = torch.empty(num_reqs_padded + 1, dtype=torch.int32)
    out[: num_reqs + 1] = torch.from_numpy(query_start_loc_np[: num_reqs + 1])
    out[num_reqs:] = out[num_reqs]
    return out


@pytest.mark.parametrize("num_query_per_req", [1, 2, 3, 8])
@pytest.mark.parametrize("num_reqs_padded", [1, 2, 8, 33])
def test_max_query_len_identity(num_query_per_req, num_reqs_padded):
    """The passed value must equal what the tensor reduction would have found."""
    for num_reqs in range(0, num_reqs_padded + 1):
        qsl = _build_query_start_loc(num_reqs, num_query_per_req, num_reqs_padded)
        recovered = int((qsl[1:] - qsl[:-1]).max())
        passed = num_query_per_req if num_reqs >= 1 else 0
        assert passed == recovered, (
            f"num_reqs={num_reqs} num_query_per_req={num_query_per_req} "
            f"num_reqs_padded={num_reqs_padded}: passed {passed}, "
            f"tensor reduction gives {recovered}"
        )


def test_max_query_len_identity_empty_batch():
    """num_reqs == 0 is the case the `>= 1` guard exists for.

    With no requests every difference is 0, so recovering the max from the
    tensor yields 0 and the guard must agree; returning num_query_per_req here
    would over-report the query width for an empty batch.
    """
    qsl = _build_query_start_loc(0, 4, 8)
    assert int((qsl[1:] - qsl[:-1]).max()) == 0
    assert torch.all(qsl == 0)


def test_padded_tail_is_constant():
    """The identity relies on the tail being flat; pin that separately."""
    qsl = _build_query_start_loc(num_reqs=3, num_query_per_req=5, num_reqs_padded=9)
    assert qsl[3] == 15
    assert torch.all(qsl[3:] == 15)
    diffs = qsl[1:] - qsl[:-1]
    assert torch.all(diffs[:3] == 5)
    assert torch.all(diffs[3:] == 0)
