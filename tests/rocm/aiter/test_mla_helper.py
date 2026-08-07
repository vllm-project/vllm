# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head padding/unpadding and validity tests for ``AiterMLAHelper``.

AITER MLA asm persistent decode requires ``num_heads >= 16``. For models with
fewer heads, ``AiterMLAHelper`` pads Q up to exactly 16 heads -- divisors of 16
via ``repeat_interleave`` and non-divisors (e.g. 12 heads/rank at TP8, 6 at
TP16) via tile-and-slice -- runs the kernel, then slices the output back down.

These are pure-Python tests (no GPU kernels) covering both padding strategies,
output contiguity, and the pad -> unpad round-trip, which guards against silent
wrong-output for small-head models such as Kimi-K2.5 at TP8.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AITER MLA is ROCm-only"
)

MIN_HEADS = 16
Q_HEAD_DIM = 576  # kv_lora_rank + qk_rope_head_dim

# Head counts < 16 that evenly divide 16 -> repeat_interleave padding path.
DIVISOR_HEADS = [1, 2, 4, 8]
# Head counts < 16 that do NOT divide 16 -> tile-and-slice padding path.
# This is the branch models like Kimi-K2.5 (12 heads/rank at TP8) actually hit.
NON_DIVISOR_HEADS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
SMALL_HEADS = sorted(DIVISOR_HEADS + NON_DIVISOR_HEADS)


# --- validity -------------------------------------------------------------


@pytest.mark.parametrize(
    ("nhead", "valid"),
    [
        # < 16 (padded) and multiples of 16 are valid.
        (1, True),
        (8, True),
        (15, True),
        (16, True),
        (32, True),
        (128, True),
        # non-positive, or >= 16 but not a multiple of 16, are invalid.
        (0, False),
        (-1, False),
        (17, False),
        (24, False),
        (33, False),
    ],
)
def test_is_valid_num_heads(nhead, valid):
    assert AiterMLAHelper.is_valid_num_heads(nhead) is valid


@pytest.mark.parametrize("nhead", [0, 17, 24])
def test_check_num_heads_validity_raises(nhead):
    with pytest.raises(AssertionError, match="ROCM AITER MLA requires"):
        AiterMLAHelper.check_num_heads_validity(nhead)


@pytest.mark.parametrize("nhead", [8, 16])
def test_check_num_heads_validity_passes(nhead):
    AiterMLAHelper.check_num_heads_validity(nhead)


@pytest.mark.parametrize(
    ("nhead", "expected"),
    [(1, 16), (8, 16), (15, 16), (16, 16), (128, 128)],
)
def test_get_actual_mla_num_heads(nhead, expected):
    assert AiterMLAHelper.get_actual_mla_num_heads(nhead) == expected


# --- get_mla_padded_q -----------------------------------------------------


@pytest.mark.parametrize("nhead", [16, 128])
def test_padded_q_large_is_noop(nhead):
    q = torch.randn(4, nhead, Q_HEAD_DIM)
    assert AiterMLAHelper.get_mla_padded_q(nhead, q) is q


@pytest.mark.parametrize("nhead", DIVISOR_HEADS)
def test_padded_q_divisor_uses_repeat_interleave(nhead):
    q = torch.randn(2, nhead, Q_HEAD_DIM)
    padded = AiterMLAHelper.get_mla_padded_q(nhead, q)
    assert padded.shape == (2, MIN_HEADS, Q_HEAD_DIM)
    assert padded.is_contiguous()
    reps = MIN_HEADS // nhead
    # repeat_interleave repeats each head `reps` times consecutively.
    for j in range(MIN_HEADS):
        torch.testing.assert_close(padded[:, j, :], q[:, j // reps, :])


@pytest.mark.parametrize("nhead", NON_DIVISOR_HEADS)
def test_padded_q_non_divisor_tiles(nhead):
    q = torch.randn(2, nhead, Q_HEAD_DIM)
    padded = AiterMLAHelper.get_mla_padded_q(nhead, q)
    assert padded.shape == (2, MIN_HEADS, Q_HEAD_DIM)
    # The asm persistent decode reads Q as a packed [tokens, 16, head_dim]
    # buffer, so the tile-and-slice path must materialize a contiguous copy.
    assert padded.is_contiguous()
    # tile-and-slice repeats the whole head block, so head j maps to j % nhead.
    for j in range(MIN_HEADS):
        torch.testing.assert_close(padded[:, j, :], q[:, j % nhead, :])


# --- pad -> unpad round-trip (the silent-wrong-output guard) ---------------


@pytest.mark.parametrize("nhead", SMALL_HEADS + [16, 128])
def test_pad_unpad_roundtrip_recovers_original(nhead):
    """``unpad(pad(q))`` must return the original per-head values.

    Padding then unpadding shares the exact index math the kernel relies on, so
    this is the core guard that real heads are never dropped, duplicated, or
    reordered -- the failure mode that would silently corrupt output for
    small-head models. This also exercises both unpad paths (strided slice for
    divisors, leading-slice for non-divisors) and the >= 16 no-op.
    """
    q = torch.randn(3, nhead, Q_HEAD_DIM)
    padded = AiterMLAHelper.get_mla_padded_q(nhead, q)
    recovered = AiterMLAHelper.get_mla_unpadded_o(nhead, padded)
    assert recovered.shape == q.shape
    torch.testing.assert_close(recovered, q)
