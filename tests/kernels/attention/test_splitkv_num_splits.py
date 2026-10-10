# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the decode split-KV split-count heuristic.

`_get_num_splits` takes `num_sms` as an argument, so these run on CPU without
a GPU or ROCm. This file deliberately avoids importing `vllm.platforms.rocm`,
which resolves the GCN arch at module load and needs an AMD device.

Expected values below are measured on gfx1201 and gfx1100.
"""

import pytest
import torch

from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    _MAX_SPLITS,
    _choose_compute_block_size,
    _get_num_splits,
    paged_attention_2d_splitkv_decode,
)

HEAD_SIZE = 256
NUM_KV_HEADS = 4

# torch reports WGPs rather than CUs: gfx1201 reports 32 against 64 CUs and
# gfx1100 reports 42 against 84 CUs.
GFX1201_SMS = 32
GFX1100_SMS = 42


def _splits(
    max_seq_len: int,
    num_sms: int,
    physical_block_size: int,
    batch_size: int = 1,
    max_num_splits: int = _MAX_SPLITS,
) -> int:
    return _get_num_splits(
        batch_size=batch_size,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        block_size=physical_block_size,
        max_seq_len=max_seq_len,
        max_num_splits=max_num_splits,
        num_sms=num_sms,
    )


@pytest.mark.parametrize(
    "physical_block_size,expected",
    [(1056, 32), (528, 16), (784, 16), (32, 32), (16, 16)],
)
def test_choose_compute_block_size(physical_block_size: int, expected: int):
    """The split tables below depend on this physical -> compute mapping."""
    assert _choose_compute_block_size(physical_block_size) == expected


# (num_sms, physical_block_size, max_seq_len, expected_splits)
MEASURED_SPLITS = [
    # gfx1201, physical block 1056 -> compute block 32
    (GFX1201_SMS, 1056, 512, 1),
    (GFX1201_SMS, 1056, 1024, 1),
    (GFX1201_SMS, 1056, 2048, 16),
    (GFX1201_SMS, 1056, 4096, 15),
    (GFX1201_SMS, 1056, 8192, 14),
    (GFX1201_SMS, 1056, 16384, 14),
    (GFX1201_SMS, 1056, 32768, 14),
    # gfx1201, physical block 528 -> compute block 16
    (GFX1201_SMS, 528, 512, 1),
    (GFX1201_SMS, 528, 1024, 16),
    (GFX1201_SMS, 528, 2048, 15),
    (GFX1201_SMS, 528, 4096, 14),
    (GFX1201_SMS, 528, 8192, 14),
    (GFX1201_SMS, 528, 16384, 14),
    (GFX1201_SMS, 528, 32768, 14),
    # gfx1100, physical block 784 -> compute block 16
    (GFX1100_SMS, 784, 512, 1),
    (GFX1100_SMS, 784, 1024, 1),
    (GFX1100_SMS, 784, 2048, 15),
    (GFX1100_SMS, 784, 4096, 14),
    (GFX1100_SMS, 784, 8192, 14),
    (GFX1100_SMS, 784, 16384, 14),
    (GFX1100_SMS, 784, 32768, 14),
]


@pytest.mark.parametrize(
    "num_sms,physical_block_size,max_seq_len,expected", MEASURED_SPLITS
)
def test_num_splits_matches_measured(
    num_sms: int, physical_block_size: int, max_seq_len: int, expected: int
):
    assert _splits(max_seq_len, num_sms, physical_block_size) == expected


BATCH_COLLAPSE_SEQ_LEN = 30000

# (num_sms, physical_block_size, batch_size, expected_splits)
MEASURED_BATCH_SPLITS = [
    (GFX1201_SMS, 1056, 1, 14),
    (GFX1201_SMS, 1056, 2, 7),
    (GFX1201_SMS, 1056, 4, 4),
    (GFX1201_SMS, 1056, 8, 2),
    (GFX1201_SMS, 1056, 16, 1),
    (GFX1100_SMS, 784, 1, 14),
    (GFX1100_SMS, 784, 2, 9),
    (GFX1100_SMS, 784, 4, 5),
    (GFX1100_SMS, 784, 8, 5),
    (GFX1100_SMS, 784, 16, 5),
]


@pytest.mark.parametrize(
    "num_sms,physical_block_size,batch_size,expected", MEASURED_BATCH_SPLITS
)
def test_num_splits_collapses_with_batch(
    num_sms: int, physical_block_size: int, batch_size: int, expected: int
):
    """Splitting tapers as the batch fills the machine on its own."""
    splits = _splits(
        BATCH_COLLAPSE_SEQ_LEN,
        num_sms,
        physical_block_size,
        batch_size=batch_size,
    )
    assert splits == expected


@pytest.mark.parametrize("num_sms", [16, GFX1201_SMS, GFX1100_SMS, 64])
def test_low_context_early_return_boundary(num_sms: int):
    """No splitting below 2 * num_sms compute blocks of context."""
    physical_block_size = 784
    compute_block_size = _choose_compute_block_size(physical_block_size)
    boundary = 2 * num_sms * compute_block_size

    below = _splits(boundary - compute_block_size, num_sms, physical_block_size)
    at_boundary = _splits(boundary, num_sms, physical_block_size)

    assert below == 1
    assert at_boundary > 1


@pytest.mark.parametrize(
    "num_sms,physical_block_size",
    [(GFX1201_SMS, 1056), (GFX1100_SMS, 784)],
)
def test_batch_guard_threshold_scales_with_num_sms(
    num_sms: int, physical_block_size: int
):
    """Splitting is off once batch_nheads reaches 0.8 * (2 * num_sms).

    The threshold is 51.2 on a 32-WGP part and 67.2 on a 42-WGP one, so a
    batch of 16 (64 KV-head groups) collapses on the former but not the
    latter. Assert the relationship rather than one card's numbers.
    """
    threshold = 0.8 * 2 * num_sms

    for batch_size in (1, 2, 4, 8, 16, 32):
        splits = _splits(
            BATCH_COLLAPSE_SEQ_LEN,
            num_sms,
            physical_block_size,
            batch_size=batch_size,
        )
        if batch_size * NUM_KV_HEADS >= threshold:
            assert splits == 1, f"expected collapse at batch {batch_size}"
        else:
            assert splits > 1, f"expected splitting at batch {batch_size}"


@pytest.mark.parametrize("num_sms", [1, 2, 8, 16, 32, 42, 64, 128, 304])
@pytest.mark.parametrize("max_num_splits", [1, 8, _MAX_SPLITS])
def test_num_splits_invariants(num_sms: int, max_num_splits: int):
    """Never 0, never above the requested bound.

    The launcher rejects a count above max_num_splits, so staying inside the
    bound is what keeps that guard from firing.
    """
    for physical_block_size in (16, 32, 528, 784, 1056):
        for batch_size in (1, 4, 16):
            for max_seq_len in (1, 512, 4096, 32768, 131072):
                splits = _splits(
                    max_seq_len,
                    num_sms,
                    physical_block_size,
                    batch_size=batch_size,
                    max_num_splits=max_num_splits,
                )
                assert 1 <= splits <= max_num_splits, (
                    f"{num_sms=} {physical_block_size=} "
                    f"{batch_size=} {max_seq_len=} -> {splits}"
                )


def test_small_head_size_short_context_never_splits():
    """head_size <= 64 with short context never splits."""
    assert (
        _get_num_splits(
            batch_size=1,
            num_kv_heads=NUM_KV_HEADS,
            head_size=64,
            block_size=32,
            max_seq_len=2048,
            max_num_splits=_MAX_SPLITS,
            num_sms=GFX1201_SMS,
        )
        == 1
    )


def test_launcher_rejects_splits_above_max(monkeypatch):
    """The launcher raises before launching any kernel.

    The stub triton used when no GPU driver is active has no next_power_of_2,
    so supply it; the guard under test runs before any kernel launch.
    """
    from vllm.v1.attention.ops import chunked_prefill_paged_decode as ops

    if not hasattr(ops.triton, "next_power_of_2"):
        monkeypatch.setattr(
            ops.triton,
            "next_power_of_2",
            lambda n: 1 << (n - 1).bit_length(),
            raising=False,
        )

    block_size, x = 32, 8
    query = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE)
    key_cache = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE // x, block_size, x)
    value_cache = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE, block_size)
    block_tables = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)

    with pytest.raises(ValueError, match="must be <="):
        paged_attention_2d_splitkv_decode(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            scale=1.0,
            actual_max_splits=_MAX_SPLITS + 1,
            max_num_splits=_MAX_SPLITS,
            max_seq_len=1024,
        )
