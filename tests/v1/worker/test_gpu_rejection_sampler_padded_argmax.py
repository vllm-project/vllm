# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A padded-lane argmax index must not escape the real block count.

Two reductions in the V2 rejection sampler run ``tl.argmax`` over
``PADDED_*_NUM_BLOCKS`` lanes -- the next power of two -- and then index a
companion tensor that is only ``*_num_blocks`` wide::

    _gather_global_argmax (target)  vocab 163840 / 8192 ->  20 real,  32 padded
    _gather_global_argmax (resample) vocab 163840 / 1024 -> 160 real, 256 padded

The companion max load is masked with ``other=-inf``, so a padded lane cannot
win on value alone. It can win when a real lane holds NaN. Triton's argmax
combine is ``gt = (value1 > value2) or (value1 == value2 and index1 < index2)``
and both comparisons are false for an unordered operand, so a NaN yields the
*other* operand and an arbitrary surviving index. The gather that follows is
neither masked nor clamped, so whatever index survives is applied directly to
the companion tensor -- for the last row of a batch, past its allocation.

Which lane survives depends on the order in which the reduction happens to
combine lanes, which is a property of the Triton build rather than of this
code. So the kernel-level tests here do not assert an index; they assert that
the block index cannot escape the real block count, by filling the companion
tensor with in-vocabulary ids so that only an escaped *index* can be observed.
The end-to-end test at the bottom of this file covers the complementary half,
that the *id* the sampler emits is inside the vocabulary, which the companion's
own producers can otherwise violate.

To observe an escaped index rather than read past the allocation, the
companion tensor is over-allocated to the padded width and its out-of-range
columns are filled with a sentinel. The strides handed to the kernels are the
over-allocated ones, so the indexing under test is the production indexing with
a wider row: an escaped index lands on the sentinel rather than on the next
row's valid id, which in production would be indistinguishable from a correct
result on any row but the last. The sentinel is the value a mixed batch
produced in the field -- two INT32_MAX halves, i.e. uninitialized memory read
as an int64.

The batch geometry mirrors the one that first exposed this: two requests, the
first carrying draft tokens and the second none. ``warmup.py`` builds that
shape deliberately, and the scheduler produces it in ordinary serving whenever
the drafter proposes nothing for a request.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _gather_global_argmax,
    _insert_resampled_kernel,
    rejection_sample,
)

# An exact multiple of both block sizes, so the real block counts are 20 and
# 160 and neither is a power of two -- which is what leaves padded lanes for
# the reduction to reach.
VOCAB_SIZE = 163840
# Mirrors the constants in `rejection_sample`.
VOCAB_BLOCK_SIZE = 8192
RESAMPLE_BLOCK_SIZE = 1024

NUM_SPECULATIVE_STEPS = 2

# What the escaped gather returned in the field: 0x7FFFFFFF7FFFFFFF, two
# INT32_MAX halves. Any id at or beyond the vocabulary would do; this one names
# the failure.
SENTINEL = 0x7FFFFFFF7FFFFFFF

# NaN placements among the real lanes. "all" is the one that escapes: with no
# finite real lane left, every combine order settles on the lowest-indexed
# masked lane, which is exactly `num_blocks`. The others leave a finite lane
# that usually survives, and are here because the invariant has to hold for
# them too however the reduction is ordered.
NAN_PATTERNS = ["all", "leading_half", "first", "last"]


def _local_max_row(num_blocks: int, winner: int, dtype, device) -> torch.Tensor:
    """Strictly decreasing values with a unique maximum at ``winner``.

    ``winner`` is deliberately not the last real block, so a test that clamps
    to ``num_blocks - 1`` cannot be mistaken for one that found the true max.
    """
    row = -torch.arange(num_blocks, dtype=dtype, device=device)
    row[winner] = 1000.0
    return row


def _poison(real_lanes: torch.Tensor, pattern: str) -> None:
    """Write ``pattern`` into a view of the real lanes, in place."""
    num_blocks = real_lanes.shape[-1]
    if pattern == "all":
        real_lanes.fill_(float("nan"))
    elif pattern == "leading_half":
        real_lanes[..., : num_blocks // 2] = float("nan")
    elif pattern == "first":
        real_lanes[..., 0] = float("nan")
    elif pattern == "last":
        real_lanes[..., num_blocks - 1] = float("nan")
    else:
        raise AssertionError(f"unknown NaN pattern {pattern!r}")


def _companion_tensors(
    num_rows: int,
    num_blocks: int,
    padded_num_blocks: int,
    block_size: int,
    max_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Over-allocated (local_argmax, local_max), sentinel past the real width.

    Production sizes both to ``num_blocks``; widening them to the padded width
    is what turns an out-of-bounds read into an observable value.
    """
    local_argmax = torch.full(
        (num_rows, padded_num_blocks), SENTINEL, dtype=torch.int64, device=device
    )
    # Real lanes hold the first token id of their block, as the producers store.
    local_argmax[:, :num_blocks] = (
        torch.arange(num_blocks, dtype=torch.int64, device=device) * block_size
    )
    # Padded lanes of the max are read under `mask`, so +inf here is unreachable
    # while the mask holds -- and wins outright the moment it stops holding.
    local_max = torch.full(
        (num_rows, padded_num_blocks),
        float("inf"),
        dtype=max_dtype,
        device=device,
    )
    return local_argmax, local_max


@triton.jit
def _target_argmax_probe_kernel(
    out_ptr,
    target_local_max_ptr,
    target_local_max_stride,
    target_local_argmax_ptr,
    target_local_argmax_stride,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    """Expose the device function under test one row per program.

    ``_gather_global_argmax`` is a ``triton.jit`` device function called from
    inside ``_rejection_kernel``, so reaching it needs a launchable wrapper.
    This one adds nothing but the store.
    """
    logit_idx = tl.program_id(0)
    token_id = _gather_global_argmax(
        target_local_max_ptr,
        target_local_max_stride,
        target_local_argmax_ptr,
        target_local_argmax_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    tl.store(out_ptr + logit_idx, token_id)


def _run_target_argmax(
    nan_pattern: str | None, winner: int, device: torch.device
) -> torch.Tensor:
    num_blocks = triton.cdiv(VOCAB_SIZE, VOCAB_BLOCK_SIZE)
    padded_num_blocks = triton.next_power_of_2(num_blocks)
    assert padded_num_blocks > num_blocks, "no padded lanes; test proves nothing"

    # One row per logit of the mixed batch below: 3 for the request with draft
    # tokens, 1 for the request without.
    num_logits = (NUM_SPECULATIVE_STEPS + 1) + 1
    local_argmax, local_max = _companion_tensors(
        num_logits,
        num_blocks,
        padded_num_blocks,
        VOCAB_BLOCK_SIZE,
        torch.float32,
        device,
    )
    local_max[:, :num_blocks] = _local_max_row(
        num_blocks, winner, torch.float32, device
    )
    if nan_pattern is not None:
        _poison(local_max[:, :num_blocks], nan_pattern)

    out = torch.zeros(num_logits, dtype=torch.int64, device=device)
    _target_argmax_probe_kernel[(num_logits,)](
        out,
        local_max,
        local_max.stride(0),
        local_argmax,
        local_argmax.stride(0),
        num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_num_blocks,
    )
    return out


def _run_insert_resampled(
    nan_pattern: str | None,
    winner: int,
    max_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return the id ``_insert_resampled_kernel`` wrote for each request."""
    num_blocks = triton.cdiv(VOCAB_SIZE, RESAMPLE_BLOCK_SIZE)
    padded_num_blocks = triton.next_power_of_2(num_blocks)
    assert padded_num_blocks > num_blocks, "no padded lanes; test proves nothing"

    # Request 0 carries draft tokens (1 + k logits), request 1 carries none (1
    # logit). This is the mixed shape; it is also why the second request is the
    # last row, the one whose escaped gather leaves the allocation.
    num_reqs = 2
    cu_num_logits = torch.tensor(
        [0, NUM_SPECULATIVE_STEPS + 1, NUM_SPECULATIVE_STEPS + 2],
        dtype=torch.int32,
        device=device,
    )
    num_logits = int(cu_num_logits[-1])
    # Request 0 accepted one token and resamples at its second logit; request 1
    # has no drafts, so its only logit is the bonus. Both reach the gather.
    num_sampled = torch.tensor([1, 0], dtype=torch.int32, device=device)
    write_col = num_sampled.clone()

    expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int32, device=device)
    expanded_idx_mapping[NUM_SPECULATIVE_STEPS + 1 :] = 1
    # Non-zero temperature everywhere, so neither request takes the greedy
    # early return above the gather.
    temperature = torch.ones(num_reqs, dtype=torch.float32, device=device)

    local_argmax, local_max = _companion_tensors(
        num_reqs,
        num_blocks,
        padded_num_blocks,
        RESAMPLE_BLOCK_SIZE,
        max_dtype,
        device,
    )
    local_max[:, :num_blocks] = _local_max_row(num_blocks, winner, max_dtype, device)
    if nan_pattern is not None:
        _poison(local_max[:, :num_blocks], nan_pattern)

    sampled = torch.zeros(
        num_reqs, NUM_SPECULATIVE_STEPS + 1, dtype=torch.int64, device=device
    )
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_num_blocks,
    )
    return sampled.gather(1, write_col.to(torch.int64).unsqueeze(1)).squeeze(1)


def _assert_in_vocab(ids: torch.Tensor) -> None:
    """Every real lane of the companion holds an in-vocabulary id, so this can
    only fire on an id gathered from an escaped block index."""
    out_of_range = (ids < 0) | (ids >= VOCAB_SIZE)
    assert not out_of_range.any(), (
        f"sampler emitted ids outside [0, {VOCAB_SIZE}): {ids.tolist()}"
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("nan_pattern", NAN_PATTERNS)
def test_target_argmax_stays_in_vocab_with_nan_logits(nan_pattern: str):
    """A NaN block max must not let the greedy argmax gather past the row."""
    ids = _run_target_argmax(nan_pattern, winner=7, device=torch.device("cuda:0"))
    _assert_in_vocab(ids)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("nan_pattern", NAN_PATTERNS)
@pytest.mark.parametrize("max_dtype", [torch.float32, torch.float64])
def test_insert_resampled_stays_in_vocab_with_nan_logits(
    nan_pattern: str, max_dtype: torch.dtype
):
    """The same for the resample gather, over both `use_fp64` layouts.

    This is the site with the widest overrun: 160 real lanes against 256
    padded, so an escaped index reaches up to 95 int64 words past the last
    request's row.
    """
    ids = _run_insert_resampled(
        nan_pattern, winner=37, max_dtype=max_dtype, device=torch.device("cuda:0")
    )
    _assert_in_vocab(ids)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_target_argmax_unchanged_without_nan():
    """The control. Finite logits must still resolve to the true maximum.

    Without this the tests above could be satisfied by a fix that clamps every
    lookup to the last block.
    """
    winner = 7
    ids = _run_target_argmax(None, winner=winner, device=torch.device("cuda:0"))
    _assert_in_vocab(ids)
    assert torch.all(ids == winner * VOCAB_BLOCK_SIZE), ids.tolist()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("max_dtype", [torch.float32, torch.float64])
def test_insert_resampled_unchanged_without_nan(max_dtype: torch.dtype):
    """The same control for the resample gather."""
    winner = 37
    ids = _run_insert_resampled(
        None, winner=winner, max_dtype=max_dtype, device=torch.device("cuda:0")
    )
    _assert_in_vocab(ids)
    assert torch.all(ids == winner * RESAMPLE_BLOCK_SIZE), ids.tolist()


# The two reductions above consume a companion tensor that `rejection_sample`
# fills in with its own masked reductions, which build a token id by arithmetic
# rather than by dereferencing an index. Those can put an out-of-vocabulary lane
# of a partial block into the companion, so the end-to-end invariant needs its
# own coverage. A vocabulary smaller than a block size makes block 0 itself
# partial, which is the case an all-NaN row settles on.
SMALL_VOCAB_SIZES = [900, 5000]


def _run_rejection_sample(vocab_size: int, temperature: float) -> torch.Tensor:
    """Drive `rejection_sample` on an all-NaN target and return the emitted ids."""
    device = torch.device("cuda:0")
    num_reqs = 2
    logits_per_req = NUM_SPECULATIVE_STEPS + 1
    num_logits = num_reqs * logits_per_req

    target_logits = torch.full(
        (num_logits, vocab_size), float("nan"), dtype=torch.float32, device=device
    )
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    sampled, num_sampled = rejection_sample(
        target_logits=target_logits,
        draft_logits=None,
        draft_sampled=torch.zeros(num_logits, dtype=torch.int64, device=device),
        cu_num_logits=torch.arange(
            0, num_logits + 1, logits_per_req, dtype=torch.int32, device=device
        ),
        pos=torch.arange(num_logits, dtype=torch.int64, device=device),
        idx_mapping=idx_mapping,
        expanded_idx_mapping=idx_mapping.repeat_interleave(logits_per_req),
        expanded_local_pos=torch.arange(
            logits_per_req, dtype=torch.int32, device=device
        )
        .repeat(num_reqs)
        .contiguous(),
        temperature=torch.full(
            (num_reqs,), temperature, dtype=torch.float32, device=device
        ),
        seed=torch.full((num_reqs,), 42, dtype=torch.int64, device=device),
        num_speculative_steps=NUM_SPECULATIVE_STEPS,
    )
    steps = torch.arange(NUM_SPECULATIVE_STEPS + 1, device=device)
    return sampled[steps.unsqueeze(0) < num_sampled.unsqueeze(1)]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("vocab_size", SMALL_VOCAB_SIZES)
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_rejection_sample_stays_in_vocab_with_nan_logits(
    vocab_size: int, temperature: float
):
    """No NaN target row may make the sampler emit an id outside the vocabulary.

    Covers greedy (the target-argmax companion) and sampling (the resample
    companion), whose block sizes differ, so one vocabulary leaves block 0
    partial for one path and not the other.
    """
    ids = _run_rejection_sample(vocab_size, temperature)
    assert ids.numel() > 0
    assert torch.all((ids >= 0) & (ids < vocab_size)), (
        f"emitted out-of-vocabulary id: {ids.tolist()} (vocab_size={vocab_size})"
    )
