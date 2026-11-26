# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy W2: the slot_mapping generation check.

The kernel tests run the real `_compute_slot_mappings_kernel` on CUDA, or on
CPU tensors under the Triton interpreter (`TRITON_INTERPRET=1`). The
`PercyGenCheck` tests need neither.
"""

import os

import numpy as np
import pytest
import torch

from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.grant_table import NULL_GEN
from vllm.v1.worker.gpu.block_table import PercyGenCheck

pytestmark = pytest.mark.cpu_test

BS = 4
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
needs_kernel = pytest.mark.skipif(
    not HAS_TRITON or (_DEVICE.type == "cpu" and not _INTERPRET),
    reason="needs CUDA or the Triton interpreter (TRITON_INTERPRET=1)",
)


def _run_kernel(
    block_table: list[list[int]],
    req_tokens: list[tuple[int, list[int]]],
    gen_check: PercyGenCheck | None,
    block_sizes: tuple[int, ...] = (BS,),
    max_num_tokens: int = 32,
) -> list[list[int]]:
    """Slot mappings for `req_tokens` = [(req_state_idx, positions)], one
    block table shared by every group (rows indexed by req_state_idx)."""
    from vllm.v1.worker.gpu.block_table import _compute_slot_mappings_kernel

    num_groups = len(block_sizes)
    bt = torch.tensor(block_table, dtype=torch.int32, device=_DEVICE)
    tables = [bt.clone() for _ in range(num_groups)]
    ptrs = torch.tensor(
        [t.data_ptr() for t in tables], dtype=torch.uint64, device=_DEVICE
    )
    strides = torch.tensor(
        [t.stride(0) for t in tables], dtype=torch.int64, device=_DEVICE
    )
    sizes = torch.tensor(block_sizes, dtype=torch.int32, device=_DEVICE)
    enabled = torch.ones(num_groups, dtype=torch.bool, device=_DEVICE)
    dcp = torch.ones(num_groups, dtype=torch.bool, device=_DEVICE)
    idx = torch.tensor([r for r, _ in req_tokens], dtype=torch.int32, device=_DEVICE)
    qsl = [0]
    positions: list[int] = []
    for _, pos in req_tokens:
        positions += pos
        qsl.append(len(positions))
    qsl_t = torch.tensor(qsl, dtype=torch.int32, device=_DEVICE)
    pos_t = torch.tensor(positions or [0], dtype=torch.int64, device=_DEVICE)
    out = torch.full(
        (num_groups, max_num_tokens), 123, dtype=torch.int64, device=_DEVICE
    )
    if gen_check is None:
        unused = torch.zeros(1, dtype=torch.int32, device=_DEVICE)
        gen_args = (unused, unused, 0, unused)
    else:
        gen_args = (
            gen_check.table,
            gen_check.meta,
            gen_check.meta.stride(0),
            gen_check.gens,
        )
    _compute_slot_mappings_kernel[(num_groups, len(req_tokens) + 1)](
        max_num_tokens,
        idx,
        qsl_t,
        pos_t,
        ptrs,
        strides,
        sizes,
        enabled,
        dcp,
        out,
        out.stride(0),
        0,
        *gen_args,
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=16,
        GEN_CHECK=gen_check is not None,
    )
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return out[:, : len(positions)].cpu().tolist()


def _gen_check(
    gens: dict[int, int] | None = None, num_groups: int = 1, num_blocks: int = 16
) -> PercyGenCheck:
    gc = PercyGenCheck(num_blocks, [BS] * num_groups, 4, 32, _DEVICE)
    if gens:
        gc.apply_gen_updates((list(gens), list(gens.values())))
    gc.begin_windows()
    return gc


# Request 0 owns blocks 3, 5, 7 (columns 0-2); request 1 owns 9 and the null
# block at column 1 (a sliding-window revoke).
TABLE = [[3, 5, 7, 0], [9, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


@needs_kernel
def test_matching_windows_are_bit_identical_to_the_unchecked_kernel():
    gc = _gen_check({3: 2, 5: 4, 7: 1})
    gc.set_windows(0, ((0, 2, 4, 1),))
    gc.commit_windows()
    tokens = [(0, list(range(10)))]
    assert _run_kernel(TABLE, tokens, gc) == _run_kernel(TABLE, tokens, None)
    assert PAD_SLOT_ID not in _run_kernel(TABLE, tokens, gc)[0]


@needs_kernel
def test_stale_gen_pads_only_that_columns_tokens():
    gc = _gen_check({5: 4})
    # The window still holds gen 3 for block 5 (column 1): stale by one.
    gc.set_windows(0, ((0, 1, 3, 1),))
    gc.commit_windows()
    tokens = [(0, list(range(12)))]
    got = _run_kernel(TABLE, tokens, gc)[0]
    want = _run_kernel(TABLE, tokens, None)[0]
    for pos, (g, w) in enumerate(zip(got, want)):
        assert g == (PAD_SLOT_ID if pos // BS == 1 else w)


@needs_kernel
def test_write_aimed_at_the_null_block_pads():
    # Request 1's column 1 holds the null block. M1 sends gen 0 for a null
    # column; even a window claiming NULL_GEN for it must not pass.
    tokens = [(1, [0, 4, 5])]
    for null_col_gen in (0, NULL_GEN):
        gc = _gen_check()
        gc.set_windows(1, ((0, 1, null_col_gen),))
        gc.commit_windows()
        assert _run_kernel(TABLE, tokens, gc)[0] == [9 * BS, PAD_SLOT_ID, PAD_SLOT_ID]
    # Unchecked, the same tokens would have been written into the null block.
    assert _run_kernel(TABLE, tokens, None)[0] == [9 * BS, 0, 1]


@needs_kernel
def test_tokens_outside_the_window_pad():
    gc = _gen_check()
    gc.set_windows(0, ((1, 1),))  # only column 1 may be written
    gc.commit_windows()
    got = _run_kernel(TABLE, [(0, [3, 4, 7, 8])], gc)[0]
    assert got == [PAD_SLOT_ID, 5 * BS, 5 * BS + 3, PAD_SLOT_ID]


@needs_kernel
def test_refused_request_pads_every_token_and_others_are_untouched():
    gc = _gen_check()
    gc.set_windows(0, ((0, 1, 1, 1),), refused=True)
    gc.set_windows(1, ((0, 1),))
    gc.commit_windows()
    tokens = [(0, [0, 5, 9]), (1, [0, 1])]
    got = _run_kernel(TABLE, tokens, gc)[0]
    want = _run_kernel(TABLE, tokens, None)[0]
    assert got[:3] == [PAD_SLOT_ID] * 3
    assert got[3:] == want[3:]


@needs_kernel
def test_unchecked_requests_and_dummy_rows_are_unchanged():
    gc = _gen_check({3: 9})  # gens differ from any window: would fail
    gc.set_windows(0, ())  # unchecked
    gc.commit_windows()
    tokens = [(0, [0, 1, 4]), (-1, [0, 1])]
    assert _run_kernel(TABLE, tokens, gc) == _run_kernel(TABLE, tokens, None)


@needs_kernel
def test_group_without_window_is_unchecked():
    gc = _gen_check({3: 2}, num_groups=2)
    gc.set_windows(0, ((0, 1), None))  # group 0 stale, group 1 unchecked
    gc.commit_windows()
    got = _run_kernel(TABLE, [(0, [0, 1])], gc, block_sizes=(BS, BS))
    assert got[0] == [PAD_SLOT_ID, PAD_SLOT_ID]
    assert got[1] == [3 * BS, 3 * BS + 1]


@needs_kernel
def test_high_u32_gens_compare_by_bit_pattern():
    gen = 0x8000_0001
    gc = _gen_check({3: gen})
    gc.set_windows(0, ((0, gen),))
    gc.commit_windows()
    assert _run_kernel(TABLE, [(0, [0])], gc)[0] == [3 * BS]


@needs_kernel
def test_windows_reset_each_step():
    gc = _gen_check()
    gc.set_windows(0, ((0, 7),), refused=True)
    gc.commit_windows()
    assert _run_kernel(TABLE, [(0, [0])], gc)[0] == [PAD_SLOT_ID]
    gc.begin_windows()
    gc.commit_windows()
    assert _run_kernel(TABLE, [(0, [0])], gc)[0] == [3 * BS]


def test_gen_table_initial_state_and_updates():
    gc = _gen_check()
    table = gc.table.cpu().numpy().view(np.uint32)
    assert table[0] == NULL_GEN and (table[1:] == 1).all()
    gc.apply_gen_updates(([2, 5], [3, 0xFFFF_FFFE]))
    table = gc.table.cpu().numpy().view(np.uint32)
    assert table[2] == 3 and table[5] == 0xFFFF_FFFE and table[3] == 1


def test_window_csr_layout():
    gc = _gen_check(num_groups=2)
    gc.set_windows(1, ((4, 7, 8), (2, 9)))
    gc.set_windows(3, (None, (0, 1, 2, 3)))
    gc.set_windows(2, ((1, 5),), refused=True)
    gc.commit_windows()
    meta = gc.meta.cpu().numpy()
    gens = gc.gens.cpu().numpy()
    assert meta[0, 1].tolist() == [4, 2, 0]
    assert meta[1, 1].tolist() == [2, 1, 2]
    assert meta[0, 3, 1] == PercyGenCheck.UNCHECKED
    assert meta[1, 3].tolist() == [0, 3, 3]
    assert meta[:, 2, 1].tolist() == [PercyGenCheck.REFUSED] * 2
    assert meta[:, 0, 1].tolist() == [PercyGenCheck.UNCHECKED] * 2
    assert gens[:6].tolist() == [7, 8, 9, 1, 2, 3]


def test_window_overflow_is_a_hard_error():
    # Capacity cdiv(4, 4) + 2 * (cdiv(0, 4) + 2) = 5: a broken bound is loud.
    gc = PercyGenCheck(16, [BS], 2, 4, _DEVICE)
    gc.begin_windows()
    gc.set_windows(0, ((0, *range(1, 4)),))
    with pytest.raises(RuntimeError, match="above the bounded capacity 5"):
        gc.set_windows(1, ((0, *range(1, 4)),))


def _m1_window_cols(computed: int, scheduled: int, slack: int, look: int, bs: int):
    """Columns of a write window, as PercyMessageBuilder._write_window
    computes them (before clamping to allocated columns, which only
    shrinks it)."""
    first = max(0, computed - slack) // bs
    last = -(-(computed + scheduled + look) // bs)
    return max(0, last - first)


@pytest.mark.parametrize("num_spec", [0, 3, 7, 16])
@pytest.mark.parametrize("block_sizes", [(16,), (1, 64), (16, 32, 528)])
def test_worst_case_windows_fit_the_capacity(num_spec, block_sizes):
    """max_num_reqs requests share max_num_batched_tokens, each placed at the
    misalignment that maximises its window in every group, with the largest
    slack and lookahead: the CSR must fit without overflowing."""
    max_reqs, max_tokens = 64, 1024
    slack = look = num_spec
    gc = PercyGenCheck(
        1, list(block_sizes), max_reqs, max_tokens, _DEVICE, slack + look
    )
    gc.begin_windows()
    # Every request gets at least one token; the rest go to request 0.
    scheduled = [max_tokens - (max_reqs - 1)] + [1] * (max_reqs - 1)
    for req, s in enumerate(scheduled):
        windows = []
        for bs in block_sizes:
            n = max(
                _m1_window_cols(computed, s, slack, look, bs)
                for computed in range(slack, slack + bs)
            )
            windows.append((0, *([1] * n)))
        gc.set_windows(req, tuple(windows))
    gc.commit_windows()
    used = gc._num_gens
    assert used <= gc.gens_np.shape[0]


def test_wrong_group_count_is_unchecked():
    gc = _gen_check(num_groups=2)
    gc.set_windows(0, ((0, 1),))
    gc.commit_windows()
    assert gc.meta.cpu().numpy()[:, 0, 1].tolist() == [PercyGenCheck.UNCHECKED] * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refused_write_leaves_kv_bytes_unchanged():
    """A slot built from (block, gen - 1) is PAD and reshape_and_cache_flash
    leaves the block's KV bytes untouched."""
    from vllm import _custom_ops as ops

    gc = _gen_check({5: 4})
    gc.set_windows(0, ((1, 3),))  # stale gen for block 5
    gc.commit_windows()
    slots = _run_kernel(TABLE, [(0, [4, 5, 6, 7])], gc)[0]
    assert slots == [PAD_SLOT_ID] * 4
    num_heads, head_size, num_blocks = 2, 16, 8
    shape = (num_blocks, BS, num_heads, head_size)
    key_cache = torch.randn(shape, dtype=torch.float16, device="cuda")
    value_cache = torch.randn(shape, dtype=torch.float16, device="cuda")
    before = (key_cache.clone(), value_cache.clone())
    key = torch.randn(4, num_heads, head_size, dtype=torch.float16, device="cuda")
    value = torch.randn_like(key)
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        torch.tensor(slots, dtype=torch.int64, device="cuda"),
        "auto",
        scale,
        scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(key_cache, before[0])
    assert torch.equal(value_cache, before[1])
