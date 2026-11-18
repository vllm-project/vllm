# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How kernels find rows: which GPU owns an id (transports), and where a
fetched row is once it has landed (`landed_row`, for models' kernels)."""

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton

# A failed transfer or dead proxy must not hang the GPU forever.
TIMEOUT_NS: tl.constexpr = tl.constexpr(60 * 10**9)


class Tables(NamedTuple):
    """A fetch's tables: rows per shard and rows `[L]`, and per shard its address,
    NIC key and gathering NIC's rank `[L, world]` (the last two for RDMA)."""

    rows_per_shard: torch.Tensor
    num_rows: torch.Tensor
    bases: torch.Tensor
    keys: torch.Tensor
    gatherers: torch.Tensor


class Landed(NamedTuple):
    """Where one table's fetched rows are: in place in `rows`, or (RDMA) at
    `slots` of `rows` once every QP's flag holds this fetch's `seq`."""

    rows: torch.Tensor
    slots: torch.Tensor | None = None
    seq: torch.Tensor | None = None
    used: torch.Tensor | None = None
    flags: torch.Tensor | None = None
    done: torch.Tensor | None = None
    world: int = 0
    qps: int = 0
    placed: tl.constexpr = tl.constexpr(True)


@triton.jit
def locate(ids, tables, item, valid, world, TABLES: tl.constexpr, K: tl.constexpr):
    """Item `item` of `[..., TABLES, K]` ids: its table, its row's shard (an
    index into `tables`' `[L, world]` arrays), the row there, and liveness."""
    index = item // K % TABLES
    row = tl.load(ids + item, mask=valid, other=-1).to(tl.int64)
    rows_per_shard = tl.load(tables.rows_per_shard + index, mask=valid, other=1)
    num_rows = tl.load(tables.num_rows + index, mask=valid, other=0)
    live = valid & (row >= 0) & (row < num_rows)
    owner = tl.where(live, row // rows_per_shard, 0)
    return index, index * world + owner, row - owner * rows_per_shard, live


@triton.jit
def wait_landed(landed, table):
    """Spin until every QP that gathered rows of `table` has flagged this
    fetch's `seq`; trap after TIMEOUT_NS."""
    seq = tl.load(landed.seq)
    # The first program to see every flag records `seq` here for the others.
    done = landed.done + table
    waiting = 0
    # Acquire: rows are read only after the word that says they landed.
    mark = tl.inline_asm_elementwise(
        "ld.acquire.gpu.global.b64 $0, [$1];",
        "=l,l",
        [done.to(tl.int64)],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    if mark != seq:
        deadline = tl.extra.cuda.globaltimer() + TIMEOUT_NS
        q = tl.arange(0, 16)[None, :]
        # Eight gatherers' flags (QPs per peer <= 16) per load.
        for first in range(0, landed.world, 8):
            peer = first + tl.arange(0, 8)
            group = table * landed.world + peer
            count = tl.load(landed.used + group, mask=peer < landed.world, other=0)
            at = landed.flags + group[:, None] * landed.qps + q
            waiting = 1
            while (waiting > 0) & (tl.extra.cuda.globaltimer() < deadline):
                flag = tl.load(at, mask=q < count[:, None], other=seq, volatile=True)
                waiting = tl.sum(tl.sum((flag != seq).to(tl.int32), 1))
        if waiting == 0:
            tl.atomic_xchg(done, seq, sem="release")
    if waiting > 0:
        tl.device_print("Engram pool RDMA fetch timed out, seq", seq)
        tl.inline_asm_elementwise(
            "trap;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )


@triton.jit
def landed_row(landed, table, row, valid, ROW_STRIDE: tl.constexpr):
    """Address of fetched row `row` of `table` (`landed` from `EngramFetch`), and
    whether it holds data (else load zeros). Call from every program."""
    slot, live = row, valid
    if not landed.placed:
        wait_landed(landed, table)
        slot = tl.load(landed.slots + row, mask=valid, other=-1).to(tl.int64)
        live = valid & (slot >= 0)
    offset = tl.multiple_of(slot * ROW_STRIDE, ROW_STRIDE & -ROW_STRIDE)
    return landed.rows + offset, live
