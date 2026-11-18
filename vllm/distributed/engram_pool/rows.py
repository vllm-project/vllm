# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How kernels find rows: which GPU owns an id (transports), and where a
fetched row is once it has landed (`landed_row`, for models' kernels)."""

from vllm.triton_utils import tl, triton

# A table's int64 `meta` words: its split, then device arrays per shard.
ROWS_PER_SHARD: tl.constexpr = tl.constexpr(0)
NUM_ROWS: tl.constexpr = tl.constexpr(1)
SHARD_BASES: tl.constexpr = tl.constexpr(2)
SHARD_KEYS: tl.constexpr = tl.constexpr(3)
GATHERERS: tl.constexpr = tl.constexpr(4)
META_WORDS: tl.constexpr = tl.constexpr(5)

# An RDMA fetch's `ready` words.
READY_SEQ: tl.constexpr = tl.constexpr(0)
READY_USED: tl.constexpr = tl.constexpr(1)
READY_FLAGS: tl.constexpr = tl.constexpr(2)
READY_WORLD: tl.constexpr = tl.constexpr(3)
READY_QPS: tl.constexpr = tl.constexpr(4)
READY_TIMEOUT: tl.constexpr = tl.constexpr(5)
READY_LANDED: tl.constexpr = tl.constexpr(6)
READY_FAILED: tl.constexpr = tl.constexpr(7)


@triton.jit
def locate(ids, meta, item, valid, TABLES: tl.constexpr, K: tl.constexpr):
    """Item `item` of `[..., TABLES, K]` ids: its table index and `meta` row,
    the owner GPU of its row and the row's index there, and whether it is live."""
    index = item // K % TABLES
    table = meta + index * META_WORDS
    row = tl.load(ids + item, mask=valid, other=-1).to(tl.int64)
    rows_per_shard = tl.load(table + ROWS_PER_SHARD, mask=valid, other=1)
    num_rows = tl.load(table + NUM_ROWS, mask=valid, other=0)
    live = valid & (row >= 0) & (row < num_rows)
    owner = tl.where(live, row // rows_per_shard, 0)
    return index, table, owner, row - owner * rows_per_shard, live


@triton.jit
def wait_landed(ready, table):
    """Spin until every QP that gathered rows of `table` in the fetch `ready`
    describes has flagged them landed; trap after the timeout."""
    seq = tl.load(tl.load(ready + READY_SEQ).to(tl.pointer_type(tl.int64)))
    # The first program to see every flag records `seq` here for the others.
    landed = tl.load(ready + READY_LANDED).to(tl.pointer_type(tl.int64)) + table
    waiting = 0
    # Acquire: rows are read only after the word that says they landed.
    mark = tl.inline_asm_elementwise(
        "ld.acquire.gpu.global.b64 $0, [$1];",
        "=l,l",
        [landed.to(tl.int64)],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )
    if mark != seq:
        used = tl.load(ready + READY_USED).to(tl.pointer_type(tl.int32))
        flags = tl.load(ready + READY_FLAGS).to(tl.pointer_type(tl.int64))
        world, qps = tl.load(ready + READY_WORLD), tl.load(ready + READY_QPS)
        deadline = tl.extra.cuda.globaltimer() + tl.load(ready + READY_TIMEOUT)
        # Set by the host once a request failed or the proxy exited.
        failed = tl.load(ready + READY_FAILED).to(tl.pointer_type(tl.int64))
        q = tl.arange(0, 16)[None, :]
        # Eight gatherers' flags (QPs per peer <= 16) per load.
        for first in range(0, world, 8):
            peer = first + tl.arange(0, 8)
            group = table * world + peer
            count = tl.load(used + group, mask=peer < world, other=0)
            at = flags + group[:, None] * qps + q
            waiting = 1
            while (
                (waiting > 0)
                & (tl.extra.cuda.globaltimer() < deadline)
                & (tl.load(failed, volatile=True) == 0)
            ):
                flag = tl.load(at, mask=q < count[:, None], other=seq, volatile=True)
                waiting = tl.sum(tl.sum((flag != seq).to(tl.int32), 1))
        if waiting == 0:
            tl.atomic_xchg(landed, seq, sem="release")
    if waiting > 0:
        tl.device_print("Engram pool RDMA fetch failed (see log) or timed out", seq)
        tl.inline_asm_elementwise(
            "trap;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )


@triton.jit
def landed_row(
    rows,
    slots,
    ready,
    table,
    row,
    valid,
    ROW_STRIDE: tl.constexpr,
    PLACED: tl.constexpr,
):
    """Address of fetched row `row` of `table` (args from `EngramFetch.landed`),
    and whether it holds data (else load zeros). Call from every program."""
    slot, live = row, valid
    if not PLACED:
        wait_landed(ready, table)
        slot = tl.load(slots + row, mask=valid, other=-1).to(tl.int64)
        live = valid & (slot >= 0)
    return rows + tl.multiple_of(slot * ROW_STRIDE, ROW_STRIDE & -ROW_STRIDE), live
