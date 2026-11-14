# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer HBM over NVLink: every GPU maps every shard (cuMem FABRIC handles), so
a fetch is one kernel that loads each row from its owner's memory."""

import torch

from vllm.distributed.engram_pool.memory import CuMem, device_view
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_items"])
def _gather_kernel(
    ids,
    meta,
    out,
    num_items,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Load `[..., TABLES, K]` rows from their owners into `[TABLES, ..., K]`
    rows, one 8-byte word per lane; dead ids give zeros. `meta` rows hold rows
    per shard, rows and a device array of shard addresses (words 1, 2, 3)."""
    word = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    item = word // WORDS
    valid = item < num_items
    index = item // K % TABLES
    table = meta + index * 8
    row = tl.load(ids + item, mask=valid, other=-1).to(tl.int64)
    rows_per_shard = tl.load(table + 1, mask=valid, other=1)
    live = valid & (row >= 0) & (row < tl.load(table + 2, mask=valid, other=0))
    owner = tl.where(live, row // rows_per_shard, 0)
    local = row - owner * rows_per_shard
    shards = tl.load(table + 3, mask=live, other=0).to(tl.pointer_type(tl.int64))
    base = tl.load(shards + owner, mask=live, other=0).to(tl.pointer_type(tl.uint64))
    column = word - item * WORDS
    data = tl.load(base + local * WORDS + column, mask=live, other=0)
    place = index * (num_items // TABLES) + item // (TABLES * K) * K + item % K
    tl.store(
        out.to(tl.pointer_type(tl.uint64)) + place * WORDS + column, data, mask=valid
    )


class NvlinkTransport:
    """`group` provides rank_in_group and all_gather(bytes)."""

    def __init__(self, group) -> None:
        self.group = group

    def share(self, num_bytes: int) -> tuple[torch.Tensor, list[int]]:
        """Allocate this GPU's shard and map every GPU's; returns the shard
        and every shard's address in rank order. Collective."""
        memory = CuMem(num_bytes, fabric=True)
        exported = self.group.all_gather(memory.export_fabric())
        addresses = []
        for rank, handle in enumerate(exported):
            if rank == self.group.rank_in_group:
                addresses.append(memory.map())
                continue
            imported = memory.import_fabric(handle)
            addresses.append(memory.map(imported))
            memory.release(imported)
        own = addresses[self.group.rank_in_group]
        return device_view(own, num_bytes, memory), addresses

    def wake(self) -> None:
        """Nothing to wake: SMs move rows."""

    @staticmethod
    def wait(state: None, table: int) -> None:
        """Rows are in place once `start` returns: SMs moved them."""

    @staticmethod
    def unplaced(state: None, table: int) -> None:
        """Rows are in place: see `wait`."""

    def start(
        self,
        meta: torch.Tensor,
        ids: torch.Tensor,
        out: torch.Tensor,
        words: int,
        slots: torch.Tensor,
    ) -> None:
        if n := ids.numel():
            _gather_kernel[(triton.cdiv(n * words, 1024),)](
                ids,
                meta,
                out,
                n,
                TABLES=meta.shape[0],
                K=ids.shape[-1],
                WORDS=words,
                BLOCK=1024,
            )
