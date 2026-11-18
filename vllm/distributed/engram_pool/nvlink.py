# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer HBM over NVLink: every GPU maps every shard (cuMem FABRIC handles), so
a fetch is one kernel that loads each row from its owner's memory."""

import torch

from vllm.distributed.engram_pool.memory import CuMem, device_view
from vllm.distributed.engram_pool.rows import Tables, locate
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_items"])
def _gather_kernel(
    ids,
    tables,
    out,
    num_items,
    WORLD: tl.constexpr,
    TABLES: tl.constexpr,
    K: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Load `[..., TABLES, K]` rows from their owners into `[TABLES, ..., K]`
    rows, one 8-byte word per lane; dead ids give zeros."""
    word = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    item = word // WORDS
    valid = item < num_items
    index, shard, local, live = locate(ids, tables, item, valid, WORLD, TABLES, K)
    base = tl.load(tables.bases + shard, mask=live, other=0)
    column = word - item * WORDS
    data = tl.load(
        base.to(tl.pointer_type(tl.uint64)) + local * WORDS + column, mask=live, other=0
    )
    place = index * (num_items // TABLES) + item // (TABLES * K) * K + item % K
    tl.store(
        out.to(tl.pointer_type(tl.uint64)) + place * WORDS + column, data, mask=valid
    )


class NvlinkTransport:
    """`group` provides rank_in_group, world_size and all_gather(bytes)."""

    def __init__(self, group) -> None:
        self.group = group

    def shard(self, num_bytes: int, name: str) -> tuple[torch.Tensor, tuple, list]:
        """Allocate this GPU's shard and map every GPU's. Returns the shard,
        every shard's address, key and gatherer, and what keeps them. Collective."""
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
        zeros = [0] * len(addresses)
        return device_view(own, num_bytes, memory), (addresses, zeros, zeros), []

    def start(self, tables: Tables, ids: torch.Tensor, fetch) -> None:
        if n := ids.numel():
            words = fetch.row_stride // 8
            _gather_kernel[(triton.cdiv(n * words, 1024),)](
                ids,
                tables,
                fetch.rows,
                n,
                WORLD=self.group.world_size,
                TABLES=tables.num_rows.numel(),
                K=ids.shape[-1],
                WORDS=words,
                BLOCK=1024,
            )

    @staticmethod
    def wait(fetch: None, table: int, out: torch.Tensor) -> None:
        """Rows are in place once `start` returns: SMs moved them."""

    @staticmethod
    def landed(fetch: None, table: int) -> None:
        """Rows are in place: see `wait`."""

    def close(self) -> None:
        pass
