# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer device or host memory over NVLink: every GPU maps each shard (cuMem FABRIC), so
a fetch is one kernel that loads each row from its owner's memory."""

from contextlib import ExitStack

import torch

from vllm.distributed.engram_pool.group import PeerGroup
from vllm.distributed.engram_pool.memory import CuMem, device_view
from vllm.distributed.engram_pool.rows import SHARD_BASES, locate
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
    rows, one 8-byte word per lane; dead ids give zeros."""
    word = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    item = word // WORDS
    valid = item < num_items
    index, table, owner, local, live = locate(ids, meta, item, valid, TABLES, K)
    shards = tl.load(table + SHARD_BASES, mask=live, other=0)
    base = tl.load(shards.to(tl.pointer_type(tl.int64)) + owner, mask=live, other=0)
    column = word - item * WORDS
    data = tl.load(
        base.to(tl.pointer_type(tl.uint64)) + local * WORDS + column, mask=live, other=0
    )
    place = index * (num_items // TABLES) + item // (TABLES * K) * K + item % K
    tl.store(
        out.to(tl.pointer_type(tl.uint64)) + place * WORDS + column, data, mask=valid
    )


class NvlinkTransport:
    def __init__(self, group: PeerGroup, host_shards: bool) -> None:
        self.group = group
        self.host_shards = host_shards

    def shard(self, num_bytes: int, name: str) -> tuple[torch.Tensor, list, list]:
        """Allocate this GPU's shard and map every GPU's. Returns the shard, its
        `meta` arrays (shard addresses) and what keeps them alive. Collective."""
        with ExitStack() as rollback:
            with self.group.startup_stage(f"Allocating Engram table {name!r}"):
                memory = CuMem(num_bytes, fabric=True, host_numa=self.host_shards)
                rollback.callback(memory.release, memory.handle)
                handle = memory.export_fabric()

            handles = self.group.all_gather(handle)
            with self.group.startup_stage(f"Mapping Engram table {name!r}"):
                addresses = []
                for rank, handle in enumerate(handles):
                    if rank == self.group.rank_in_group:
                        address = memory.map()
                    else:
                        address = memory.map_fabric(handle, read_only=self.host_shards)
                    rollback.callback(memory.unmap, address)
                    addresses.append(address)
                shard = device_view(memory.address, num_bytes, memory)
                bases = torch.tensor(addresses, dtype=torch.int64, device="cuda")
            rollback.pop_all()
        return shard, [bases.data_ptr(), 0, 0], [bases]

    def start(self, meta: torch.Tensor, ids: torch.Tensor, fetch) -> None:
        if n := ids.numel():
            out, words = fetch.rows, fetch.row_stride // 8
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

    @staticmethod
    def wait(fetch: None, table: int, out: torch.Tensor) -> None:
        """Rows are in place once `start` returns: SMs moved them."""

    @staticmethod
    def landed(fetch: None, table: int) -> None:
        """Rows are in place: see `wait`."""

    def close(self) -> None:
        pass
