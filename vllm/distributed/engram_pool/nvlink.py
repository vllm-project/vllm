# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer HBM or EGM over NVLink: every GPU maps each shard (cuMem FABRIC), so
a fetch is one kernel that loads each row from its owner's memory."""

import torch

from vllm.distributed.engram_pool.memory import CuMem, device_view
from vllm.distributed.engram_pool.rows import SHARD_BASES, locate
from vllm.platforms import current_platform
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
    """`group` provides rank_in_group and all_gather(bytes)."""

    def __init__(self, group, egm: bool) -> None:
        self.group = group
        self.egm = egm

    def _map_shards(self, memory, exported, addresses, num_bytes):
        for rank, handle in enumerate(exported):
            if rank == self.group.rank_in_group:
                addresses.append(memory.map())
            else:
                imported = memory.import_fabric(handle)
                try:
                    addresses.append(memory.map(imported, read_only=self.egm))
                finally:
                    memory.release(imported)
        own = addresses[self.group.rank_in_group]
        bases = torch.tensor(addresses, dtype=torch.int64, device="cuda")
        return device_view(own, num_bytes, memory), [bases.data_ptr(), 0, 0], [bases]

    def shard(self, num_bytes: int, name: str) -> tuple[torch.Tensor, list, list]:
        """Allocate this GPU's shard and map every GPU's. Returns the shard, its
        `meta` arrays (shard addresses) and what keeps them alive. Collective."""
        if not self.egm:
            device_memory = CuMem(num_bytes, fabric=True)
            exported = self.group.all_gather(device_memory.export_fabric())
            return self._map_shards(device_memory, exported, [], num_bytes)

        memory: CuMem | None = None
        addresses: list[int] = []
        try:
            error = None
            exported_handle = b""
            try:
                host_numa = current_platform.get_device_numa_node(
                    torch.accelerator.current_device_index()
                )
                if host_numa is None:
                    raise RuntimeError("CUDA host NUMA node is unavailable")
                memory = CuMem(num_bytes, fabric=True, host_numa=host_numa)
                exported_handle = memory.export_fabric()
            except RuntimeError as exc:
                error = str(exc) or type(exc).__name__
            exchanged = self.group.all_gather(
                b"E" + error.encode() if error is not None else b"O" + exported_handle
            )
            failures = [
                f"rank {i}: {item[1:].decode(errors='replace')}"
                for i, item in enumerate(exchanged)
                if item[:1] == b"E"
            ]
            if failures:
                raise RuntimeError(
                    "NVLink Engram EGM allocation failed: " + "; ".join(failures)
                )
            assert memory is not None
            error = None
            try:
                result = self._map_shards(
                    memory, [item[1:] for item in exchanged], addresses, num_bytes
                )
            except RuntimeError as exc:
                error = str(exc) or type(exc).__name__
            exchanged = self.group.all_gather((error or "").encode())
            failures = [
                f"rank {i}: {item.decode(errors='replace')}"
                for i, item in enumerate(exchanged)
                if item
            ]
            if failures:
                raise RuntimeError(
                    "NVLink Engram EGM mapping failed: " + "; ".join(failures)
                )
            return result
        except (RuntimeError, TimeoutError):
            if memory is not None:
                for address in addresses:
                    memory.unmap(address)
                memory.release(memory.handle)
            raise

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
