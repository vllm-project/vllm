# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only lookup tables split by rows over every GPU of a peer group.

The pool owns registration, rendezvous, row sharding, address maps and
fetches; models own hashing, row encoding and decoding. A fetch gathers
`[..., L, K]` row ids of L tables into `[L, ..., K, row_bytes]` HBM rows on a
stream, with no host sync, so it captures into CUDA graphs.

Transports: `nvlink` maps every GPU's HBM shard (cuMem FABRIC) and loads rows
directly; with `rdma`, the NICs of the GPUs holding host or HBM shards gather
rows into RDMA WRITEs that each GPU builds, carried by proxy processes.
"""

import json
from dataclasses import dataclass, field

import torch

from vllm.distributed.engram_pool.group import PeerGroup, join_peer_group
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = ["EngramPool", "EngramTable", "PeerGroup", "join_peer_group"]

# Rows are copied 8 bytes at a time.
ROW_ALIGN = 8


@dataclass(eq=False)
class EngramTable:
    """One table's rows, split evenly over the group's GPUs in rank order.

    `shard` is this GPU's rows (host memory for host shards), for loading;
    `peer_ptrs` (NVLink) is every shard's GPU address, for fused kernels.
    """

    num_rows: int
    row_bytes: int
    rows_per_shard: int
    row_start: int
    shard: torch.Tensor
    meta: torch.Tensor
    peer_ptrs: torch.Tensor | None = None
    _keep: list = field(default_factory=list)

    def load(self, row_start: int, rows: torch.Tensor, col: int = 0) -> None:
        """Store `rows` (table rows from `row_start` on) at byte column `col`,
        keeping only the rows this GPU holds."""
        rows = rows.reshape(rows.shape[0], -1).view(torch.uint8)
        lo = max(row_start, self.row_start)
        hi = min(row_start + rows.shape[0], self.row_start + self.shard.shape[0])
        if lo < hi:
            self.shard[lo - self.row_start : hi - self.row_start].narrow(
                1, col, rows.shape[1]
            ).copy_(rows[lo - row_start : hi - row_start])


class EngramFetch:
    """A started fetch: `wait(i)` returns table i's rows once they landed."""

    def __init__(self, transport, rows: torch.Tensor, state) -> None:
        self.transport, self.rows, self.state = transport, rows, state

    def wait(self, index: int) -> torch.Tensor:
        """Rows `[..., K, row_bytes]` of table `index`; the current stream
        waits for them."""
        self.transport.wait(self.state, index)
        return self.rows[index]

    def unplaced(self, index: int) -> tuple:
        """Table `index`'s rows where they land, for a kernel to read without
        `wait`: `(buffer, slots, ready)`. Such a kernel first calls
        `rdma.wait_landed(ready, index)`, then reads row `slots[i]` of `buffer`
        for row i of `[..., K]` (zeros if -1). Rows that SMs moved (NVLink) are
        in place already: `(rows, None, None)`. Launches nothing."""
        got = self.transport.unplaced(self.state, index)
        return (self.rows[index], None, None) if got is None else got


class EngramPool:
    """Tables shared by a peer group, and the buffers fetches land in.

    Every GPU of the group creates the same pool and tables in the same order
    (each call is collective). Fetches run on the caller's stream; over RDMA,
    rows move without SMs between `start` and each table's `wait`. A fetch's
    rows stay valid until the next starts, which must follow every wait.

    Args:
        group: The peer group splitting the tables.
        transport: "nvlink" or "rdma".
        host_shards: Keep shards in host memory (RDMA only).
        max_items: Most rows of one table one fetch gathers on this GPU.
        max_row_bytes: Largest row of any table.
        max_tables: Most tables one fetch reads.

    """

    def __init__(
        self,
        group: PeerGroup,
        transport: str,
        host_shards: bool,
        max_items: int,
        max_row_bytes: int,
        max_tables: int = 1,
    ) -> None:
        if transport not in ("nvlink", "rdma"):
            raise ValueError(f"Unknown Engram pool transport {transport!r}")
        if host_shards and transport != "rdma":
            raise ValueError("Host shards require the rdma transport")
        if max_row_bytes % ROW_ALIGN:
            raise ValueError(f"Rows must be a multiple of {ROW_ALIGN} bytes")
        self._check_same(group, ("pool", transport, host_shards))
        self.group, self.host_shards = group, host_shards
        self.max_row_bytes = max_row_bytes
        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        if transport == "nvlink":
            from vllm.distributed.engram_pool.nvlink import NvlinkTransport

            self.transport: object = NvlinkTransport(group)
        else:
            from vllm.distributed.engram_pool.rdma import RdmaTransport

            self.transport = RdmaTransport(group, max_items, max_row_bytes, max_tables)
        self.kind = transport
        self.output = torch.empty(
            max_items * max_tables * max_row_bytes,
            dtype=torch.uint8,
            device=self.device,
        )
        # Each row's slot where it lands, for `unplaced`.
        self.slots = torch.empty(
            max_tables * max_items, dtype=torch.int32, device=self.device
        )
        self._metas: dict[tuple[int, ...], torch.Tensor] = {}

    @staticmethod
    def _check_same(group: PeerGroup, identity: tuple) -> None:
        """Fail startup on every GPU unless all of them passed `identity`."""
        mine = json.dumps(identity)
        everyone = [v.decode() for v in group.all_gather(mine.encode())]
        if differ := [r for r, theirs in enumerate(everyone) if theirs != mine]:
            raise RuntimeError(
                f"Engram pool GPUs {differ} disagree with GPU "
                f"{group.rank_in_group}: {[everyone[r] for r in differ]} vs {identity}"
            )

    def table(self, num_rows: int, row_bytes: int, name: str = "") -> EngramTable:
        """Create a table of `num_rows` rows of `row_bytes` bytes. Collective;
        `name` identifies it across the group."""
        if row_bytes % ROW_ALIGN or row_bytes > self.max_row_bytes:
            raise ValueError(
                f"Row of {row_bytes} bytes must be a multiple of {ROW_ALIGN} and "
                f"at most {self.max_row_bytes}"
            )
        self._check_same(self.group, ("table", name, num_rows, row_bytes))
        rank, world = self.group.rank_in_group, self.group.world_size
        rows_per_shard = -(-num_rows // world)
        row_start = min(rank * rows_per_shard, num_rows)
        shard_rows = min(rows_per_shard, num_rows - row_start)
        num_bytes = max(rows_per_shard * row_bytes, ROW_ALIGN)
        keep: list = []
        keys_ptr = gatherers_ptr = 0
        peer_ptrs = None
        if self.kind == "nvlink":
            local, addresses = self.transport.share(num_bytes)  # type: ignore[attr-defined]
            peer_ptrs = torch.tensor(addresses, dtype=torch.int64, device=self.device)
            bases_ptr, local_gpu = peer_ptrs.data_ptr(), local
        else:
            from vllm.distributed.engram_pool.memory import (
                CuMem,
                SharedHost,
                device_view,
            )

            registered: object
            if self.host_shards:
                host = SharedHost(num_bytes, f"table_{name}")
                local, local_gpu, registered = host.cpu, host.gpu, host
                keep.append(host)
            else:
                cumem = CuMem(num_bytes, rdma=True)
                address = cumem.map()
                local = local_gpu = device_view(address, num_bytes, cumem)
                registered = (cumem, address)
            bases, keys, gatherers = self.transport.register(registered)  # type: ignore[attr-defined]
            arrays = [
                torch.tensor(values, dtype=dtype, device=self.device)
                for values, dtype in (
                    (bases, torch.int64),
                    (keys, torch.int32),
                    (gatherers, torch.int32),
                )
            ]
            keep += arrays
            bases_ptr, keys_ptr, gatherers_ptr = (array.data_ptr() for array in arrays)
        meta = torch.tensor(
            [local_gpu.data_ptr(), rows_per_shard, num_rows, bases_ptr, keys_ptr]
            + [gatherers_ptr, 0, 0],
            dtype=torch.int64,
        )
        shard = local[: shard_rows * row_bytes].view(shard_rows, row_bytes)
        logger.info(
            "Engram pool table %r: rows [%d, %d) of %d on this GPU, %.2f GiB in %s",
            name,
            row_start,
            row_start + shard_rows,
            num_rows,
            num_bytes / 2**30,
            "host memory" if self.host_shards else "HBM",
        )
        return EngramTable(
            num_rows, row_bytes, rows_per_shard, row_start, shard, meta, peer_ptrs, keep
        )

    def ready(self) -> None:
        """Serve only once every GPU has loaded its shards: peers read them."""
        torch.accelerator.synchronize()
        self.group.barrier()

    def _meta(self, tables: list[EngramTable]) -> torch.Tensor:
        key = tuple(id(t) for t in tables)
        if (meta := self._metas.get(key)) is None:
            if len({t.row_bytes for t in tables}) != 1:
                raise ValueError("One fetch's tables must have equal row sizes")
            meta = torch.stack([t.meta for t in tables]).to(self.device)
            self._metas[key] = meta
        return meta

    def wake(self) -> None:
        """Ready the transport for the next `start` on the current stream; work
        queued in between hides its latency (RDMA: ~45 us)."""
        self.transport.wake()  # type: ignore[attr-defined]

    def start(self, tables: list[EngramTable], ids: torch.Tensor) -> EngramFetch:
        """Start fetching `ids` (`[..., L, K]`, row L of table L) on the
        current stream; ids < 0 or past a table give zero rows."""
        if ids.shape[-2] != len(tables):
            raise ValueError(
                f"ids {tuple(ids.shape)} do not match {len(tables)} tables"
            )
        meta = self._meta(tables)
        row_bytes = tables[0].row_bytes
        shape = (len(tables), *ids.shape[:-2], ids.shape[-1], row_bytes)
        rows = self.output[: ids.numel() * row_bytes].view(shape)
        slots = self.slots[: ids.numel()].view(len(tables), -1)
        state = self.transport.start(  # type: ignore[attr-defined]
            meta, ids.contiguous(), rows, row_bytes // 8, slots
        )
        return EngramFetch(self.transport, rows, state)

    def lookup(self, tables: list[EngramTable], ids: torch.Tensor) -> torch.Tensor:
        """Rows `[L, ..., K, row_bytes]` (uint8) of `ids`, as `start` takes."""
        fetch = self.start(tables, ids)
        for index in range(len(tables)):
            fetch.wait(index)
        return fetch.rows

    def close(self) -> None:
        if hasattr(self.transport, "close"):
            self.transport.close()
