# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only lookup tables split by rows over every GPU of a peer group.

The pool owns registration, rendezvous, row sharding, address maps and
fetches; models own hashing, row encoding and decoding. A fetch gathers
`[..., L, K]` row ids of L tables into `[..., L, K, row_bytes]` HBM rows on a
stream, with no host sync, so it captures into CUDA graphs.

Transports: `nvlink` maps every GPU's HBM shard (cuMem FABRIC) and loads rows
directly; `rdma` reads remote rows with one-sided READs from host or HBM
shards, posted by a proxy process per GPU.
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
    """A fetch running on the pool's stream; `wait()` returns its rows."""

    def __init__(self, rows: torch.Tensor, done: torch.cuda.Event) -> None:
        self.rows, self.done = rows, done

    def wait(self) -> torch.Tensor:
        torch.cuda.current_stream().wait_event(self.done)
        return self.rows


class EngramPool:
    """Tables shared by a peer group, and the buffers fetches land in.

    Every GPU of the group creates the same pool and tables in the same order
    (each call is collective). A fetch's rows stay valid until the next fetch
    on the pool starts; fetches start on the pool stream after all work
    already queued on the caller's stream.

    Args:
        group: The peer group splitting the tables.
        transport: "nvlink" or "rdma".
        host_shards: Keep shards in host memory (RDMA only).
        max_items: Most rows one fetch gathers on this GPU.
        max_row_bytes: Largest row of any table.

    """

    def __init__(
        self,
        group: PeerGroup,
        transport: str,
        host_shards: bool,
        max_items: int,
        max_row_bytes: int,
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

            self.transport = RdmaTransport(group, max_items, max_row_bytes)
        self.kind = transport
        self.output = torch.empty(
            max_items * max_row_bytes, dtype=torch.uint8, device=self.device
        )
        self.stream = torch.cuda.Stream(device=self.device)
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
        keys_ptr = mapped_ptr = gatherers_ptr = 0
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
            bases, keys, gatherers, mapped = self.transport.register(registered)  # type: ignore[attr-defined]
            arrays = [
                torch.tensor(values, dtype=dtype, device=self.device)
                for values, dtype in (
                    (bases, torch.int64),
                    (keys, torch.int32),
                    (mapped, torch.int64),
                    (gatherers, torch.int32),
                )
            ]
            keep += arrays
            bases_ptr, keys_ptr, mapped_ptr, gatherers_ptr = (
                array.data_ptr() for array in arrays
            )
        meta = torch.tensor(
            [local_gpu.data_ptr(), rows_per_shard, num_rows, bases_ptr, keys_ptr]
            + [mapped_ptr, gatherers_ptr, 0],
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

    def lookup(self, tables: list[EngramTable], ids: torch.Tensor) -> torch.Tensor:
        """Rows `[*ids.shape, row_bytes]` (uint8) of `ids` (`[..., L, K]`, row
        L of table L); ids < 0 or past a table give zero rows."""
        if ids.shape[-2] != len(tables):
            raise ValueError(
                f"ids {tuple(ids.shape)} do not match {len(tables)} tables"
            )
        meta = self._meta(tables)
        row_bytes = tables[0].row_bytes
        rows = self.output[: ids.numel() * row_bytes].view(*ids.shape, row_bytes)
        self.transport.fetch(meta, ids.contiguous(), rows, row_bytes // 8)  # type: ignore[attr-defined]
        return rows

    def prefetch(self, tables: list[EngramTable], ids: torch.Tensor) -> EngramFetch:
        """Run `lookup` on the pool stream, overlapping the caller's work."""
        self.stream.wait_stream(torch.cuda.current_stream())
        ids.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            rows = self.lookup(tables, ids)
        done = torch.cuda.Event()
        done.record(self.stream)
        return EngramFetch(rows, done)

    def close(self) -> None:
        if hasattr(self.transport, "close"):
            self.transport.close()
