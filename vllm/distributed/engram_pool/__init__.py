# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only tables split by rows over every GPU of a peer group.

The pool owns sharding, rendezvous, transfers and landing buffers; a model owns
hashing, row formats and the kernels that read fetched rows. A model:
1. creates `get_engram_pool(...)`'s tables, loads `table.shard`, then calls
   `pool.ready(table)`;
2. per step, calls `EngramPrefetch.start(ids)` before its layers and `wait(i)`
   in layer i; its `consume` reads `fetch.wait(i)`, or `fetch.landed(i)` in its
   own kernel through `landed_row`.

Transports: `nvlink` maps every GPU's HBM shard (cuMem FABRIC) and loads rows
directly; with `rdma`, the owners' NICs gather rows from host or HBM shards.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed.engram_pool.group import PeerGroup, join_peer_group
from vllm.distributed.engram_pool.rows import landed_row
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import current_stream

if TYPE_CHECKING:
    from vllm.distributed.engram_pool.nvlink import NvlinkTransport
    from vllm.distributed.engram_pool.rdma import RdmaTransport

logger = init_logger(__name__)

__all__ = [
    "EngramFetch",
    "EngramPool",
    "EngramPrefetch",
    "EngramTable",
    "Landed",
    "PeerGroup",
    "get_engram_pool",
    "join_peer_group",
    "landed_row",
]

# Transports copy rows in 8-byte words.
_ROW_ALIGN = 8


@dataclass(eq=False)
class EngramTable:
    """One table's rows, split evenly over the group's GPUs in rank order.
    `shard` is this GPU's rows from `row_start` (host memory for host shards)."""

    num_rows: int
    row_bytes: int
    row_stride: int
    row_start: int
    shard: torch.Tensor
    _meta: torch.Tensor
    _keep: list
    _loaded: bool = False


class Landed(NamedTuple):
    """`landed_row`'s view of one table of a fetch."""

    rows: torch.Tensor
    slots: torch.Tensor
    ready: torch.Tensor
    placed: bool


class EngramFetch:
    """A started fetch: `wait(i)` places table i's rows, `landed(i)` finds them."""

    def __init__(self, pool: "EngramPool", shape: tuple[int, ...]) -> None:
        self.pool, self.shape, self.row_stride = pool, shape, shape[-1]
        self.state: Any = None

    @cached_property
    def rows(self) -> torch.Tensor:
        """Placed rows, `[L, ..., K, row_stride]` uint8."""
        return self.pool._output(math.prod(self.shape)).view(self.shape)

    def wait(self, index: int) -> torch.Tensor:
        """Table `index`'s rows; the current stream waits for them."""
        rows = self.rows[index]
        self.pool.transport.wait(self.state, index, rows)
        return rows

    def landed(self, index: int) -> Landed:
        """Table `index`'s rows where they land, for a kernel that reads them
        with `landed_row` instead of `wait`. Launches nothing."""
        got = self.pool.transport.landed(self.state, index)
        if got is None:
            rows = self.rows[index]
            return Landed(rows, rows, rows, True)
        rows, slots, ready = got
        return Landed(rows, slots, ready, False)


class EngramPool:
    """Tables shared by a peer group, and the buffers fetches land in.

    Every GPU of the group creates the same pool and tables in the same order
    (each call is collective). A fetch's rows stay valid until the next starts.

    Args:
        group: The peer group splitting the tables.
        transport: "nvlink" or "rdma".
        host_shards: Keep shards in host memory (RDMA only).
        max_items: Most rows of one table one fetch gathers on this GPU.
        max_row_bytes: Largest row of any table.
        max_tables: Most tables one fetch reads.
        rdma_inflight: Most prefill-sized gathers each NIC keeps in flight
            (RDMA; 0: no limit), so they delay others' decode fetches little.
        fetch_mode: Where `EngramPrefetch` fetches: "sync" or "async".
        thp_dir: A tmpfs mounted `huge=always` for host shards, so they
            get transparent huge pages (RDMA host shards only).

    """

    def __init__(
        self,
        group: PeerGroup,
        transport: str,
        host_shards: bool,
        max_items: int,
        max_row_bytes: int,
        max_tables: int = 1,
        rdma_inflight: int = 8,
        fetch_mode: str = "sync",
        thp_dir: str | None = None,
    ) -> None:
        self.group, self.fetch_mode = group, fetch_mode
        self._check_same(("pool", transport, host_shards))
        self.max_row_bytes = round_up(max_row_bytes, _ROW_ALIGN)
        self.max_items, self.max_tables = max_items, max_tables
        self.transport: NvlinkTransport | RdmaTransport
        if transport == "nvlink":
            from vllm.distributed.engram_pool import nvlink

            self.transport = nvlink.NvlinkTransport(group)
        else:
            from vllm.distributed.engram_pool import rdma

            self.transport = rdma.RdmaTransport(
                group,
                host_shards,
                max_items,
                self.max_row_bytes,
                max_tables,
                rdma_inflight,
                thp_dir,
            )
        self._where = "host memory" if host_shards else "HBM"
        self._buffer: torch.Tensor | None = None
        self._metas: dict[tuple[int, ...], torch.Tensor] = {}

    def _check_same(self, identity: tuple) -> None:
        """Fail startup on every GPU unless all of them passed `identity`."""
        mine = json.dumps(identity)
        everyone = [v.decode() for v in self.group.all_gather(mine.encode())]
        rank = self.group.rank_in_group
        if differ := [r for r, theirs in enumerate(everyone) if theirs != mine]:
            raise RuntimeError(
                f"Engram pool GPUs {differ} disagree with GPU {rank}: "
                f"{[everyone[r] for r in differ]} vs {identity}"
            )

    def table(self, num_rows: int, row_bytes: int, name: str = "") -> EngramTable:
        """Create a table of `num_rows` rows of `row_bytes` bytes. Collective;
        `name` identifies it across the group."""
        row_stride = round_up(row_bytes, _ROW_ALIGN)
        if row_stride > self.max_row_bytes:
            raise ValueError(f"Rows of {row_bytes} bytes exceed the pool's rows")
        self._check_same(("table", name, num_rows, row_bytes))
        rank, world = self.group.rank_in_group, self.group.world_size
        rows_per_shard = cdiv(num_rows, world)
        row_start = min(rank * rows_per_shard, num_rows)
        shard_rows = min(rows_per_shard, num_rows - row_start)
        num_bytes = rows_per_shard * row_stride
        local, arrays, keep = self.transport.shard(num_bytes, name)
        meta = torch.tensor([rows_per_shard, num_rows, *arrays], dtype=torch.int64)
        shard = local[: shard_rows * row_stride].view(shard_rows, row_stride)
        logger.info(
            "Engram pool table %r: rows [%d, %d) of %d on this GPU, %.2f GiB in %s",
            name,
            row_start,
            row_start + shard_rows,
            num_rows,
            num_bytes / 2**30,
            self._where,
        )
        return EngramTable(
            num_rows, row_bytes, row_stride, row_start, shard[:, :row_bytes], meta, keep
        )

    def ready(self, table: EngramTable) -> None:
        """Serve `table` once every GPU has loaded its shard. Peers read it at
        any time from then on, so it cannot be reloaded."""
        if table._loaded:
            raise RuntimeError(
                "Engram tables split across engines (peer_group_size) cannot "
                "reload weights; restart the peer group instead"
            )
        torch.accelerator.synchronize()
        self.group.barrier()
        table._loaded = True

    def _meta(self, tables: list[EngramTable]) -> torch.Tensor:
        key = tuple(id(t) for t in tables)
        if (meta := self._metas.get(key)) is None:
            if len({t.row_stride for t in tables}) != 1:
                raise ValueError("One fetch's tables must have equal row sizes")
            meta = torch.stack([t._meta for t in tables]).cuda()
            self._metas[key] = meta
        return meta

    def _output(self, numel: int) -> torch.Tensor:
        """Placed rows, allocated once a fetch places any."""
        if self._buffer is None:
            numel_max = self.max_items * self.max_tables * self.max_row_bytes
            self._buffer = torch.empty(numel_max, dtype=torch.uint8, device="cuda")
        return self._buffer[:numel]

    def start(self, tables: list[EngramTable], ids: torch.Tensor) -> EngramFetch:
        """Start fetching `ids` (`[..., L, K]`: rows of table l at `[..., l, :]`)
        on the current stream; ids < 0 or past a table give zero rows."""
        if ids.shape[-2] != len(tables):
            raise ValueError(
                f"ids {tuple(ids.shape)} do not match {len(tables)} tables"
            )
        if len(tables) > self.max_tables or ids.numel() > self.max_items * len(tables):
            raise ValueError(f"Fetching {tuple(ids.shape)} exceeds the pool's capacity")
        meta = self._meta(tables)
        shape = (len(tables), *ids.shape[:-2], ids.shape[-1], tables[0].row_stride)
        fetch = EngramFetch(self, shape)
        fetch.state = self.transport.start(meta, ids.contiguous(), fetch)
        return fetch

    def lookup(self, tables: list[EngramTable], ids: torch.Tensor) -> torch.Tensor:
        """Rows `[L, ..., K, row_stride]` (uint8) of `ids`, as `start` takes."""
        fetch = self.start(tables, ids)
        for index in range(len(tables)):
            fetch.wait(index)
        return fetch.rows

    def close(self) -> None:
        self.transport.close()


@cache
def get_engram_pool(max_items: int, max_row_bytes: int, max_tables: int = 1):
    """This GPU's pool as EngramConfig's `peer_group_*` describe it, shared by
    every table of the process that fits these sizes."""
    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config().engram_config
    assert config is not None and config.peer_group_store is not None
    return EngramPool(
        join_peer_group(config.peer_group_size, config.peer_group_store),
        config.peer_group_transport,
        config.cpu_offload,
        max_items,
        max_row_bytes,
        max_tables,
        config.peer_group_rdma_inflight,
        config.peer_group_fetch,
        config.peer_group_thp_dir,
    )


class EngramPrefetch:
    """Fetches `tables` once per step; `wait(i)` hands table i's rows to
    `consume(fetch, i)` there, or after the fetch on a side stream (async)."""

    def __init__(
        self,
        pool: EngramPool,
        tables: list[EngramTable],
        consume: Callable[[EngramFetch, int], None],
    ) -> None:
        self.pool, self.tables, self.consume = pool, tables, consume
        self._async = pool.fetch_mode == "async"
        self._fetch: EngramFetch | None = None
        if self._async:
            self._stream = torch.cuda.Stream(priority=-1)
            self._done = [torch.cuda.Event() for _ in tables]

    def start(self, ids: torch.Tensor) -> None:
        """Start fetching `ids` (`[..., L, K]`) before the layers run."""
        if self._async:
            self._start_async(ids)
        else:
            self._fetch = self.pool.start(self.tables, ids)

    def wait(self, index: int) -> None:
        """Make table `index`'s rows consumed before the current stream goes on."""
        if self._async:
            self._wait_async(index)
        else:
            assert self._fetch is not None
            self.consume(self._fetch, index)

    @eager_break_during_capture
    def _start_async(self, ids: torch.Tensor) -> None:
        stream = self._stream
        stream.wait_stream(current_stream())
        ids.record_stream(stream)
        with torch.cuda.stream(stream):
            fetch = self.pool.start(self.tables, ids)
            for index in range(len(self.tables)):
                self.consume(fetch, index)
                self._done[index].record(stream)

    @eager_break_during_capture
    def _wait_async(self, index: int) -> None:
        current_stream().wait_event(self._done[index])
