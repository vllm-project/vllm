# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake-backed immutable row storage shared by Engram model families."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _load_mooncake_config(path: str) -> dict[str, Any]:
    """Load the deployment options accepted by Mooncake EngramStore."""
    try:
        options = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Mooncake Engram config {path!r}") from exc
    if not isinstance(options, dict):
        raise ValueError("Mooncake Engram config must be a JSON object")

    allowed = {
        "local_dir",
        "shard_id",
        "num_shards",
        "master_addr",
        "session_id",
        "caller_id",
        "max_lookup_rows",
        "rdma_device",
        "timeout",
        "num_callers",
    }
    if unknown := options.keys() - allowed:
        raise ValueError(f"Unknown Mooncake Engram options: {sorted(unknown)}")
    required = {
        "local_dir",
        "shard_id",
        "num_shards",
        "master_addr",
        "session_id",
        "max_lookup_rows",
        "num_callers",
    }
    if missing := required - options.keys():
        raise ValueError(f"Missing Mooncake Engram options: {sorted(missing)}")
    if not isinstance(options["local_dir"], str) or not options["local_dir"]:
        raise ValueError("Mooncake Engram local_dir must be a nonempty string")
    for name in ("shard_id", "num_shards", "max_lookup_rows", "num_callers"):
        if type(options[name]) is not int:
            raise ValueError(f"Mooncake Engram {name} must be an integer")
    if (
        not 0 <= options["shard_id"] < options["num_shards"]
        or options["max_lookup_rows"] < 1
        or options["num_callers"] < 1
    ):
        raise ValueError("Invalid Mooncake Engram node shard or lookup capacity")
    for name in ("master_addr", "session_id"):
        if not isinstance(options[name], str) or not options[name]:
            raise ValueError(f"Mooncake Engram {name} must be a nonempty string")
    if "caller_id" in options and (
        type(options["caller_id"]) is not int
        or not 0 <= options["caller_id"] < options["num_callers"]
    ):
        raise ValueError("Mooncake Engram caller_id must be in [0,num_callers)")
    if "rdma_device" in options and (
        not isinstance(options["rdma_device"], str) or not options["rdma_device"]
    ):
        raise ValueError("Mooncake Engram rdma_device must be a nonempty string")
    if "timeout" in options and (
        type(options["timeout"]) not in (int, float) or options["timeout"] <= 0
    ):
        raise ValueError("Mooncake Engram timeout must be a positive number")
    return options


def _head_shard(
    head_sizes: tuple[int, ...], num_shards: int, shard_id: int
) -> tuple[int, int, int, int]:
    """Return owned head and concatenated-row bounds for one storage node."""
    if not head_sizes or any(size <= 0 for size in head_sizes):
        raise ValueError("Mooncake Engram requires positive head sizes")
    if num_shards > len(head_sizes):
        raise ValueError("More Mooncake node shards than Engram hash heads")
    head_begin = len(head_sizes) * shard_id // num_shards
    head_end = len(head_sizes) * (shard_id + 1) // num_shards
    row_begin = sum(head_sizes[:head_begin])
    row_end = sum(head_sizes[:head_end])
    return head_begin, head_end, row_begin, row_end


def pack_mooncake_rows(*planes: torch.Tensor) -> torch.Tensor:
    """Pack same-row CPU tensors into EngramStore's byte-row representation."""
    if not planes:
        raise ValueError("At least one Engram row plane is required")
    rows = planes[0].shape[0]
    if any(plane.device.type != "cpu" or plane.shape[0] != rows for plane in planes):
        raise ValueError("Engram row planes must be same-length CPU tensors")
    byte_planes = [
        plane.detach().contiguous().view(torch.uint8).reshape(rows, -1)
        for plane in planes
    ]
    return byte_planes[0] if len(byte_planes) == 1 else torch.cat(byte_planes, dim=1)


class MooncakeEngramStorage:
    """One model layer's node-sharded EngramStore deployment."""

    def __init__(
        self,
        config_path: str,
        *,
        namespace: str,
        head_sizes: tuple[int, ...],
        row_bytes: int,
    ) -> None:
        if not namespace:
            raise ValueError("Mooncake Engram namespace must be nonempty")
        if row_bytes <= 0:
            raise ValueError("Mooncake Engram row_bytes must be positive")
        self.options = _load_mooncake_config(config_path)
        self.head_sizes = head_sizes
        self.row_bytes = row_bytes
        self.head_begin, self.head_end, self.row_begin, self.row_end = _head_shard(
            head_sizes,
            self.options["num_shards"],
            self.options["shard_id"],
        )
        self.names = tuple(f"head{index}" for index in range(len(head_sizes)))
        num_shards = self.options["num_shards"]
        num_heads = len(head_sizes)
        self.owners = {
            name: ((index + 1) * num_shards - 1) // num_heads
            for index, name in enumerate(self.names)
        }
        self.offsets = tuple(
            sum(head_sizes[:index]) for index in range(len(head_sizes))
        )
        self.namespace = namespace
        self._store: Any | None = None

    @property
    def local_rows(self) -> int:
        return self.row_end - self.row_begin

    def initialize(self, rows: torch.Tensor) -> None:
        """Publish this node's owned heads and connect every storage shard."""
        if self._store is not None:
            raise RuntimeError("Mooncake Engram storage is already initialized")
        if (
            rows.device.type != "cpu"
            or rows.dtype != torch.uint8
            or rows.shape != (self.local_rows, self.row_bytes)
            or not rows.is_contiguous()
        ):
            raise ValueError(
                "Mooncake Engram initialization requires contiguous CPU uint8 "
                f"[{self.local_rows},{self.row_bytes}] rows"
            )

        try:
            from mooncake.engram import EngramStore
        except ImportError as exc:
            raise RuntimeError(
                "Mooncake Engram requires the node-sharded API from Mooncake #5129"
            ) from exc
        if not hasattr(EngramStore, "populate_shard_and_sync"):
            raise RuntimeError(
                "Mooncake Engram requires populate_shard_and_sync from Mooncake #5129"
            )

        from vllm.distributed import get_world_group

        local_rank = get_world_group().local_rank
        remote = {
            key: value for key, value in self.options.items() if key != "local_dir"
        }
        remote.setdefault("caller_id", local_rank)
        caller_id = remote["caller_id"]
        if not 0 <= caller_id < remote["num_callers"]:
            raise ValueError(
                "Mooncake Engram caller_id must identify a local worker in "
                f"[0,{remote['num_callers']}), got {caller_id}"
            )
        if rdma_device := remote.get("rdma_device"):
            remote["rdma_device"] = rdma_device.format(local_rank=local_rank)

        store = EngramStore(
            tables={
                name: {"num_rows": size, "row_bytes": self.row_bytes}
                for name, size in zip(self.names, self.head_sizes)
            },
            namespace=self.namespace,
            local_dir=self.options["local_dir"],
            persistent=False,
        )
        local_tables = {}
        for index in range(self.head_begin, self.head_end):
            start = self.offsets[index] - self.row_begin
            end = start + self.head_sizes[index]
            local_tables[self.names[index]] = rows[start:end]
        try:
            store.populate_shard_and_sync(
                local_tables,
                table_owners=self.owners,
                **remote,
            )
        except BaseException:
            store.close()
            raise
        self._store = store

    def lookup(
        self,
        ids: torch.Tensor,
        output: torch.Tensor,
        *,
        head_start: int = 0,
    ) -> None:
        """Enqueue a token-major lookup for a contiguous compute-head shard."""
        if self._store is None:
            raise RuntimeError("Initialize Mooncake Engram storage before lookup")
        if (
            ids.ndim != 2
            or ids.shape[1] != len(self.names)
            or ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_cuda
        ):
            raise ValueError(
                "Mooncake Engram IDs must be CUDA int32/int64 [tokens,heads]"
            )
        if not 0 <= head_start < len(self.names):
            raise ValueError("Mooncake Engram head_start is out of range")
        if output.ndim != 3:
            raise ValueError("Mooncake Engram output must be a 3D byte tensor")
        active_heads = min(output.shape[1], len(self.names) - head_start)
        expected = (ids.shape[0], output.shape[1], self.row_bytes)
        if (
            output.dtype != torch.uint8
            or output.shape != expected
            or not output.is_cuda
        ):
            raise ValueError(f"Mooncake Engram output must be CUDA uint8 {expected}")
        if ids.shape[0] and active_heads:
            head_end = head_start + active_heads
            self._store.lookup(
                self.names[head_start:head_end],
                ids[:, head_start:head_end],
                output[:, :active_heads],
                offsets=self.offsets[head_start:head_end],
                stream=torch.cuda.current_stream(ids.device).cuda_stream,
            )

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def collect_mooncake_engram_storages(
    model: torch.nn.Module,
) -> tuple[MooncakeEngramStorage, ...]:
    """Collect live stores in a deterministic cross-worker shutdown order."""
    storages: dict[int, MooncakeEngramStorage] = {}
    for module in model.modules():
        storage = getattr(module, "mooncake_storage", None)
        if isinstance(storage, MooncakeEngramStorage):
            storages[id(storage)] = storage
    return tuple(sorted(storages.values(), key=lambda storage: storage.namespace))


def close_mooncake_engram_storages(
    storages: tuple[MooncakeEngramStorage, ...],
) -> None:
    """Close every store after model CUDA graphs have been released."""
    for storage in storages:
        try:
            storage.close()
        except Exception:
            logger.exception(
                "Failed to close Mooncake Engram storage %s during shutdown",
                storage.namespace,
            )
