# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish PP ownership epochs and transfer each changed layer's cache backing."""

import json
import time
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from vllm.v1.worker.pp.boundary import PPBoundary

CONTROL_KEY_PREFIX = "vllm_dynamic_pp_control/"
STATE_TRANSFER_CHUNK_BYTES = 256 * 1024 * 1024


class PPControlChannel:
    """One producer and consumer on TP0 of the same PP CPU-group store.

    The producer must acquire its bounded activation-send slot before send().
    The consumer reads the header before posting that batch's activation receive,
    so completed activation sends also bound unconsumed control keys. Each new
    group store starts at batch zero; successful receives delete their keys.
    """

    def __init__(self, store: Any):
        self.store = store
        self._sent: PPBoundary | None = None
        self._received: PPBoundary | None = None

    @staticmethod
    def _validate(header: PPBoundary, previous: PPBoundary | None) -> None:
        if not isinstance(header, PPBoundary):
            raise ValueError("PP control requires a boundary header")
        PPBoundary.from_wire(header.to_wire())
        expected_batch = 0 if previous is None else previous.batch_id + 1
        if header.batch_id != expected_batch or header.boundary <= 0:
            raise ValueError("PP control batch is out of order or boundary is invalid")
        if previous is not None and (
            header.version < previous.version
            or header.version == previous.version
            and header.boundary != previous.boundary
        ):
            raise ValueError("PP control boundary version is inconsistent")

    def send(self, header: PPBoundary) -> None:
        """Publish one immutable batch header without waiting for its consumer."""
        self._validate(header, self._sent)
        key = f"{CONTROL_KEY_PREFIX}{header.batch_id}"
        if self.store.check([key]):
            raise ValueError("PP control batch header was already published")
        self.store.set(key, json.dumps(header.to_wire()))
        self._sent = header

    def receive(self, expected_batch_id: int) -> PPBoundary:
        """Read the next batch's header, validate its epoch, then remove its key."""
        next_batch = 0 if self._received is None else self._received.batch_id + 1
        if type(expected_batch_id) is not int or expected_batch_id != next_batch:
            raise ValueError("PP control expected batch is out of order")
        key = f"{CONTROL_KEY_PREFIX}{expected_batch_id}"
        header = PPBoundary.from_wire(json.loads(self.store.get(key)))
        if header.batch_id != expected_batch_id:
            raise ValueError("PP control header does not match the expected batch")
        self._validate(header, self._received)
        if not self.store.delete_key(key):
            raise ValueError("PP control batch header disappeared before consumption")
        self._received = header
        return header


class PPStateTransfer:
    """Both stages must call move with the same CPU-agreed boundary header.

    The adapter must expose each component's entire dedicated byte storage,
    including allocator padding. It must keep these allocations stable for the
    lifetime of this object. Unchanged boundaries perform no device work.
    """

    def __init__(
        self,
        pp_group: Any,
        device_module: Any,
        state: Mapping,
        *,
        max_chunk_bytes: int = STATE_TRANSFER_CHUNK_BYTES,
        cache_layout: Mapping | None = None,
    ):
        if pp_group.world_size != 2 or pp_group.rank_in_group not in (0, 1):
            raise ValueError("PP state transfer requires a two-stage PP group")
        self.pp = pp_group
        self.device_module = device_module
        self.state = state
        self._max_chunk_bytes = max_chunk_bytes
        self._boundary: int | None = None
        self._last_batch = -1
        self._moves = 0
        self._bytes = 0
        self._chunks = 0
        self._seconds = 0.0
        self._last_move: dict[str, Any] | None = None
        description, storage, error = self._inspect()
        peers = self._gather(
            {
                "description": description,
                "chunk_bytes": max_chunk_bytes,
                "cache_layout": cache_layout,
                "error": error,
            }
        )
        self._raise_peer_errors(peers)
        if (
            peers[0]["description"] != peers[1]["description"]
            or peers[0]["chunk_bytes"] != peers[1]["chunk_bytes"]
            or peers[0]["cache_layout"] != peers[1]["cache_layout"]
        ):
            raise ValueError(
                "PP cache layer/component shape, dtype, bytes, or block layout differ"
            )
        self._description = description
        self._storage = storage

    def _gather(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        peers: list[Any] = [None, None]
        dist.all_gather_object(peers, value, group=self.pp.cpu_group)
        return peers

    @staticmethod
    def _raise_peer_errors(peers: list[dict[str, Any]]) -> None:
        errors = [
            f"PP{rank}: {peer['error']}"
            for rank, peer in enumerate(peers)
            if peer["error"] is not None
        ]
        if errors:
            raise ValueError("PP cache state validation failed: " + "; ".join(errors))

    def _inspect(self) -> tuple[list[Any], list[Any], str | None]:
        description, storage = [], []
        try:
            if (
                type(self._max_chunk_bytes) is not int
                or not 0 < self._max_chunk_bytes <= STATE_TRANSFER_CHUNK_BYTES
            ):
                raise ValueError("PP cache transfer chunk size must be in [1, 256 MiB]")
            if not callable(getattr(self.device_module, "synchronize", None)):
                raise ValueError("Device module has no synchronize operation")
            if not isinstance(self.state, Mapping) or not self.state:
                raise ValueError("Cache state must be a nonempty layer mapping")
            if any(type(layer) is not int or layer < 0 for layer in self.state):
                raise ValueError("Cache layer keys must be nonnegative integers")
            allocations: set[tuple[str, int]] = set()
            device = None
            for layer in sorted(self.state):
                components = self.state[layer]
                if not isinstance(components, Mapping) or not components:
                    raise ValueError(f"Layer {layer} has no cache component mapping")
                if any(type(key) is not str or not key for key in components):
                    raise ValueError("Cache component keys must be nonempty strings")
                for key in sorted(components):
                    raw = components[key]
                    if not isinstance(raw, torch.Tensor):
                        raise TypeError(f"Cache {layer}/{key} is not a tensor")
                    if raw.dtype != torch.int8 or raw.ndim != 1:
                        raise ValueError(
                            f"Cache {layer}/{key} must be 1D int8 raw backing"
                        )
                    backing = raw.untyped_storage()
                    if (
                        raw.layout != torch.strided
                        or not raw.is_contiguous()
                        or raw.numel() == 0
                        or raw.storage_offset() != 0
                        or raw.nbytes != backing.nbytes()
                    ):
                        raise ValueError(
                            f"Cache {layer}/{key} is not a complete raw allocation"
                        )
                    if device is None:
                        device = raw.device
                    if raw.device != device:
                        raise ValueError("Local cache components must share one device")
                    allocation = (str(raw.device), backing.data_ptr())
                    if allocation in allocations:
                        raise ValueError(
                            f"Cache {layer}/{key} aliases another component"
                        )
                    allocations.add(allocation)
                    description.append(
                        (
                            layer,
                            key,
                            tuple(raw.shape),
                            str(raw.dtype),
                            raw.nbytes,
                            raw.device.type,
                        )
                    )
                    storage.append((layer, key, allocation, backing.nbytes()))
            return description, storage, None
        except (TypeError, ValueError, RuntimeError) as exc:
            return [], [], str(exc)

    def move(self, oldcut: int, newcut: int, batch_id: int) -> None:
        """Quiesce both stages, then copy each changed layer from its old owner."""
        valid_numbers = all(
            type(value) is int and value >= 0 for value in (oldcut, newcut, batch_id)
        )
        if valid_numbers and oldcut == newcut:
            if (
                not valid_numbers
                or batch_id <= self._last_batch
                or self._boundary is not None
                and self._boundary != oldcut
            ):
                raise ValueError("Invalid unchanged PP cache boundary/batch")
            self._boundary, self._last_batch = newcut, batch_id
            return

        started = time.monotonic()
        description, storage, error = self._inspect()
        layers = (
            list(range(min(oldcut, newcut), max(oldcut, newcut)))
            if valid_numbers
            else []
        )
        if error is None and (
            not valid_numbers
            or batch_id <= self._last_batch
            or self._boundary is not None
            and self._boundary != oldcut
        ):
            error = "Invalid changed PP cache boundary/batch"
        if error is None and (
            description != self._description or storage != self._storage
        ):
            error = "Cache allocations changed after validation"
        if error is None and any(layer not in self.state for layer in layers):
            error = "Changed boundary crosses a layer without replicated cache state"
        peers = self._gather(
            {
                "oldcut": oldcut if type(oldcut) is int else None,
                "newcut": newcut if type(newcut) is int else None,
                "batch_id": batch_id if type(batch_id) is int else None,
                "error": error,
            }
        )
        self._raise_peer_errors(peers)
        if any(
            (peer["oldcut"], peer["newcut"], peer["batch_id"])
            != (oldcut, newcut, batch_id)
            for peer in peers
        ):
            raise ValueError("PP cache handoff boundary/batch differs between stages")

        error = None
        try:
            self.device_module.synchronize()
        except RuntimeError as exc:
            error = str(exc)
        self._raise_peer_errors(self._gather({"error": error}))

        moved_bytes = 0
        chunks = 0
        for layer in layers:
            owner = 0 if layer < oldcut else 1
            for key in sorted(self.state[layer]):
                raw = self.state[layer][key]
                for start in range(0, raw.numel(), self._max_chunk_bytes):
                    self.pp.broadcast(
                        raw[start : start + self._max_chunk_bytes], src=owner
                    )
                    chunks += 1
                moved_bytes += raw.nbytes
        self.device_module.synchronize()
        elapsed = time.monotonic() - started
        self._boundary, self._last_batch = newcut, batch_id
        self._moves += 1
        self._bytes += moved_bytes
        self._chunks += chunks
        self._seconds += elapsed
        self._last_move = {
            "batch_id": batch_id,
            "oldcut": oldcut,
            "newcut": newcut,
            "layers": layers,
            "bytes": moved_bytes,
            "chunks": chunks,
            "seconds": elapsed,
        }

    def stats(self) -> dict[str, Any]:
        last_move = (
            self._last_move | {"layers": list(self._last_move["layers"])}
            if self._last_move is not None
            else None
        )
        return {
            "moves": self._moves,
            "bytes": self._bytes,
            "chunks": self._chunks,
            "seconds": self._seconds,
            "last_move": last_move,
        }
