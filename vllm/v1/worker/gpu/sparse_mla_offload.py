# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import math
import mmap
import os
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import regex as re
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator, in_the_same_node_as
from vllm.v1.kv_cache_interface import (
    SparseMLAAllocationManifestEntry,
    SparseMLAAllocationOwner,
    SparseMLAAllocationTier,
    SparseMLAOffloadMemoryPlan,
)

_SCHEMA_VERSION = 1
_BACKING_PATTERN = re.compile(r"vllm_sparse_mla_([0-9a-f]{32})_dp([0-9]+)\.mmap\Z")


def _names(value: str) -> tuple[str, ...]:
    return tuple(value.split())


_PER_LAYER_BUFFERS = frozenset(
    _names(
        "resident_main_kv resident_logical_ids resident_last_access "
        "resident_generation newest_main_kv newest_logical_ids "
        "newest_generation provisional_slots"
    )
)
_LOCAL_BUFFER_NAMES = _names(
    "resident_main_kv resident_logical_ids resident_last_access "
    "resident_generation newest_main_kv newest_logical_ids newest_generation "
    "request_block_ids request_num_blocks request_num_tokens request_generation "
    "request_active topk_logical_ids topk_physical_ids topk_hit_mask "
    "miss_logical_ids miss_victim_slots miss_counts provisional_slots "
    "accepted_counts hit_output hit_lse miss_output miss_lse tp_fence_token"
)
_FailureStatus = tuple[str, int, str, str]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _payload_bytes(entry: SparseMLAAllocationManifestEntry) -> int:
    return math.prod(entry.shape) * torch.empty((), dtype=entry.dtype).element_size()


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _shared_memory_root() -> Path:
    root = Path("/dev/shm")
    return root if root.exists() else Path(tempfile.gettempdir())


def _failure(stage: str, rank: int, error: BaseException) -> _FailureStatus:
    message = " ".join(str(error).splitlines())
    return stage, rank, type(error).__name__, message


def _gather_failure(
    status: _FailureStatus | None, tp_group: GroupCoordinator
) -> _FailureStatus | None:
    if tp_group.world_size == 1:
        return status
    statuses: list[_FailureStatus | None] = [None] * tp_group.world_size
    dist.all_gather_object(statuses, status, group=tp_group.cpu_group)
    failures = [item for item in statuses if item is not None]
    return min(failures, key=lambda item: item[1]) if failures else None


def _format_failure(status: _FailureStatus) -> str:
    stage, rank, error_type, message = status
    return (
        f"Sparse MLA offload {stage} failed on TP rank {rank}: {error_type}: {message}"
    )


@dataclass(frozen=True, slots=True)
class SparseMLAPoolHandle:
    schema_version: int
    backing_name: str
    byte_length: int
    layer_names: tuple[str, ...]
    layer_offsets: tuple[int, ...]
    layer_shapes: tuple[tuple[int, ...], ...]
    layer_dtypes: tuple[str, ...]
    dp_replica_id: int
    tp_global_ranks: tuple[int, ...]
    creation_nonce: str


@dataclass(frozen=True, slots=True)
class SparseMLALayerView:
    layer_name: str
    layer_index: int
    is_host_writer: bool
    main_host_kv: torch.Tensor
    local_buffers: Mapping[str, torch.Tensor]
    side_stream: torch.cuda.Stream | None
    fork_ready_events: tuple[torch.cuda.Event | None, ...]
    miss_ready_events: tuple[torch.cuda.Event | None, ...]


class SparseMLAOffloadManager:
    _plan: SparseMLAOffloadMemoryPlan
    _tp_group: GroupCoordinator
    _is_host_writer: bool
    _created_backing = _registered = _closing = _closed = _unlinked = False
    _fd: int | None
    _mmap: mmap.mmap | None
    _host_views: dict[str, torch.Tensor]
    _local_buffers: dict[str, torch.Tensor]
    _indexer_inventory: dict[str, torch.Tensor]
    _layer_views: dict[str, SparseMLALayerView]
    _device_slab: torch.Tensor | None
    _side_stream: torch.cuda.Stream | None
    _fork_ready_events: tuple[tuple[torch.cuda.Event | None, ...], ...]
    _miss_ready_events: tuple[tuple[torch.cuda.Event | None, ...], ...]
    _pool_handle: SparseMLAPoolHandle
    _backing_path: Path
    _host_base_ptr = 0
    _host_entry: SparseMLAAllocationManifestEntry
    _indexer_entry: SparseMLAAllocationManifestEntry
    _local_layout: tuple[tuple[SparseMLAAllocationManifestEntry, int], ...]
    _layer_offsets: tuple[int, ...]
    _dp_replica_id = 0

    @classmethod
    def create_for_tp_group(
        cls,
        plan: SparseMLAOffloadMemoryPlan,
        tp_group: GroupCoordinator,
        indexer_inventory: Mapping[str, torch.Tensor],
    ) -> "SparseMLAOffloadManager":
        manager = cls.__new__(cls)
        preflight_error = None
        try:
            manager._validate_and_layout(plan, tp_group, indexer_inventory)
        except Exception as error:
            preflight_error = _failure("preflight", tp_group.rank_in_group, error)
        canonical = _gather_failure(preflight_error, tp_group)
        if canonical is not None:
            raise RuntimeError(_format_failure(canonical))

        same_host_error = None
        try:
            if not all(in_the_same_node_as(tp_group.cpu_group, source_rank=0)):
                raise RuntimeError("tensor-parallel ranks are not on one host")
        except Exception as error:
            same_host_error = _failure("same-host proof", tp_group.rank_in_group, error)
        canonical = _gather_failure(same_host_error, tp_group)
        if canonical is not None:
            raise RuntimeError(_format_failure(canonical))

        envelope: tuple[SparseMLAPoolHandle | None, _FailureStatus | None]
        if tp_group.rank_in_group == 0:
            try:
                manager._initialize_host_mapping(None)
                envelope = manager.pool_handle, None
            except Exception as error:
                envelope = (
                    None,
                    _failure("creator mapping", tp_group.rank_in_group, error),
                )
        else:
            envelope = None, None
        envelope_list = [envelope]
        dist.broadcast_object_list(
            envelope_list,
            src=tp_group.ranks[0],
            group=tp_group.cpu_group,
        )
        handle, creator_error = envelope_list[0]

        local_error = creator_error
        if creator_error is None:
            try:
                assert handle is not None
                if tp_group.rank_in_group != 0:
                    manager._initialize_host_mapping(handle)
                manager._initialize_local_resources(indexer_inventory)
                manager._register_host_mapping()
                manager._build_layer_views()
            except Exception as error:
                local_error = _failure(
                    "local initialization", tp_group.rank_in_group, error
                )
        canonical = _gather_failure(local_error, tp_group)
        if canonical is None:
            dist.barrier(group=tp_group.cpu_group)
            return manager

        cleanup_error = manager._close_local()
        cleanup_failure = _gather_failure(cleanup_error, tp_group)
        try:
            dist.barrier(group=tp_group.cpu_group)
        finally:
            unlink_error = None
            if tp_group.rank_in_group == 0 and cleanup_failure is None:
                try:
                    manager.unlink()
                except Exception as error:
                    unlink_error = _failure(
                        "factory unlink", tp_group.rank_in_group, error
                    )
            unlink_list = [unlink_error]
            dist.broadcast_object_list(
                unlink_list,
                src=tp_group.ranks[0],
                group=tp_group.cpu_group,
            )
        details = _format_failure(canonical)
        if cleanup_failure is not None:
            details += f"; cleanup: {_format_failure(cleanup_failure)}"
        if unlink_list[0] is not None:
            details += f"; unlink: {_format_failure(unlink_list[0])}"
        raise RuntimeError(details)

    @property
    def pool_handle(self) -> SparseMLAPoolHandle:
        return self._pool_handle

    def indexer_kv(self, layer_name: str) -> torch.Tensor:
        self._require_open()
        return self._indexer_inventory[layer_name]

    def layer_view(self, layer_name: str) -> SparseMLALayerView:
        self._require_open()
        return self._layer_views[layer_name]

    def main_host_write_view(self, layer_name: str) -> torch.Tensor:
        self._require_open()
        if not self._is_host_writer:
            raise PermissionError("only TP-local rank 0 may write shared Main KV")
        return self._host_views[layer_name]

    def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        first_error: BaseException | None = None
        if self._side_stream is not None:
            try:
                self._side_stream.synchronize()
            except Exception as error:
                first_error = error
        if self._registered:
            result = torch.cuda.cudart().cudaHostUnregister(self._host_base_ptr)
            if result.value != 0:
                first_error = first_error or RuntimeError(
                    f"cudaHostUnregister failed with code {result.value}"
                )
            else:
                self._registered = False

        self._fork_ready_events = self._miss_ready_events = ()
        self._side_stream = None
        self._local_buffers = self._indexer_inventory = {}
        self._device_slab = None
        if not self._registered:
            self._layer_views = self._host_views = {}
            gc.collect()
            if self._mmap is not None:
                try:
                    self._mmap.close()
                except (BufferError, OSError) as error:
                    first_error = first_error or error
                else:
                    self._mmap = None
            if self._mmap is None and self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError as error:
                    first_error = first_error or error
                else:
                    self._fd = None
        if first_error is not None:
            raise first_error
        self._closed = True

    def unlink(self) -> None:
        if not self._is_host_writer:
            raise PermissionError("only TP-local rank 0 may unlink shared Main KV")
        if not self._closed:
            raise RuntimeError("close must succeed before unlink")
        if self._unlinked:
            return
        if not self._created_backing:
            self._unlinked = True
            return
        with suppress(FileNotFoundError):
            os.unlink(self._backing_path)
        self._unlinked = True

    def _validate_and_layout(
        self,
        plan: SparseMLAOffloadMemoryPlan,
        tp_group: GroupCoordinator,
        indexer_inventory: Mapping[str, torch.Tensor],
    ) -> None:
        self._plan = plan
        self._tp_group = tp_group
        self._is_host_writer = tp_group.rank_in_group == 0
        self._created_backing = self._registered = False
        self._closing = self._closed = self._unlinked = False
        self._fd = self._mmap = None
        self._host_views = self._local_buffers = {}
        self._indexer_inventory = self._layer_views = {}
        self._device_slab = self._side_stream = None
        self._fork_ready_events = self._miss_ready_events = ()

        rows = [tuple(row) for row in tp_group.group_ranks]
        ranks = tuple(tp_group.ranks)
        if (
            tp_group.world_size != plan.tensor_parallel_size
            or len(rows) != plan.num_dp_replicas
            or rows.count(ranks) != 1
            or not 0 <= tp_group.rank_in_group < len(ranks)
            or ranks[tp_group.rank_in_group] != tp_group.rank
        ):
            raise RuntimeError("unsupported sparse MLA distributed topology")
        self._dp_replica_id = rows.index(ranks)

        entries = plan.manifest
        names = [entry.name for entry in entries]
        expected_names = ["main_host_kv", "indexer_kv", *_LOCAL_BUFFER_NAMES]
        if names != expected_names or len(set(names)) != len(names):
            raise ValueError("invalid sparse MLA allocation manifest names")
        for entry in entries:
            if (
                not entry.shape
                or any(dimension <= 0 for dimension in entry.shape)
                or entry.alignment_bytes <= 0
                or entry.alignment_bytes & (entry.alignment_bytes - 1)
                or entry.allocation_count <= 0
            ):
                raise ValueError("invalid sparse MLA manifest dimensions")
        host_entry, indexer_entry, *local_entries = entries
        if len(host_entry.shape) != 3:
            raise ValueError("invalid sparse MLA manifest dimensions")
        if (
            host_entry.owner is not SparseMLAAllocationOwner.OFFLOAD_MANAGER
            or host_entry.tier is not SparseMLAAllocationTier.HOST_SHARED
            or host_entry.alignment_bytes != 4096
            or host_entry.allocation_count != len(plan.main_layer_names)
            or host_entry.shape[0] != plan.num_blocks
        ):
            raise ValueError("invalid shared Host manifest entry")
        if (
            indexer_entry.owner is not SparseMLAAllocationOwner.GENERIC_KV_ALLOCATOR
            or indexer_entry.tier is not SparseMLAAllocationTier.HBM_LOCAL
            or indexer_entry.allocation_count != len(plan.indexer_layer_names)
        ):
            raise ValueError("invalid Indexer manifest entry")
        if set(indexer_inventory) != set(plan.indexer_layer_names):
            raise ValueError("Indexer inventory names do not match the plan")
        intervals: list[tuple[int, int]] = []
        for name in plan.indexer_layer_names:
            tensor = indexer_inventory[name]
            if (
                tensor.dtype != indexer_entry.dtype
                or tensor.device != tp_group.device
                or not tensor.is_contiguous()
                or tensor.numel() * tensor.element_size()
                != _payload_bytes(indexer_entry)
            ):
                raise ValueError("invalid Indexer tensor")
            start = tensor.data_ptr()
            interval = start, start + tensor.numel() * tensor.element_size()
            if any(
                start < end and interval[1] > previous for previous, end in intervals
            ):
                raise ValueError("Indexer tensor storage intervals overlap")
            intervals.append(interval)

        self._host_entry = host_entry
        payload = _payload_bytes(host_entry)
        offsets = tuple(
            index * _round_up(payload, 4096)
            for index in range(len(plan.main_layer_names))
        )
        if offsets[-1] + _round_up(payload, 4096) > plan.host_pool_bytes_per_dp_replica:
            raise ValueError("shared Host payload exceeds the planned pool")
        self._layer_offsets = offsets

        cursor = 0
        local_layout = []
        for entry in local_entries:
            if (
                entry.owner is not SparseMLAAllocationOwner.OFFLOAD_MANAGER
                or entry.tier is not SparseMLAAllocationTier.HBM_LOCAL
                or entry.allocation_count != 1
            ):
                raise ValueError("invalid Manager-local manifest entry")
            offset = _round_up(cursor, entry.alignment_bytes)
            cursor = offset + _round_up(_payload_bytes(entry), entry.alignment_bytes)
            local_layout.append((entry, offset))
        aligned_payload_sum = sum(
            _round_up(_payload_bytes(entry), entry.alignment_bytes)
            for entry in local_entries
        )
        if (
            cursor != plan.fixed_offload_hbm_bytes_per_tp_rank
            or aligned_payload_sum != cursor
        ):
            raise ValueError("fixed HBM manifest does not equal the planned reserve")
        indexer_bytes = len(plan.indexer_layer_names) * _payload_bytes(indexer_entry)
        if plan.device_bytes_per_tp_rank != cursor + indexer_bytes:
            raise ValueError("device byte accounting does not match the manifest")
        self._indexer_entry = indexer_entry
        self._local_layout = tuple(local_layout)

    def _initialize_local_resources(
        self, indexer_inventory: Mapping[str, torch.Tensor]
    ) -> None:
        if self._device_slab is not None:
            raise RuntimeError("local resources are already initialized")
        self._indexer_inventory = {
            name: indexer_inventory[name].view(self._indexer_entry.shape)
            for name in self._plan.indexer_layer_names
        }
        slab = torch.empty(
            self._plan.fixed_offload_hbm_bytes_per_tp_rank,
            dtype=torch.uint8,
            device=self._tp_group.device,
        )
        local_buffers = {}
        for entry, offset in self._local_layout:
            payload = _payload_bytes(entry)
            if (
                self._tp_group.device.type == "cuda"
                and (slab.data_ptr() + offset) % entry.alignment_bytes
            ):
                raise RuntimeError("CUDA local buffer alignment is invalid")
            local_buffers[entry.name] = (
                slab[offset : offset + payload].view(entry.dtype).view(entry.shape)
            )
        self._device_slab = slab
        self._local_buffers = local_buffers
        main_count = len(self._plan.main_layer_names)
        newest_width = local_buffers["newest_main_kv"].shape[2]
        if self._tp_group.device.type == "cuda":
            self._side_stream = torch.cuda.Stream(device=self._tp_group.device)
            self._fork_ready_events = tuple(
                tuple(
                    torch.cuda.Event(enable_timing=False) for _ in range(newest_width)
                )
                for _ in range(main_count)
            )
            self._miss_ready_events = tuple(
                tuple(
                    torch.cuda.Event(enable_timing=False) for _ in range(newest_width)
                )
                for _ in range(main_count)
            )
        else:
            empty_row = (None,) * newest_width
            self._fork_ready_events = tuple(empty_row for _ in range(main_count))
            self._miss_ready_events = tuple(empty_row for _ in range(main_count))

    def _initialize_host_mapping(self, handle: SparseMLAPoolHandle | None) -> None:
        if self._mmap is not None or self._fd is not None:
            raise RuntimeError("Host mapping is already initialized")
        if handle is None:
            nonce = uuid.uuid4().hex
            backing_name = f"vllm_sparse_mla_{nonce}_dp{self._dp_replica_id}.mmap"
            handle = SparseMLAPoolHandle(
                schema_version=_SCHEMA_VERSION,
                backing_name=backing_name,
                byte_length=self._plan.host_pool_bytes_per_dp_replica,
                layer_names=self._plan.main_layer_names,
                layer_offsets=self._layer_offsets,
                layer_shapes=(self._host_entry.shape,)
                * len(self._plan.main_layer_names),
                layer_dtypes=(_dtype_name(self._host_entry.dtype),)
                * len(self._plan.main_layer_names),
                dp_replica_id=self._dp_replica_id,
                tp_global_ranks=tuple(self._tp_group.ranks),
                creation_nonce=nonce,
            )
        match = _BACKING_PATTERN.fullmatch(handle.backing_name)
        expected_shapes = (self._host_entry.shape,) * len(self._plan.main_layer_names)
        expected_dtypes = (_dtype_name(self._host_entry.dtype),) * len(
            self._plan.main_layer_names
        )
        if (
            match is None
            or match.group(1) != handle.creation_nonce
            or int(match.group(2)) != self._dp_replica_id
            or handle.schema_version != _SCHEMA_VERSION
            or handle.byte_length != self._plan.host_pool_bytes_per_dp_replica
            or handle.layer_names != self._plan.main_layer_names
            or handle.layer_offsets != self._layer_offsets
            or handle.layer_shapes != expected_shapes
            or handle.layer_dtypes != expected_dtypes
            or handle.dp_replica_id != self._dp_replica_id
            or handle.tp_global_ranks != tuple(self._tp_group.ranks)
        ):
            raise ValueError("shared Host pool handle does not match the local plan")
        path = _shared_memory_root() / handle.backing_name
        self._backing_path = path
        if self._is_host_writer:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            self._created_backing = True
            try:
                os.ftruncate(fd, handle.byte_length)
            except OSError:
                os.close(fd)
                raise
        else:
            fd = os.open(path, os.O_RDWR)
            if os.fstat(fd).st_size != handle.byte_length:
                os.close(fd)
                raise ValueError("shared Host backing size does not match the handle")
        try:
            mapping = mmap.mmap(
                fd,
                handle.byte_length,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except (OSError, ValueError):
            os.close(fd)
            raise
        self._pool_handle = handle
        self._fd = fd
        self._mmap = mapping
        self._host_views = {
            name: torch.frombuffer(
                mapping,
                dtype=self._host_entry.dtype,
                count=math.prod(self._host_entry.shape),
                offset=offset,
            ).view(self._host_entry.shape)
            for name, offset in zip(handle.layer_names, handle.layer_offsets)
        }
        self._host_base_ptr = self._host_views[handle.layer_names[0]].data_ptr()

    def _register_host_mapping(self) -> None:
        if self._tp_group.device.type != "cuda":
            return
        if self._registered:
            raise RuntimeError("Host mapping is already registered")
        result = torch.cuda.cudart().cudaHostRegister(
            self._host_base_ptr, self._pool_handle.byte_length, 0
        )
        if result.value != 0:
            raise RuntimeError(f"cudaHostRegister failed with code {result.value}")
        self._registered = True

    def _build_layer_views(self) -> None:
        views = {}
        for index, name in enumerate(self._plan.main_layer_names):
            buffers = {
                key: value[index] if key in _PER_LAYER_BUFFERS else value
                for key, value in self._local_buffers.items()
            }
            views[name] = SparseMLALayerView(
                layer_name=name,
                layer_index=index,
                is_host_writer=self._is_host_writer,
                main_host_kv=self._host_views[name],
                local_buffers=MappingProxyType(buffers),
                side_stream=self._side_stream,
                fork_ready_events=self._fork_ready_events[index],
                miss_ready_events=self._miss_ready_events[index],
            )
        self._layer_views = views

    def _require_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("Sparse MLA offload Manager is closing")

    def _close_local(self) -> _FailureStatus | None:
        try:
            self.close()
        except Exception as error:
            return _failure("local cleanup", self._tp_group.rank_in_group, error)
        return None
