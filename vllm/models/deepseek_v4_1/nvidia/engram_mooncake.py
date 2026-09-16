# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched Mooncake RDMA lookup for DeepSeek V4.1 Engram rows."""

from __future__ import annotations

import json
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import triton
import triton.language as tl

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.logger import init_logger
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from .engram import ParallelEngramEmbedding

logger = init_logger(__name__)


def _gpudirect_flush_required() -> bool:
    error, device = cudart.cudaGetDevice()
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Could not query the current CUDA device: {error}")
    error, ordering = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAWritesOrdering, device
    )
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Could not query GPUDirect RDMA ordering: {error}")
    owner_scope = (
        cudart.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner
    )
    if int(ordering) >= int(owner_scope):
        return False
    error, options = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAFlushWritesOptions, device
    )
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Could not query GPUDirect RDMA flush support: {error}")
    flush_options = cudart.cudaFlushGPUDirectRDMAWritesOptions
    host_flush = flush_options.cudaFlushGPUDirectRDMAWritesOptionHost
    if not (int(options) & int(host_flush)):
        raise RuntimeError(
            "This CUDA device cannot make GPUDirect RDMA writes visible to kernels"
        )
    return True


def _flush_gpudirect_writes() -> None:
    result = cudart.cudaDeviceFlushGPUDirectRDMAWrites(
        cudart.cudaFlushGPUDirectRDMAWritesTarget.cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
        cudart.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner,
    )[0]
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA GPUDirect RDMA write flush failed: {result}")


def engram_head_shard(
    head_sizes: tuple[int, ...], num_shards: int, shard_rank: int
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Return the first head, local sizes, and global offsets for one shard."""
    if not head_sizes or num_shards <= 0 or not 0 <= shard_rank < num_shards:
        raise ValueError("invalid Engram head shard geometry")
    part_heads = (len(head_sizes) + num_shards - 1) // num_shards
    head_start = shard_rank * part_heads
    if head_start >= len(head_sizes):
        raise ValueError(f"Engram sharding leaves rank {shard_rank} without hash heads")
    head_end = min(head_start + part_heads, len(head_sizes))
    offsets = np.cumsum((0, *head_sizes[:-1]), dtype=np.int64)
    return (
        head_start,
        head_sizes[head_start:head_end],
        tuple(int(value) for value in offsets[head_start:head_end]),
    )


@triton.jit(do_not_specialize=["num_rows"])
def _dequant_packed_engram_rows(
    packed,
    dead,
    output,
    num_rows,
    ACTUAL_HEADS: tl.constexpr,
    PART_HEADS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cols = tl.arange(0, DIM)
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = rows < num_rows
    row = rows.to(tl.int64)
    alive = tl.load(dead + row, mask=valid, other=1) == 0
    raw = tl.load(
        packed + row[:, None] * ROW_BYTES + cols[None, :],
        mask=valid[:, None],
        other=0,
    )
    # Decode E4M3FN from bytes instead of bitcasting to Triton's fp8 type.
    # The latter cannot be compiled for pre-Ada GPUs such as A10, even though
    # PyTorch can store and convert E4M3FN weights there.
    bits = raw.to(tl.int32)
    magnitude = bits & 0x7F
    exponent = magnitude >> 3
    mantissa = magnitude & 0x7
    exponent_bits = (exponent + 120) << 23  # fp32 bias 127 - E4M3 bias 7
    power = exponent_bits.to(tl.float32, bitcast=True)
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * power
    subnormal = mantissa.to(tl.float32) * 0.001953125
    values = tl.where(exponent == 0, subnormal, normal)
    values = tl.where((bits & 0x80) != 0, -values, values)
    scale_col = cols // QUANT_BLOCK
    scale = tl.load(
        packed + row[:, None] * ROW_BYTES + DIM + scale_col[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    token = row // ACTUAL_HEADS
    head = row % ACTUAL_HEADS
    out_row = token * PART_HEADS + head
    tl.store(
        output + out_row[:, None] * DIM + cols[None, :],
        tl.where(alive[:, None], values * scale, 0.0).to(tl.bfloat16),
        mask=valid[:, None],
    )


@dataclass
class _LayerBuffers:
    embedding: weakref.ReferenceType[ParallelEngramEmbedding]
    model_layer_id: int
    store_layer_id: int
    layer_hash_index: int
    head_start: int
    head_sizes: tuple[int, ...]
    global_offsets: np.ndarray
    row_bytes: int
    max_tokens: int = 0
    host_ids: list[torch.Tensor] = field(default_factory=list)
    local_ids: list[np.ndarray] = field(default_factory=list)
    host_dead: list[torch.Tensor] = field(default_factory=list)
    device_packed: list[torch.Tensor] = field(default_factory=list)
    device_dead: list[torch.Tensor] = field(default_factory=list)
    device_rows: list[torch.Tensor] = field(default_factory=list)

    @property
    def actual_heads(self) -> int:
        return len(self.head_sizes)


@dataclass
class _LookupSlot:
    ids_ready: torch.cuda.Event
    rows_consumed: torch.cuda.Event
    future: Future[int] | None = None
    num_tokens: int = 0
    writes_flushed: bool = True


def _consume_previous_lookup(slot: _LookupSlot) -> None:
    future = slot.future
    slot.future = None
    if future is not None:
        future.result()


def _release_backend(resources: dict[str, Any], executor: ThreadPoolExecutor) -> None:
    executor.shutdown(wait=True)
    store = resources.get("store")
    if store is None:
        return
    for buffer in resources["registered"]:
        rc = store.unregister_buffer(buffer.data_ptr())
        if rc != 0:
            logger.warning("Mooncake Engram buffer unregister failed: %s", rc)
    store.close()


class MooncakeEngramBackend:
    """One Store client and one multi-layer lookup queue per model rank."""

    def __init__(
        self,
        config_path: str,
        num_shards: int,
        shard_rank: int,
        num_slots: int,
    ) -> None:
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.num_shards = num_shards
        self.shard_rank = shard_rank
        self.num_slots = num_slots
        with open(self.config_path, encoding="utf-8") as manifest_file:
            self.manifest = json.load(manifest_file)
        if self.manifest.get("version") != 1:
            raise ValueError("Mooncake Engram manifest version must be 1")
        if self.manifest.get("num_shards") != num_shards:
            raise ValueError(
                "Mooncake Engram manifest num_shards does not match the active "
                f"TP/Engram-DP topology: {self.manifest.get('num_shards')} != "
                f"{num_shards}"
            )
        if not isinstance(self.manifest.get("layers"), dict):
            raise ValueError("Mooncake Engram manifest requires a layers mapping")

        self._layers: dict[int, _LayerBuffers] = {}
        self._connect_lock = threading.Lock()
        self._store: Any | None = None
        self._table: Any | None = None
        self._needs_gpudirect_flush = False
        self._executor = ThreadPoolExecutor(
            max_workers=num_slots, thread_name_prefix="vllm-engram-mooncake"
        )
        self._slots = [
            _LookupSlot(torch.cuda.Event(), torch.cuda.Event())
            for _ in range(num_slots)
        ]
        self._resources: dict[str, Any] = {"store": None, "registered": []}
        self._finalizer = weakref.finalize(
            self, _release_backend, self._resources, self._executor
        )

    def attach(
        self,
        embedding: ParallelEngramEmbedding,
        model_layer_id: int,
        layer_hash_index: int,
        full_head_sizes: tuple[int, ...],
    ) -> None:
        manifest_layer = self.manifest["layers"].get(str(model_layer_id))
        if manifest_layer is None:
            raise ValueError(
                f"Mooncake Engram manifest has no model layer {model_layer_id}"
            )
        expected_row_bytes = embedding.dim + embedding.dim // embedding.block_size
        expected = {
            "table_vocab_sizes": list(full_head_sizes),
            "head_dim": embedding.dim,
            "row_bytes": expected_row_bytes,
        }
        if manifest_layer != expected:
            raise ValueError(
                f"Mooncake Engram layout mismatch for layer {model_layer_id}: "
                f"expected {expected}, got {manifest_layer}"
            )
        head_start, head_sizes, offsets = engram_head_shard(
            full_head_sizes, self.num_shards, self.shard_rank
        )
        if head_start != embedding.head_start:
            raise ValueError("Mooncake and vLLM Engram head sharding disagree")
        store_layer_id = model_layer_id * self.num_shards + self.shard_rank
        if layer_hash_index in self._layers:
            raise ValueError(f"duplicate Engram hash layer {layer_hash_index}")
        self._layers[layer_hash_index] = _LayerBuffers(
            embedding=weakref.ref(embedding),
            model_layer_id=model_layer_id,
            store_layer_id=store_layer_id,
            layer_hash_index=layer_hash_index,
            head_start=head_start,
            head_sizes=head_sizes,
            global_offsets=np.asarray(offsets, dtype=np.int64),
            row_bytes=expected_row_bytes,
        )

    def initialize_buffers(self, layer_hash_index: int, max_tokens: int) -> None:
        layer = self._layers[layer_hash_index]
        if layer.max_tokens:
            if layer.max_tokens != max_tokens:
                raise ValueError("Mooncake Engram staging size changed after init")
            return
        embedding = layer.embedding()
        if embedding is None:
            raise RuntimeError("Mooncake Engram embedding was released during init")
        device = torch.device("cuda", torch.accelerator.current_device_index())
        layer.max_tokens = max_tokens
        for _ in range(self.num_slots):
            layer.host_ids.append(
                torch.empty(
                    max_tokens,
                    layer.actual_heads,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
            )
            layer.local_ids.append(
                np.empty((max_tokens, layer.actual_heads), dtype=np.int64)
            )
            layer.host_dead.append(
                torch.empty(
                    max_tokens,
                    layer.actual_heads,
                    dtype=torch.bool,
                    device="cpu",
                    pin_memory=True,
                )
            )
            layer.device_packed.append(
                torch.empty(
                    max_tokens,
                    layer.actual_heads,
                    layer.row_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
            )
            layer.device_dead.append(
                torch.empty(
                    max_tokens,
                    layer.actual_heads,
                    dtype=torch.bool,
                    device=device,
                )
            )
            layer.device_rows.append(
                torch.zeros(
                    max_tokens,
                    embedding.part_n_hash_cols,
                    embedding.dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
            )

    def _ordered_layers(self) -> list[_LayerBuffers]:
        layers = [self._layers[index] for index in sorted(self._layers)]
        if not layers or any(layer.max_tokens == 0 for layer in layers):
            raise RuntimeError("Mooncake Engram layers are not fully initialized")
        return layers

    def _connect(self) -> None:
        if self._table is not None:
            return
        with self._connect_lock:
            if self._table is not None:
                return
            from mooncake.mooncake_config import MooncakeConfig

            try:
                from mooncake.store import (
                    EngramStore,
                    EngramStoreConfig,
                    MooncakeDistributedStore,
                )
            except ImportError:
                # Mooncake source builds expose the extension as top-level
                # ``store``; release wheels install the same API as
                # ``mooncake.store``.
                from store import (  # type: ignore[no-redef]
                    EngramStore,
                    EngramStoreConfig,
                    MooncakeDistributedStore,
                )

            connection = MooncakeConfig.load_from_env()
            store = MooncakeDistributedStore()
            rc = store.setup(
                local_hostname=connection.local_hostname,
                metadata_server=connection.metadata_server,
                global_segment_size=connection.global_segment_size,
                local_buffer_size=connection.local_buffer_size,
                protocol=connection.protocol,
                rdma_devices=connection.device_name or "",
                master_server_addr=connection.master_server_address,
                enable_ssd_offload=connection.enable_ssd_offload,
                ssd_offload_path=connection.ssd_offload_path,
                tenant_id=connection.tenant_id,
                enable_client_http_server=connection.enable_client_http_server,
                client_http_port=connection.client_http_port,
            )
            if rc != 0:
                store.close()
                raise RuntimeError(f"Mooncake Store setup failed, rc={rc}")
            if not hasattr(EngramStore, "lookup_many_into_registered"):
                store.close()
                raise RuntimeError(
                    "Mooncake Engram requires direct registered output support"
                )

            try:
                needs_gpudirect_flush = _gpudirect_flush_required()
                configs = {}
                for layer in self._ordered_layers():
                    config = EngramStoreConfig()
                    config.table_vocab_sizes = list(layer.head_sizes)
                    config.row_bytes = layer.row_bytes
                    configs[layer.store_layer_id] = config
                table = EngramStore(configs, store=store)
                keys = [
                    key
                    for layer in self._ordered_layers()
                    for key in table.get_store_keys(layer.store_layer_id)
                ]
                exists = store.batch_is_exist(keys)
            except Exception:
                store.close()
                raise
            if len(exists) != len(keys) or any(value != 1 for value in exists):
                store.close()
                missing = max(0, len(keys) - len(exists)) + sum(
                    value != 1 for value in exists
                )
                raise RuntimeError(
                    f"Mooncake Engram tables are incomplete ({missing} missing keys)"
                )
            registered: list[torch.Tensor] = []
            try:
                for layer in self._ordered_layers():
                    for packed in layer.device_packed:
                        if (
                            store.register_buffer(packed.data_ptr(), packed.numel())
                            != 0
                        ):
                            raise RuntimeError(
                                "Could not register Mooncake Engram output buffer"
                            )
                        registered.append(packed)
            except Exception:
                for host in registered:
                    store.unregister_buffer(host.data_ptr())
                store.close()
                raise
            self._resources["store"] = store
            self._resources["registered"] = registered
            self._store = store
            self._table = table
            self._needs_gpudirect_flush = needs_gpudirect_flush
            logger.info(
                "Connected Mooncake Engram shard %d/%d with model layers %s",
                self.shard_rank,
                self.num_shards,
                [layer.model_layer_id for layer in self._ordered_layers()],
            )

    def _lookup(self, slot_index: int, num_tokens: int) -> int:
        slot = self._slots[slot_index]
        slot.rows_consumed.synchronize()
        slot.ids_ready.synchronize()
        layers = self._ordered_layers()
        active_end = 0
        global_ids_by_layer = []
        for layer in layers:
            global_ids = layer.host_ids[slot_index][:num_tokens].numpy()
            global_ids_by_layer.append(global_ids)
            active = np.flatnonzero(np.any(global_ids != -1, axis=1))
            if active.size:
                active_end = max(active_end, int(active[-1]) + 1)
        if active_end == 0:
            for layer in layers:
                layer.host_dead[slot_index][:num_tokens].fill_(True)
            return 0

        layer_ids: list[int] = []
        local_ids: list[np.ndarray] = []
        output_addresses: list[int] = []
        output_sizes: list[int] = []
        for layer, global_ids in zip(layers, global_ids_by_layer):
            current = layer.local_ids[slot_index][:active_end]
            np.subtract(global_ids[:active_end], layer.global_offsets, out=current)
            dead = global_ids[:num_tokens] == -1
            active_dead = dead[:active_end]
            valid = active_dead | (
                (current >= 0)
                & (current < np.asarray(layer.head_sizes, dtype=np.int64))
            )
            if not np.all(valid):
                raise RuntimeError(
                    f"Engram hashes are outside layer {layer.model_layer_id}'s "
                    "owned head ranges"
                )
            current[active_dead] = 0
            layer.host_dead[slot_index][:num_tokens].copy_(torch.from_numpy(dead))
            output = layer.device_packed[slot_index]
            layer_ids.append(layer.store_layer_id)
            local_ids.append(current[None])
            output_addresses.append(output.data_ptr())
            output_sizes.append(active_end * layer.actual_heads * layer.row_bytes)

        assert self._table is not None
        self._table.lookup_many_into_registered(
            layer_ids, local_ids, output_addresses, output_sizes
        )
        return active_end

    @eager_break_during_capture
    def prefetch(self, gathered_hashes: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mooncake Engram host I/O requires breakable CUDA graphs; set "
                "VLLM_USE_BREAKABLE_CUDAGRAPH=1 or run without CUDA graphs"
            )
        self._connect()
        slot_index = dbo_current_ubatch_id()
        if slot_index >= self.num_slots:
            raise RuntimeError("Mooncake Engram has no buffer for this DBO slot")
        slot = self._slots[slot_index]
        _consume_previous_lookup(slot)

        num_tokens = gathered_hashes.shape[0]
        layers = self._ordered_layers()
        for layer in layers:
            if num_tokens > layer.max_tokens:
                raise RuntimeError("Mooncake Engram staging buffer is too small")
            source = gathered_hashes[
                :,
                layer.layer_hash_index,
                layer.head_start : layer.head_start + layer.actual_heads,
            ]
            layer.host_ids[slot_index][:num_tokens].copy_(source, non_blocking=True)
        slot.ids_ready.record(torch.cuda.current_stream())
        slot.num_tokens = num_tokens
        slot.writes_flushed = False
        slot.future = self._executor.submit(self._lookup, slot_index, num_tokens)

    @eager_break_during_capture
    def wait(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mooncake Engram host I/O requires breakable CUDA graphs"
            )
        slot = self._slots[dbo_current_ubatch_id()]
        if slot.future is None:
            raise RuntimeError("Mooncake Engram rows were not prefetched")
        active_rows = slot.future.result()
        if active_rows and not slot.writes_flushed:
            if self._needs_gpudirect_flush:
                _flush_gpudirect_writes()
            slot.writes_flushed = True

    def rows(self, layer_hash_index: int) -> torch.Tensor:
        slot_index = dbo_current_ubatch_id()
        slot = self._slots[slot_index]
        self.wait()
        layer = self._layers[layer_hash_index]
        num_tokens = slot.num_tokens
        packed = layer.device_packed[slot_index][:num_tokens]
        dead = layer.device_dead[slot_index][:num_tokens]
        dead.copy_(layer.host_dead[slot_index][:num_tokens], non_blocking=True)
        output = layer.device_rows[slot_index][:num_tokens]
        num_rows = num_tokens * layer.actual_heads
        if num_rows:
            embedding = layer.embedding()
            if embedding is None:
                raise RuntimeError("Mooncake Engram embedding was released")
            _dequant_packed_engram_rows[(triton.cdiv(num_rows, 16),)](
                packed,
                dead,
                output,
                num_rows,
                ACTUAL_HEADS=layer.actual_heads,
                PART_HEADS=embedding.part_n_hash_cols,
                ROW_BYTES=layer.row_bytes,
                DIM=embedding.dim,
                QUANT_BLOCK=embedding.block_size,
                BLOCK_R=16,
            )
        slot.rows_consumed.record(torch.cuda.current_stream())
        return output


_BACKENDS: weakref.WeakValueDictionary[
    tuple[str, int, int, int], MooncakeEngramBackend
] = weakref.WeakValueDictionary()
_BACKENDS_LOCK = threading.Lock()


def get_mooncake_engram_backend(
    config_path: str, num_shards: int, shard_rank: int, num_slots: int
) -> MooncakeEngramBackend:
    key = (
        str(Path(config_path).expanduser().resolve()),
        num_shards,
        shard_rank,
        num_slots,
    )
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(key)
        if backend is None:
            backend = MooncakeEngramBackend(*key)
            _BACKENDS[key] = backend
        return backend
