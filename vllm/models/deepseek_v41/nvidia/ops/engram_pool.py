# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSv4.1 Engram tables on NVIDIA beyond one engine's memory.

Host tables: one copy in host-NUMA memory per node-local DP group, shared by
FABRIC handle, which GPUs map with 2M pages instead of /dev/shm's 64K pages.

Pool tables: rows split over GPUs of several engines by an Engram pool
(`vllm.distributed.engram_pool`). One fetch per step reads every layer's rows;
each layer's WKV input is unpacked to MXFP8 where its rows land.
"""

import ctypes
import weakref
from functools import cache
from typing import NamedTuple

import torch
from cuda.bindings import driver as cu
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.engram_pool import (
    ROW_ALIGN,
    EngramFetch,
    EngramPool,
    join_peer_group,
    wait_landed,
)
from vllm.distributed.engram_pool.memory import device_view
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import (
    Engram,
    EngramBatch,
    EngramLayout,
    ParallelEngramEmbedding,
    describe_error,
    gather_errors,
)
from vllm.models.deepseek_v41.common.ops.query_quant import (
    can_fuse_query_quant,
    mxfp8_scale_bytes,
    mxfp8_scale_offset,
)
from vllm.triton_utils import tl, triton
from vllm.utils.cpu_resource_utils import get_memory_node_info
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

FABRIC_HANDLE = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
# Host tables are created in chunks, so the creator stops before draining its node.
CHUNK_BYTES = 2 << 30
# Fraction of the NUMA node left to other allocations.
NUMA_RESERVE_FRACTION = 0.05


class ExportedHostTable(NamedTuple):
    """What another rank needs to map a host table."""

    numa_node: int
    padded_bytes: int
    fabric_handles: list[bytes]


def _chunk_ranges(padded_bytes: int) -> list[tuple[int, int]]:
    return [
        (offset, min(CHUNK_BYTES, padded_bytes - offset))
        for offset in range(0, padded_bytes, CHUNK_BYTES)
    ]


def _call_driver(fn, *args):
    """Call a CUDA driver API and return its result, raising on failure."""
    error, *result = fn(*args)
    if error != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{fn.__name__} failed: {error.name}")
    return result[0] if result else None


def _gpu_numa_node() -> int:
    return _call_driver(
        cu.cuDeviceGetAttribute,
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
        torch.accelerator.current_device_index(),
    )


def _host_allocation_prop(numa_node: int) -> cu.CUmemAllocationProp:
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = numa_node
    prop.requestedHandleTypes = FABRIC_HANDLE
    return prop


def _allocation_granularity(prop: cu.CUmemAllocationProp) -> int:
    return _call_driver(cu.cuMemGetAllocationGranularity, prop, 0)


def _export_fabric_handle(handle) -> bytes:
    exported = _call_driver(cu.cuMemExportToShareableHandle, handle, FABRIC_HANDLE, 0)
    return bytes(exported.data)


def _release(handles: list) -> None:
    for handle in handles:
        cu.cuMemRelease(handle)


def _read_write_access(location_type, location_id: int) -> cu.CUmemAccessDesc:
    access = cu.CUmemAccessDesc()
    access.location.type = location_type
    access.location.id = location_id
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return access


def _unmap(address: int, size: int, ranges: list[tuple[int, int]]) -> None:
    for offset, length in ranges:
        cu.cuMemUnmap(address + offset, length)
    cu.cuMemAddressFree(address, size)


@cache
def fabric_host_memory_supported() -> bool:
    """Whether host-NUMA cuMem allocations can export FABRIC handles here."""
    try:
        prop = _host_allocation_prop(_gpu_numa_node())
        handle = _call_driver(cu.cuMemCreate, _allocation_granularity(prop), prop, 0)
        try:
            _export_fabric_handle(handle)
        finally:
            cu.cuMemRelease(handle)
    except Exception as exc:
        logger.info_once(
            "Host-NUMA cuMem FABRIC handles are unavailable (%s); Engram DP "
            "replicas share tables through /dev/shm.",
            exc,
        )
        return False
    return True


def _create_table(num_bytes: int) -> tuple[list, ExportedHostTable]:
    """Allocate and export a host table on this GPU's NUMA node.

    Stops once the node would drop below its reserve. The kernel compacts to
    find 2M pages and fails the chunk, rather than OOM-killing, if it cannot.
    """
    numa_node = _gpu_numa_node()
    prop = _host_allocation_prop(numa_node)
    padded_bytes = round_up(num_bytes, _allocation_granularity(prop))
    reserve = get_memory_node_info(numa_node).total_memory * NUMA_RESERVE_FRACTION
    handles: list = []
    try:
        for offset, length in _chunk_ranges(padded_bytes):
            handles.append(_call_driver(cu.cuMemCreate, length, prop, 0))
            if get_memory_node_info(numa_node).available_memory < reserve:
                raise MemoryError(
                    f"NUMA node {numa_node} fell below its "
                    f"{reserve / 2**30:.1f} GiB reserve after "
                    f"{(offset + length) / 2**30:.1f} of "
                    f"{padded_bytes / 2**30:.1f} GiB"
                )
        exported = [_export_fabric_handle(handle) for handle in handles]
    except BaseException:
        _release(handles)
        raise
    return handles, ExportedHostTable(numa_node, padded_bytes, exported)


def _map_table(
    table: ExportedHostTable, num_bytes: int, handles: list
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a host table, importing its handles unless given, and release them.

    Returns CPU and GPU uint8 views; either keeps the mapping alive.
    """
    device = torch.accelerator.current_device_index()
    address = None
    mapped: list[tuple[int, int]] = []
    try:
        if not handles:
            for exported in table.fabric_handles:
                handles.append(
                    _call_driver(
                        cu.cuMemImportFromShareableHandle, exported, FABRIC_HANDLE
                    )
                )
        address = int(_call_driver(cu.cuMemAddressReserve, table.padded_bytes, 0, 0, 0))
        for (offset, length), handle in zip(_chunk_ranges(table.padded_bytes), handles):
            _call_driver(cu.cuMemMap, address + offset, length, 0, handle, 0)
            mapped.append((offset, length))
        location = cu.CUmemLocationType
        access = [
            _read_write_access(location.CU_MEM_LOCATION_TYPE_DEVICE, device),
            _read_write_access(
                location.CU_MEM_LOCATION_TYPE_HOST_NUMA, table.numa_node
            ),
        ]
        _call_driver(
            cu.cuMemSetAccess, address, table.padded_bytes, access, len(access)
        )
    except BaseException:
        if address is not None:
            _unmap(address, table.padded_bytes, mapped)
        raise
    finally:
        # The mappings keep the allocations alive.
        _release(handles)

    host_buffer = (ctypes.c_uint8 * num_bytes).from_address(address)
    finalizer = weakref.finalize(
        host_buffer, _unmap, address, table.padded_bytes, mapped
    )
    finalizer.atexit = False  # type: ignore[misc]
    gpu_view = device_view(address, num_bytes, owner=host_buffer)
    return torch.frombuffer(host_buffer, dtype=torch.uint8), gpu_view


def share_host_table(
    num_bytes: int, group: GroupCoordinator
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Map one table on every rank of a node-local group, or None on all of them.

    Rank 0 creates the table and broadcasts its handles; the others import them.
    Returns CPU and GPU uint8 views of the table.
    """
    handles: list = []
    table = error = None
    if group.rank_in_group == 0 and fabric_host_memory_supported():
        try:
            handles, table = _create_table(num_bytes)
        except Exception as exc:
            error = describe_error(exc)
    table, error = group.broadcast_object((table, error))
    if table is not None:
        views = None
        try:
            views = _map_table(table, num_bytes, handles)
        except Exception as exc:
            error = describe_error(exc)
        if not (error := gather_errors(group, error)):
            return views
    if error and group.rank_in_group == 0:
        logger.warning(
            "Engram host table cannot use host-NUMA cuMem (%s); falling back "
            "to /dev/shm, whose 64K GPU pages make lookups slower.",
            error,
        )
    return None


@cache
def get_engram_pool(layout: EngramLayout) -> EngramPool:
    """This GPU's pool, read by one fetch for every Engram layer's table."""
    vllm_config = get_current_vllm_config()
    config = vllm_config.engram_config
    assert config is not None and config.peer_group_store is not None
    n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
    return EngramPool(
        join_peer_group(config.peer_group_size, config.peer_group_store),
        config.peer_group_transport,
        config.cpu_offload,
        vllm_config.scheduler_config.max_num_batched_tokens * n_hash_cols,
        _row_bytes(layout.row_bytes, layout.head_dim, 32),
        len(layout.num_embeddings),
    )


def _row_bytes(row_bytes: int, dim: int, block_size: int) -> int:
    """A pool row: values, then scales, padded to the pool's alignment."""
    return round_up(row_bytes + dim // block_size, ROW_ALIGN)


@triton.jit(do_not_specialize=["num_rows"])
def _dequant_rows_kernel(
    packed,
    out,
    num_rows,
    ROW_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    PACKED_FP4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Dequantize pool rows (FP8 or MXFP4 values, then E8M0 scales) into BF16."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < num_rows
    ROW_BYTES: tl.constexpr = DIM // 2 if PACKED_FP4 else DIM
    src = packed + rows * ROW_STRIDE
    cols = tl.arange(0, DIM)
    if PACKED_FP4:
        byte = tl.load(
            src[:, None] + (cols // 2)[None, :], mask=valid[:, None], other=0
        ).to(tl.int32)
        nib = tl.where(cols[None, :] % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
        e = (nib & 0x7) >> 1
        m = (nib & 0x1).to(tl.float32)
        pow2 = ((e - 2 + 127) << 23).to(tl.float32, bitcast=True)
        mag = tl.where(e == 0, m * 0.5, (2.0 + m) * pow2)
        values = tl.where((nib & 0x8) != 0, -mag, mag)
    else:
        values = tl.load(
            src.to(tl.pointer_type(tl.float8e4nv))[:, None] + cols[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
    scale = tl.load(
        src[:, None] + ROW_BYTES + (cols // QUANT_BLOCK)[None, :],
        mask=valid[:, None],
        other=0,
    )
    scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    tl.store(
        out + rows[:, None] * DIM + cols[None, :],
        (values * scale).to(tl.bfloat16),
        mask=valid[:, None],
    )


@triton.jit(do_not_specialize=["num_tokens", "table"])
def _unpack_fp8_kernel(
    rows,
    slots,
    ready,
    values,
    scales,
    num_tokens,
    table,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """Split pool table `table`'s `[token][head]` rows into MXFP8 values and
    F8_128x4 scales. If SLOTS, row i is row `slots[i]` of `rows` (zeros if -1),
    read once landed. Scales past `num_tokens` zero-pad the 128-row tile."""
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token, head = row // HEADS, row % HEADS
    valid = token < num_tokens
    source, live = row, valid
    if SLOTS:
        wait_landed(ready, table)
        source = tl.load(slots + row, mask=valid, other=-1).to(tl.int64)
        live = valid & (source >= 0)
    src = rows + tl.multiple_of(source * ROW_BYTES, ROW_BYTES & -ROW_BYTES)
    cols = tl.arange(0, DIM)
    data = tl.load(src[:, None] + cols[None, :], mask=live[:, None], other=0)
    tl.store(values + row[:, None] * DIM + cols[None, :], data, mask=valid[:, None])
    scale_cols = tl.arange(0, DIM // 32)
    scale = tl.load(
        src[:, None] + DIM + scale_cols[None, :], mask=live[:, None], other=0
    )
    group = head[:, None] * (DIM // 32) + scale_cols[None, :]
    offset = mxfp8_scale_offset(token[:, None], group, HEADS * DIM // 32)
    tl.store(scales + offset, scale)


class PoolEngramEmbedding(ParallelEngramEmbedding):
    """A table whose rows an Engram pool splits over GPUs of several engines.

    Pool rows interleave each row's values and scales; the parameters are
    strided views of this GPU's shard, so loaders fill it in place.
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        packed_fp4: bool,
        pool: EngramPool,
        name: str,
        block_size: int = 32,
    ) -> None:
        nn.Module.__init__(self)
        assert get_tensor_model_parallel_world_size() == 1
        self.num_embeddings, self.dim, self.block_size = num_embeddings, dim, block_size
        self.packed_fp4 = packed_fp4
        self.row_bytes = dim // 2 if packed_fp4 else dim
        self.n_hash_cols = self.part_n_hash_cols = len(head_sizes)
        self.tp_size = self.dp_size = 1
        self.cpu_offload = False
        self.pool = pool
        weight_dtype = torch.uint8 if packed_fp4 else torch.float8_e4m3fn
        scale_dim = dim // block_size
        if torch.get_default_device().type == "meta":
            self.table = None
            row_start = 0
            weight = torch.empty(sum(head_sizes), self.row_bytes, dtype=weight_dtype)
            scales = torch.empty(sum(head_sizes), scale_dim, dtype=torch.uint8)
        else:
            self.table = pool.table(
                sum(head_sizes), _row_bytes(self.row_bytes, dim, block_size), name
            )
            row_start, shard = self.table.row_start, self.table.shard
            weight = shard[:, : self.row_bytes].view(weight_dtype)
            scales = shard[:, self.row_bytes : self.row_bytes + scale_dim]
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales, requires_grad=False)
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {"weight_loader": self._weight_loader, "engram_vocab_start": row_start},
            )
        # Constant dummy values; the ue8m0 encoding of scale 1.0 is 127.
        set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
        set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 127})
        self._loaded = False

    def process_weights_after_loading(self) -> None:
        """Serve once every GPU of the group has loaded its rows."""
        assert self.table is not None
        shard = self.table.shard.data_ptr()
        if (self.weight.data_ptr(), self.weight_scale_inv.data_ptr()) != (
            shard,
            shard + self.row_bytes,
        ):
            raise RuntimeError("Engram pool table parameters must not be replaced")
        if self._loaded:
            # Other engines read these rows at any time, also while they reload.
            raise RuntimeError(
                "Engram tables split across engines (peer_group_size) cannot "
                "reload weights; restart the peer group instead"
            )
        self.pool.ready()
        self._loaded = True

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up every head of `indices` into BF16 `out`."""
        assert self.table is not None and out.dtype == torch.bfloat16
        rows = indices.shape[0] * self.n_hash_cols
        if not rows:
            return
        packed = self.pool.lookup([self.table], indices[:, None])
        _dequant_rows_kernel[(triton.cdiv(rows, 16),)](
            packed,
            out,
            rows,
            ROW_STRIDE=self.table.row_bytes,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            PACKED_FP4=self.packed_fp4,
            BLOCK=16,
        )


class PoolEngram(Engram):
    """An Engram layer whose table is split over an Engram pool."""

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> ParallelEngramEmbedding:
        return PoolEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            layout.packed_fp4,
            get_engram_pool(layout),
            f"layer{layout.layer_ids[layer_hash_index]}",
        )


class PoolEngramBatch(EngramBatch):
    """Stage every layer's MXFP8 WKV input from one pool fetch per step.

    `sync` (EngramConfig.peer_group_fetch): the fetch starts on the compute
    stream before the layers and each layer unpacks its rows as they land.
    `async`: fetch and unpacks run on a side stream; a layer waits for its own.
    """

    def __init__(self, layers: list[Engram], max_tokens: int) -> None:
        first = layers[0].layer_hash_index
        assert [layer.layer_hash_index for layer in layers] == list(
            range(first, first + len(layers))
        )
        self.layers, self.first = layers, first
        table = layers[0].embed_tokens
        assert isinstance(table, PoolEngramEmbedding)
        self.pool = table.pool
        self.tables = [layer.embed_tokens.table for layer in layers]
        self.peer_exchange = None
        device = layers[0].staged_rows.device
        width = table.n_hash_cols * table.dim
        self.fp8_values = torch.empty(
            len(layers), max_tokens, width, dtype=torch.uint8, device=device
        )
        self.fp8_scales = torch.empty(
            len(layers),
            mxfp8_scale_bytes(max_tokens, width),
            dtype=torch.uint8,
            device=device,
        )
        config = get_current_vllm_config().engram_config
        self._async = config is not None and config.peer_group_fetch == "async"
        if self._async:
            self._stream = torch.cuda.Stream(priority=-1)
            self._done = [torch.cuda.Event() for _ in layers]
        self._fetch: EngramFetch | None = None
        self._num_tokens = 0
        for layer in layers:
            layer._batch = self
            layer.staged_rows = layer.staged_rows.new_empty(
                (0, *layer.staged_rows.shape[1:])
            )

    @classmethod
    def create(cls, layers: list[Engram], max_tokens: int) -> "EngramBatch | None":
        if (
            not layers
            or any(layer.embed_tokens.packed_fp4 for layer in layers)
            or layers[0].embed_tokens.dim % 128
            or get_current_vllm_config().lora_config is not None
            or not can_fuse_query_quant([layer.wkv for layer in layers])
        ):
            return None
        return cls(layers, max_tokens)

    def prepare_embeddings(self, hashes: torch.Tensor) -> None:
        hashes = hashes[:, self.first : self.first + len(self.tables)]
        if self._async:
            self._start_async(hashes)
        else:
            self._fetch = self.pool.start(self.tables, hashes)
            self._num_tokens = hashes.shape[0]

    @eager_break_during_capture
    def _start_async(self, hashes: torch.Tensor) -> None:
        stream = self._stream
        stream.wait_stream(torch.cuda.current_stream())
        hashes.record_stream(stream)
        with torch.cuda.stream(stream):
            fetch = self.pool.start(self.tables, hashes)
            for index in range(len(self.tables)):
                self._unpack(fetch, index, hashes.shape[0])
                self._done[index].record(stream)

    @eager_break_during_capture
    def _wait_async(self, index: int) -> None:
        torch.cuda.current_stream().wait_event(self._done[index])

    def _unpack(self, fetch: EngramFetch, index: int, num_tokens: int) -> None:
        if not num_tokens:
            return
        rows, slots, ready = fetch.unplaced(index)
        table = self.layers[0].embed_tokens
        heads = table.n_hash_cols
        _unpack_fp8_kernel[(triton.cdiv(num_tokens, 128) * 128 * heads // 16,)](
            rows,
            rows if slots is None else slots,
            rows if ready is None else ready,
            self.fp8_values[index],
            self.fp8_scales[index],
            num_tokens,
            index,
            HEADS=heads,
            DIM=table.dim,
            ROW_BYTES=self.tables[index].row_bytes,
            BLOCK=16,
            SLOTS=slots is not None,
        )

    def wkv_input(self, layer: Engram, num_tokens: int) -> QuantizedActivation:
        index = layer.layer_hash_index - self.first
        if self._async:
            self._wait_async(index)
        else:
            assert self._fetch is not None
            self._unpack(self._fetch, index, self._num_tokens)
        return super().wkv_input(layer, num_tokens)
