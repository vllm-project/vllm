# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic KV parallelism (KVP): attention over a KV cache sharded across
ranks, for attention backends without native DCP support.

Rank ``r`` of ``N`` stores page ``r`` of every block (native DCP's layout with a
block-sized interleave). Attention runs unsharded on a scratch cache holding all
pages (kernel block ``b * N + r``). A side stream pulls a layer's pages from the
other ranks' symmetric-memory caches while the previous layer finishes; the
layer acquires its cache before its first KV access and releases it after its
last, which persists its new pages and starts pulling the next layer.
"""

import functools
import itertools
from collections import defaultdict
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed._symmetric_memory as symm_mem

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.gpu.buffer_utils import _load_ptr

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner

logger = init_logger(__name__)

# Ranks each KV cache block is sharded across (1: off).
_size = 1
_active: "GenericKVP | None" = None


@functools.cache
def _layer_index(name: str) -> int:
    """The decoder layer of a layer or KV cache name."""
    return int(next(part for part in name.split(".") if part.isdigit()))


def get_generic_kvp_size() -> int:
    return _size


def maybe_enter_generic_kvp_mode(vllm_config: VllmConfig) -> None:
    """With ``--dcp-gather``, shard the KV cache across the DCP group but run
    attention unsharded. Call on every rank before distributed init."""
    global _size
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    size, _size = parallel_config.decode_context_parallel_size, 1
    if size == 1 or not parallel_config.dcp_gather:
        parallel_config.dcp_gather = False
        return
    assert model_config is not None and model_config.use_mla, "MLA models only."
    if (
        not vllm_config.use_v2_model_runner
        or vllm_config.speculative_config is not None
        or parallel_config.use_ubatching
        or parallel_config.enable_elastic_ep
        or parallel_config.distributed_executor_backend in ("uni", "external_launcher")
        or model_config.enable_sleep_mode
    ):
        # Draft layers and microbatches would bypass acquire/release, in-process
        # executors would leak DCP=1 to the engine, and sleep mode uses its own
        # KV cache pool.
        raise NotImplementedError(
            "--dcp-gather needs Model Runner V2, without spec decode, ubatching, "
            "elastic EP, in-process executors or sleep mode."
        )
    _size, parallel_config.decode_context_parallel_size = size, 1
    compilation_config = vllm_config.compilation_config
    if compilation_config.cudagraph_mode.has_full_cudagraphs():
        # Acquire and release run eagerly, between graph pieces.
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    logger.info_once("Running DCP=%d by gathering the KV cache per layer.", _size)


def symmetric_kv_cache_allocation(device: torch.device) -> AbstractContextManager:
    """Allocate the KV cache so that peers can read it."""
    return torch.cuda.use_mem_pool(symm_mem.get_mem_pool(device))


def generic_kvp_kv_cache_fraction(kv_cache_spec: dict[str, KVCacheSpec]) -> float:
    """Share of the KV cache budget left once the scratch caches (``N`` pages
    per block and cache shape) are set aside."""
    pages = sum(spec.page_size_bytes for spec in kv_cache_spec.values())
    scratch = {
        _page_stride(spec)
        for spec in kv_cache_spec.values()
        if getattr(spec, "dcp_sharded", False)
    }
    return pages / (pages + _size * sum(scratch))


def _page_stride(spec: KVCacheSpec) -> int:
    """Bytes between pages that attention kernels accept for ``spec``."""
    return round_up(spec.page_size_bytes, spec.block_stride_alignment or 1)


def maybe_build_generic_kvp(
    runner: "GPUModelRunner", kv_caches: dict[str, torch.Tensor]
) -> "GenericKVP | None":
    """Build the runtime once the KV caches exist."""
    if _size == 1:
        return None
    layers = runner.compilation_config.static_forward_context
    return GenericKVP(runner.kv_cache_config, kv_caches, layers, runner.block_tables)


def get_generic_kvp() -> "GenericKVP | None":
    """The runtime, while the current forward gathers KV cache blocks."""
    return _active


class GenericKVP:
    def __init__(
        self,
        config: KVCacheConfig,
        kv_caches: dict[str, torch.Tensor],
        layers: dict[str, Any],
        block_tables: "BlockTables",
    ):
        first = torch.distributed.get_rank() // _size * _size
        # TP and PCP are the innermost rank dimensions: these ranks run the
        # same requests.
        group = torch.distributed.new_group(
            list(range(first, first + _size)), use_local_synchronization=True
        )
        # The KV caches are views into one allocation, at the same offsets on
        # every rank.
        self.handle = symm_mem.rendezvous(next(iter(kv_caches.values())), group)
        bases = torch.tensor(self.handle.buffer_ptrs, dtype=torch.int64)
        scratch: dict[tuple[Any, ...], torch.Tensor] = {}
        # Bound on first use, after KV-block zeroing captured the KV caches.
        self.unbound: list[tuple[Any, torch.Tensor]] = []
        # Per decoder layer, its caches (e.g. MLA and indexer): KV cache group,
        # scratch rows, every rank's address of the cache, and its row stride
        # (in 8-byte words).
        self.layers: defaultdict[Any, list[Any]] = defaultdict(list)
        for g, group_config in enumerate(config.kv_cache_groups):
            group_spec = group_config.kv_cache_spec
            for name in group_config.layer_names if group_spec.dcp_sharded else []:
                cache, device = kv_caches[name], kv_caches[name].device
                spec = getattr(group_spec, "kv_cache_specs", {}).get(name, group_spec)
                page = _page_stride(spec) // cache.element_size()
                strides = (page, *cache.stride()[1:])
                key = (cache.shape, strides, cache.dtype)
                if key not in scratch:
                    shape = (cache.shape[0] * _size, *cache.shape[1:])
                    scratch[key] = torch.empty_strided(
                        shape, strides, dtype=cache.dtype, device=device
                    )
                self.unbound.append((layers[name], scratch[key]))
                rows = scratch[key].view(torch.uint8).flatten(1).view(torch.int64)
                offset = cache.data_ptr() - self.handle.buffer_ptrs[self.handle.rank]
                assert 0 <= offset < self.handle.buffer_size
                stride = cache.stride(0) * cache.element_size() // 8
                peers = (bases + offset).to(device)
                self.layers[_layer_index(name)].append((g, rows, peers, stride))
        # Layers run in index order; None is before the first and after the last.
        self.next = dict(itertools.pairwise([None, *sorted(self.layers), None]))
        self.index: int | None = None
        self.event: torch.Event | None = None
        self.block_tables = block_tables
        # Per group: the block table, and the positions in it of the blocks the
        # forward writes and of all its blocks.
        self.blocks: dict[int, Any] = {
            g: None for caches in self.layers.values() for g, *_ in caches
        }
        # Copies claim the scratch rows they fill, so blocks requests share are
        # copied once.
        self.claims = torch.zeros(
            config.num_blocks * _size, dtype=torch.int64, device=device
        )
        self.epoch = 0
        self.stream = torch.Stream(device=device)

    def prepare(self, batch: "BatchReqState | None", num_computed: np.ndarray) -> None:
        """Start pulling the KV cache blocks of ``batch``'s requests."""
        global _active
        _active = None
        for layer, cache in self.unbound:
            layer.bind_kv_cache(cache)
        self.unbound = []
        if batch is None:
            return
        reqs: np.ndarray = batch.idx_mapping_np.astype(np.int64)
        start = num_computed[reqs]
        end = start + batch.num_scheduled_tokens

        def positions(table: torch.Tensor, lo: np.ndarray, hi: np.ndarray):
            """Block-table positions of each request's blocks ``lo..hi-1``."""
            n = hi - lo
            j = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n - lo, n)
            pos = np.repeat(reqs * table.stride(0), n) + j * _size
            return torch.from_numpy(pos).to(table.device, non_blocking=True)

        for g in self.blocks:
            table = self.block_tables.block_tables[g].gpu
            tokens = self.block_tables.block_sizes[g]
            num_blocks = self.block_tables.num_blocks.np[g, reqs] // _size
            self.blocks[g] = (
                table,
                positions(table, start // tokens, (end - 1) // tokens + 1),
                positions(table, np.zeros_like(num_blocks), num_blocks),
            )
        assert self.index is None, "The previous forward did not release a layer."
        _active = self
        self._advance()

    def acquire(self, layer_name: str) -> None:
        """Wait for the layer's KV cache. Call before the layer's first access."""
        index = _layer_index(layer_name)
        if index == self.index and self.event is not None:
            torch.accelerator.current_stream().wait_event(self.event)
            self.event = None
        assert index == self.index or index not in self.layers, (
            f"{layer_name} accessed its KV cache before it was gathered."
        )

    def release(self, layer_name: str) -> None:
        """Persist the layer's writes and gather the next layer. Call after the
        layer's last KV cache access."""
        if _layer_index(layer_name) == self.index:
            self._advance()

    def _advance(self) -> None:
        """After the compute stream's work so far: on the side stream, copy the
        current layer's new pages back and pull every rank's pages of the next.
        """
        global _active
        compute = torch.accelerator.current_stream()
        self.stream.wait_stream(compute)
        with self.stream:
            done, self.index = self.index, self.next[self.index]
            for index, pull in ((done, False), (self.index, True)):
                for g, rows, peers, row_stride in self.layers.get(index, []):
                    table, *positions = self.blocks[g]
                    if (n := positions[pull].numel()) == 0:
                        continue
                    self.epoch += 1
                    _copy_pages[(n, _size if pull else 1)](
                        rows, rows.stride(0), rows.shape[1], peers, row_stride,
                        0 if pull else self.handle.rank, _size, table,
                        positions[pull], self.claims, self.epoch, PULL=pull,
                        BLOCK=triton.next_power_of_2(rows.shape[1]), num_warps=8,
                    )  # fmt: skip
            self.event = self.stream.record_event()
        if self.index is None:
            # The forward is done. Peers read its write-backs, and blocks it
            # freed may be zeroed next.
            _active = None
            compute.wait_stream(self.stream)
            self.handle.barrier()


@triton.jit(do_not_specialize=["epoch"])
def _copy_pages(
    scratch_ptr, scratch_stride, row_len, peers_ptr, row_stride,
    first_rank, size, table_ptr, positions_ptr, claims_ptr, epoch,
    PULL: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    # Rank r's page of block b is row b of its cache (at peers_ptr[r]) and row
    # b * size + r of the scratch cache.
    rank = first_rank + tl.program_id(1)
    position = tl.load(positions_ptr + tl.program_id(0))
    block = (tl.load(table_ptr + position) // size).to(tl.int64)
    row = block * size + rank
    if tl.atomic_xchg(claims_ptr + row, epoch) != epoch:
        offs = tl.arange(0, BLOCK)
        mask = offs < row_len
        scratch = scratch_ptr + row * scratch_stride + offs
        page = _load_ptr(peers_ptr + rank, tl.int64) + block * row_stride + offs
        if PULL:
            tl.store(scratch, tl.load(page, mask=mask), mask=mask)
        else:
            tl.store(page, tl.load(scratch, mask=mask), mask=mask)
