# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layerwise DSA Main KV workspace on a dedicated prefill producer."""

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.stats import (
    HiSparseKVConnectorStats,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tp_group
from vllm.v1.hisparse.backend import get_hisparse_backend
from vllm.v1.hisparse.dma import DMADescriptors, HiSparseDMA
from vllm.v1.hisparse.layout import layerwise_buffer_groups
from vllm.v1.hisparse.runtime import (
    drop_hisparse_host_pool,
    get_hisparse_host_pool,
    release_pinned_state,
)
from vllm.v1.hisparse.types import LayerwiseStep
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    create_kv_cache_views,
)
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnectorMetadata,
    )


def allocate_layer_buffers(
    cache_config: KVCacheConfig,
    num_buffers: int,
    kernel_sizes: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    from vllm.model_executor.models.utils import extract_layer_index

    (source_id,) = cache_config.host_group_ids
    source = cache_config.kv_cache_groups[source_id]
    idx = cache_config.hisparse_prefill_indexer_group
    assert idx is not None
    ratio = cache_config.kv_cache_groups[idx].block_size // source.block_size
    pages = cache_config.num_blocks * ratio
    views: dict[str, torch.Tensor] = {}
    for names, count, stride in layerwise_buffer_groups(source, num_buffers):
        names.sort(key=extract_layer_index)
        spec = source.kv_cache_specs[names[0]]
        assert isinstance(spec, MLAAttentionSpec)
        if stride != spec.page_size_bytes:
            spec = replace(spec, page_size_padded=stride)
        size = stride * pages
        tensor = KVCacheTensor(size, [names[0]], size, stride)
        ring = []
        for _ in range(count):
            raw = torch.zeros(size, dtype=torch.int8, device=device)
            (view,) = create_kv_cache_views(
                raw, spec, pages, KVCacheLayout.LBNHC, tensor, kernel_sizes[source_id]
            )
            ring.append(view)
        views.update((name, ring[i % count]) for i, name in enumerate(names))
    return views


def attach_layerwise_hooks(model: torch.nn.Module, cache_config: KVCacheConfig) -> None:
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper

    (source_id,) = cache_config.host_group_ids
    expected = set(cache_config.kv_cache_groups[source_id].layer_names)
    for wrapper in model.modules():
        # DSA on NVIDIA fuses its Main KV write into the projection at the
        # beginning of the MLA module. Other MLA models own it in a wrapper.
        layer = (
            wrapper.mla_attn
            if isinstance(wrapper, MultiHeadLatentAttentionWrapper)
            else wrapper
        )
        if isinstance(layer, MLAAttention) and layer.layer_name in expected:
            name = layer.layer_name
            layer.layerwise_worker.register_layer(wrapper, name)
            expected.remove(name)
    if expected:
        raise ValueError(
            f"Layerwise HiSparse requires MLA wrappers for {sorted(expected)}"
        )


def copy_descriptors(
    dma: HiSparseDMA,
    host: torch.Tensor,
    gpu: torch.Tensor,
    spans: np.ndarray,
    to_gpu: bool,
) -> tuple[DMADescriptors, int]:
    """CPU DMA pointer lists for MLA views, including padded kernel pages."""
    row_bytes = gpu.shape[-1] * gpu.element_size()
    if any(
        cache.stride(-1) != 1
        or cache.stride(-2) * cache.element_size() != row_bytes
        or cache.shape[-1] * cache.element_size() != row_bytes
        for cache in (host, gpu)
    ):
        raise ValueError("Layerwise HiSparse needs contiguous Main KV rows")

    # Split only if a span would cross padding inside a scheduler page.
    for col, cache in enumerate((host, gpu)):
        rows = cache.shape[-2]
        if cache.stride(0) == rows * cache.stride(-2) or not np.any(
            spans[:, col] % rows + spans[:, 2] > rows
        ):
            continue
        pieces = []
        for h, g, count in spans:
            while count:
                take = min(count, rows - (h if col == 0 else g) % rows)
                pieces.append((h, g, take))
                h, g, count = h + take, g + take, count - take
        spans = np.asarray(pieces, dtype=np.int64)

    count = len(spans)
    descriptors = dma.acquire(count)
    src = descriptors.src_np[:count]
    dst = descriptors.dst_np[:count]
    host_ptrs, gpu_ptrs = (src, dst) if to_gpu else (dst, src)
    for col, (cache, ptrs) in enumerate(((host, host_ptrs), (gpu, gpu_ptrs))):
        block, offset = np.divmod(spans[:, col], cache.shape[-2])
        ptrs[:] = cache.data_ptr() + cache.element_size() * (
            block * cache.stride(0) + offset * cache.stride(-2)
        )
    descriptors.sizes_np[:count] = spans[:, 2] * row_bytes
    return descriptors, count


class LayerwiseWorker:
    def __init__(self, config: VllmConfig, cache_config: KVCacheConfig):
        self.config = config
        self.cache_config = cache_config
        self.host_pool = get_hisparse_host_pool(config, cache_config)
        device = torch.accelerator.current_stream().device
        self.load = torch.Stream(device=device)
        self.save = torch.Stream(device=device)
        self.dma = HiSparseDMA(get_hisparse_backend())
        self.closed = False
        self.step: LayerwiseStep | None = None
        self.host: dict[str, torch.Tensor] = {}
        self.slots: dict[str, torch.Tensor] = {}
        self.rings: dict[int, list[str]] = {}
        self.following: dict[str, str] = {}
        self.ready: dict[str, torch.Event] = {}
        self.finished: set[str] = set()
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.stats = HiSparseKVConnectorStats()
        self.bytes_per_row = 0
        self.completed: list[int] = []
        self.copied: list[int] = []
        self.writer = (
            not cache_config.hisparse_shared_host_pool
            or get_tensor_model_parallel_rank() == 0
        )

    def register_kv_caches(self, caches: dict[str, torch.Tensor]) -> None:
        (source_id,) = self.cache_config.host_group_ids
        ctx = self.config.compilation_config.static_forward_context
        for name in self.cache_config.kv_cache_groups[source_id].layer_names:
            self.host[name] = caches[name].squeeze(1)
            self.slots[name] = ctx[name].kv_cache
            self.bytes_per_row += (
                self.slots[name].shape[-1] * self.slots[name].element_size()
            )
            ctx[name].layerwise_worker = self

    def register_layer(self, wrapper: torch.nn.Module, name: str) -> None:
        # Module traversal follows decoder execution order. Each ring is a
        # single physical allocation reused by the layers in that order.
        ring = self.rings.setdefault(self.slots[name].data_ptr(), [])
        if ring:
            self.following[ring[-1]] = name
        ring.append(name)
        self.hooks.append(wrapper.register_forward_pre_hook(partial(self.before, name)))
        self.hooks.append(wrapper.register_forward_hook(partial(self.after, name)))

    def start(self, meta: "HiSparseConnectorMetadata") -> None:
        assert self.step is None and meta.layerwise is not None
        if meta.host_block_copies:
            if self.writer:
                num_blocks = self.cache_config.hisparse_host_num_blocks
                assert num_blocks is not None
                copy_kv_cache_blocks_inplace(
                    self.host.values(), num_blocks, meta.host_block_copies
                )
            if self.cache_config.hisparse_shared_host_pool:
                get_tp_group().barrier()
            self.copied.extend(c.dst_block_id for c in meta.host_block_copies)
        if not meta.layerwise.writes:
            return
        self.step = meta.layerwise
        self.read_spans = np.asarray(self.step.reads, dtype=np.int64).reshape(-1, 3)
        self.write_spans = np.asarray(self.step.writes, dtype=np.int64).reshape(-1, 3)
        self.load.wait_stream(torch.accelerator.current_stream())
        for ring in self.rings.values():
            self.prefetch(ring[0])

    def transfer(
        self, name: str, spans: np.ndarray, stream: torch.Stream, *, to_gpu: bool
    ) -> torch.Event:
        if not spans.size:
            return stream.record_event()
        descriptors, count = copy_descriptors(
            self.dma, self.host[name], self.slots[name], spans, to_gpu
        )
        return self.dma.submit(stream, descriptors, count)

    def prefetch(self, name: str) -> None:
        self.ready[name] = self.transfer(name, self.read_spans, self.load, to_gpu=True)

    def before(self, name: str, module, args) -> None:
        if self.step is not None:
            torch.accelerator.current_stream().wait_event(self.ready.pop(name))

    def after(self, name: str, module, args, output) -> None:
        if self.step is None:
            return
        self.save.wait_stream(torch.accelerator.current_stream())
        if self.writer:
            released = self.transfer(name, self.write_spans, self.save, to_gpu=False)
        else:
            released = self.save.record_event()
        if following := self.following.get(name):
            self.load.wait_event(released)
            self.prefetch(following)
        self.finished.add(name)

    def finish_forward(self) -> None:
        if self.step is None:
            return
        if self.finished != self.host.keys():
            raise RuntimeError(
                "Incomplete layerwise forward; host KV cannot be published"
            )
        self.save.synchronize()
        if self.cache_config.hisparse_shared_host_pool:
            get_tp_group().barrier()
        self.stats.data["prefill_h2d_bytes"].append(
            int(self.read_spans[:, 2].sum()) * self.bytes_per_row
        )
        if self.writer:
            self.stats.data["prefill_d2h_bytes"].append(
                int(self.write_spans[:, 2].sum()) * self.bytes_per_row
            )
        self.completed.extend(self.step.sealed_pages)
        self.dma.release_completed()
        self.finished.clear()
        self.step = None

    def get_kv_connector_stats(self) -> HiSparseKVConnectorStats | None:
        if self.stats.is_empty():
            return None
        stats, self.stats = self.stats, HiSparseKVConnectorStats()
        return stats

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.load.synchronize()
        self.save.synchronize()
        for hook in self.hooks:
            hook.remove()
        for name in self.host:
            layer = self.config.compilation_config.static_forward_context.get(name)
            if layer is not None and getattr(layer, "layerwise_worker", None) is self:
                del layer.layerwise_worker
        release_pinned_state(self.host_pool.pinned, self.host_pool.shared_region)
        drop_hisparse_host_pool(self.config)
