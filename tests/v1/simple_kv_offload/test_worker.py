# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side unit tests for SimpleCPUOffloadConnector.

Covers the GPU->CPU store cross-stream synchronization: the store copy must be
ordered after the compute stream that writes the KV blocks, otherwise it can
read partially written / stale blocks and silently corrupt the CPU cache.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

from tests.v1.attention.utils import dense_kv_cache_tensor, dense_kv_cache_views
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    is_kv_cache_spec_uniform,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    build_params,
    pin_tensor,
)
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker
from vllm.v1.worker.utils import allocate_kv_cache

NUM_BLOCKS = 64
BLOCK_BYTES = 4096
ITERS = 30
# Keep the compute stream busy so the KV write lands late; this makes the
# store-vs-compute race deterministic instead of timing-dependent.
SLEEP_CYCLES = 50_000_000


def _make_backend() -> tuple[DmaCopyBackend, torch.Tensor, torch.Tensor]:
    gpu = {"k": torch.zeros((NUM_BLOCKS, BLOCK_BYTES), dtype=torch.int8, device="cuda")}
    cpu = {"k": torch.zeros((NUM_BLOCKS, BLOCK_BYTES), dtype=torch.int8, device="cpu")}
    pin_tensor(cpu["k"])
    low_pri, _ = torch.cuda.Stream.priority_range()
    backend = DmaCopyBackend()
    backend.init(
        gpu,
        cpu,
        gpu["k"].device,
        torch.cuda.Stream(priority=low_pri),
        torch.cuda.Stream(priority=low_pri),
    )
    return backend, gpu["k"], cpu["k"]


def _drive_store(
    backend: DmaCopyBackend,
    gpu: torch.Tensor,
    cpu: torch.Tensor,
    *,
    with_barrier: bool,
) -> int:
    """Run ITERS store cycles; return how many landed corrupted in the CPU pool.

    Each cycle writes a unique value on a compute stream (after a deliberate
    delay) and then issues the GPU->CPU store. The store is issued *after* the
    write in host program order, mirroring the connector's deferred-store
    assumption. Only the compute-done event creates a real device-side
    happens-before edge.
    """
    block_ids = list(range(gpu.shape[0]))
    compute_stream = torch.cuda.Stream()
    corrupt = 0
    for it in range(ITERS):
        val = (it % 126) + 1  # 1..126; distinct from the zero-initialized pool
        with torch.cuda.stream(compute_stream):
            torch.cuda._sleep(SLEEP_CYCLES)
            gpu.fill_(val)

        wait_event = None
        if with_barrier:
            wait_event = torch.Event()
            wait_event.record(compute_stream)

        store_events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            block_ids,
            block_ids,
            is_store=True,
            event_idx=it,
            events_list=store_events,
            wait_event=wait_event,
        )

        deadline = time.time() + 10.0
        while not store_events and time.time() < deadline:
            time.sleep(0.0005)
        assert store_events, "background copy was never enqueued"
        store_events[0][1].synchronize()

        if int((cpu[:, 0].to(torch.int32) != val).sum().item()):
            corrupt += 1

    # Drain the compute stream before returning: in the no-barrier control
    # phase the store never waits on compute, so the host loop runs far ahead
    # and leaves a backlog of sleep+fill kernels in flight. Without this, the
    # leftover control-phase fills race the barrier phase's fill->copy window
    # on the shared gpu tensor and flakily corrupt one iteration.
    compute_stream.synchronize()
    return corrupt


def test_store_orders_after_compute_write():
    """The store must wait for the compute event; without it, it races.

    Asserts both directions so the test is self-validating: the no-barrier
    control must actually corrupt (proving the race window is exercised), and
    the fixed path with the compute-done event must be clean.
    """
    backend, gpu, cpu = _make_backend()
    try:
        control = _drive_store(backend, gpu, cpu, with_barrier=False)
        fixed = _drive_store(backend, gpu, cpu, with_barrier=True)
    finally:
        backend.shutdown()

    assert control > 0, (
        "no-barrier store did not race the compute write; the test no longer "
        "exercises the hazard it is meant to guard"
    )
    assert fixed == 0, f"store raced compute even with the barrier: {fixed} corrupt"


class _RecordingBackend:
    """Captures launch_copy calls without touching the GPU."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def launch_copy(
        self,
        src_blocks,
        dst_blocks,
        is_store,
        event_idx,
        events_list,
        wait_event=None,
    ) -> None:
        self.calls.append({"is_store": is_store, "wait_event": wait_event})


def test_transfer_hooks_pass_wait_event_for_store_only():
    """wait_for_save gates stores on a compute-done event; start_load_kv does not."""
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    recording = _RecordingBackend()
    worker._backend = recording
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        load_event=0,
        load_gpu_blocks=[0],
        load_cpu_blocks=[0],
        store_event=1,
        store_gpu_blocks=[1],
        store_cpu_blocks=[1],
    )

    worker.start_load_kv()
    worker.wait_for_save()

    store_calls = [c for c in recording.calls if c["is_store"]]
    load_calls = [c for c in recording.calls if not c["is_store"]]
    assert len(store_calls) == 1
    assert len(load_calls) == 1
    assert isinstance(store_calls[0]["wait_event"], torch.Event)
    assert load_calls[0]["wait_event"] is None


def test_build_params_src_access_order():
    """build_params defaults to ANY and honors an explicit STREAM override."""
    gpu = {"k": torch.zeros((4, 64), dtype=torch.int8, device="cuda")}
    cpu = {"k": torch.zeros((4, 64), dtype=torch.int8, device="cpu")}
    stream = torch.cuda.Stream()

    default = build_params(gpu, cpu, stream)
    assert default.attrs.srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_ANY

    ordered = build_params(
        gpu, cpu, stream, src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
    )
    assert ordered.attrs.srcAccessOrder == CU_MEMCPY_SRC_ACCESS_ORDER_STREAM


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_register_shared_kv_cache_storage(monkeypatch, layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float16,
    )
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    layer_names = [f"layer.{i}" for i in range(num_layers)]
    cache_config = MagicMock(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            dense_kv_cache_tensor(
                raw, spec, num_blocks, num_layers, layout, layer_names
            )
        ],
    )
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=cache_config,
        cpu_capacity_bytes=raw.nbytes,
    )
    worker._backend = MagicMock()
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)

    worker.register_kv_caches(
        {f"layer.{layer_idx}": cache for layer_idx, cache in enumerate(caches)}
    )

    assert worker.gpu_kv_caches is not None
    if layout.is_layer_compact and not layout.is_block_compact:
        expected_regions = num_layers * spec.num_heads
        expected_block_bytes = spec.page_size_bytes // spec.num_heads
    elif layout.is_layer_compact:
        expected_regions = num_layers
        expected_block_bytes = spec.page_size_bytes
    else:
        expected_regions = 1
        expected_block_bytes = spec.page_size_bytes * num_layers
    assert len(worker.gpu_kv_caches) == expected_regions
    assert {cache.shape for cache in worker.gpu_kv_caches.values()} == {
        (num_blocks, expected_block_bytes)
    }


def test_register_kv_cache_storage_with_trailing_padding(monkeypatch):
    num_blocks = 4
    block_bytes = 32
    cache_bytes = num_blocks * block_bytes
    raw = torch.zeros(4096, dtype=torch.int8, device="cuda")
    cache = raw[:cache_bytes].view(num_blocks, block_bytes)
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=MagicMock(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=cache_bytes,
                    layers=["layer.0"],
                    layer_stride=cache_bytes,
                    block_stride=block_bytes,
                )
            ],
        ),
        cpu_capacity_bytes=cache_bytes,
    )
    worker._backend = MagicMock()
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)

    worker.register_kv_caches({"layer.0": cache})

    assert worker.gpu_kv_caches is not None
    assert list(worker.gpu_kv_caches) == ["layer.0"]
    assert worker.gpu_kv_caches["layer.0"].shape == (num_blocks, block_bytes)


def test_register_separate_kv_head_groups(monkeypatch):
    # LHBNC hoists the K/V head groups outside the block dim, so each layer's
    # blocks are registered as one region per group (K, V).
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float16,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 2,
    )
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device="cuda",
    )
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    layer_names = [f"layer.{i}" for i in range(num_layers)]
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=MagicMock(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                dense_kv_cache_tensor(
                    raw, spec, num_blocks, num_layers, layout, layer_names
                )
            ],
        ),
        cpu_capacity_bytes=raw.nbytes,
    )
    worker._backend = MagicMock()
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)

    worker.register_kv_caches(
        {f"layer.{layer_idx}": cache for layer_idx, cache in enumerate(caches)}
    )

    assert worker.gpu_kv_caches is not None
    assert len(worker.gpu_kv_caches) == num_layers * spec.num_heads
    per_group_block_bytes = (
        spec.num_kv_heads * spec.block_size * spec.head_size * spec.dtype.itemsize
    )
    assert {cache.shape for cache in worker.gpu_kv_caches.values()} == {
        (num_blocks, per_group_block_bytes)
    }


def _dsa_specs(num_layers: int, block_size: int) -> dict[str, KVCacheSpec]:
    """A DSA model's per-layer specs (DeepSeek-V3.2, GLM-5.2).

    Each decoder layer contributes an MLA latent cache and the indexer's key
    cache, whose pages differ in size (``MLAAttentionSpec`` off ``head_size``:
    512 + 64 for the latent, 128 fp8 keys + one fp32 scale per 128 elements for
    the indexer). Both are ``MLAAttentionSpec`` with the same block size, so the
    hybrid allocator puts them in one ``UniformTypeKVCacheSpecs`` cache group.
    See ``DeepseekV32IndexerCache.get_kv_cache_spec`` and
    ``MLAAttention.get_kv_cache_spec``.
    """
    specs: dict[str, KVCacheSpec] = {}
    for i in range(num_layers):
        specs[f"model.layers.{i}.self_attn.attn"] = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=512 + 64,
            dtype=torch.uint8,
            cache_dtype_str="fp8",
        )
        specs[f"model.layers.{i}.self_attn.indexer.k_cache"] = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128 + 128 // 128 * 4,
            dtype=torch.uint8,
        )
    return specs


def test_register_mixed_page_sizes_in_one_cache_group(monkeypatch):
    """Sparse-MLA models mix page sizes inside one cache group.

    The MLA latent and indexer key caches of a layer have different page sizes,
    so the allocation has no single per-layer block size to reinterpret it with:
    ``num_layers * (mla_page + indexer_page)`` is not a multiple of either page.
    Registration must derive each region from the placement metadata instead.
    """
    num_layers = 4
    block_size = 64
    specs = _dsa_specs(num_layers, block_size)
    # Differing head sizes make the specs non-identical but same-type, which is
    # what lands both caches of a layer in one group.
    assert not is_kv_cache_spec_uniform(specs)
    assert UniformTypeKVCacheSpecs.is_uniform_type(specs)
    group = KVCacheGroupSpec(
        list(specs),
        UniformTypeKVCacheSpecs(block_size=block_size, kv_cache_specs=specs),
    )

    # DEEPSEEK_V32_INDEXER declares no supported layouts, so resolution lands on
    # the default preference: layer-outermost, one region per layer.
    layout = KVCacheLayout.LBNHC
    vllm_config = MagicMock()
    vllm_config.cache_config = CacheConfig()
    vllm_config.cache_config.kv_cache_layout = layout.name
    vllm_config.cache_config.num_gpu_blocks_override = None

    pages = [spec.page_size_bytes for spec in specs.values()]
    num_blocks = 4
    kv_cache_config = get_kv_cache_config_from_groups(
        vllm_config, [group], sum(pages) * num_blocks
    )
    assert kv_cache_config.num_blocks == num_blocks
    assert len(set(pages)) > 1, "the mixed page sizes are what this test covers"

    kv_caches = allocate_kv_cache(kv_cache_config, torch.device("cuda"), layout)
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=sum(pages) * num_blocks,
    )
    worker._backend = MagicMock()
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)

    worker.register_kv_caches(kv_caches)

    assert worker.gpu_kv_caches is not None
    # One region per layer, each with that layer's own page as its block stride.
    assert sorted(worker.gpu_kv_caches) == sorted(specs)
    assert {name: cache.shape[1] for name, cache in worker.gpu_kv_caches.items()} == {
        name: spec.page_size_bytes for name, spec in specs.items()
    }
    assert all(cache.shape[0] == num_blocks for cache in worker.gpu_kv_caches.values())

    # Every registered region must alias the bytes the model writes through, or
    # offloaded blocks would be copied from the wrong place.
    for name, cache in kv_caches.items():
        region = worker.gpu_kv_caches[name]
        assert region.data_ptr() == cache.data_ptr()
        assert region.stride(0) == specs[name].page_size_bytes


@pytest.mark.parametrize("rank_blocks", [1, 5])
def test_register_mixed_page_sizes_odd_block_counts(monkeypatch, rank_blocks):
    """Registration holds at block counts that are not powers of two.

    ``num_blocks`` comes out of ``available_memory // bytes_per_block``, so
    any integer is reachable. Registration derives regions from the config's
    own block count and strides, and the CPU pool rounds down from the same
    per-block byte total; none of it assumes anything about how the count
    was reached.
    """
    num_layers = 4
    block_size = 64
    specs = _dsa_specs(num_layers, block_size)
    group = KVCacheGroupSpec(
        list(specs),
        UniformTypeKVCacheSpecs(block_size=block_size, kv_cache_specs=specs),
    )
    layout = KVCacheLayout.LBNHC
    vllm_config = MagicMock()
    vllm_config.cache_config = CacheConfig()
    vllm_config.cache_config.kv_cache_layout = layout.name
    vllm_config.cache_config.num_gpu_blocks_override = None

    pages = [spec.page_size_bytes for spec in specs.values()]
    kv_cache_config = get_kv_cache_config_from_groups(
        vllm_config, [group], sum(pages) * rank_blocks
    )
    assert kv_cache_config.num_blocks == rank_blocks

    kv_caches = allocate_kv_cache(kv_cache_config, torch.device("cuda"), layout)
    worker = SimpleCPUOffloadWorker(
        vllm_config=None,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=sum(pages) * rank_blocks,
    )
    worker._backend = MagicMock()
    monkeypatch.setattr("vllm.v1.simple_kv_offload.worker.PIN_MEMORY", False)

    worker.register_kv_caches(kv_caches)

    assert worker.gpu_kv_caches is not None
    assert {n: c.shape for n, c in worker.gpu_kv_caches.items()} == {
        name: (rank_blocks, spec.page_size_bytes) for name, spec in specs.items()
    }
    assert worker.num_cpu_blocks == rank_blocks


def test_mixed_page_byte_placement_is_dcp_invariant():
    """DCP scales a block's token span, not its byte placement.

    ``get_kv_cache_config_from_groups`` never reads
    ``decode_context_parallel_size``, so every placement field registration
    derives regions from is identical at dcp=1 and dcp=2. What DCP scales is
    the block's token span, reported by ``resolve_kv_cache_block_sizes``;
    checking that it doubles proves DCP was actually in effect, or the
    identity above would be trivially true.
    """
    specs = _dsa_specs(4, block_size=128)

    placements = []
    for dcp_size in (1, 2):
        group = KVCacheGroupSpec(
            list(specs),
            UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs),
        )
        vllm_config = MagicMock()
        vllm_config.cache_config = CacheConfig()
        vllm_config.cache_config.block_size = 128
        vllm_config.cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name
        vllm_config.cache_config.num_gpu_blocks_override = None
        vllm_config.parallel_config.decode_context_parallel_size = dcp_size

        page_bytes = sum(spec.page_size_bytes for spec in specs.values())
        config = get_kv_cache_config_from_groups(vllm_config, [group], page_bytes * 4)
        scheduler_block_size, _ = resolve_kv_cache_block_sizes(config, vllm_config)
        assert scheduler_block_size == 128 * dcp_size
        placements.append(
            [
                (t.size, tuple(t.layers), t.layer_stride, t.block_stride, t.offset)
                for t in config.kv_cache_tensors
            ]
        )

    assert placements[0] == placements[1]
