# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.musa_connectors import (
    VLLMPagedMemLayerwiseMUSAConnector,
    VLLMPagedMemMUSAConnectorV2,
)
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from tests.v1.utils import (
    check_paged_kv_cache_equal,
    generate_kv_cache_paged_list_tensors,
)
import lmcache.c_ops as lmc_ops


def _skip_if_no_musa() -> None:
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("torch.musa is not available")


def _make_unique_slot_mapping(
    *, total_slots: int, num_tokens: int, device: torch.device
) -> torch.Tensor:
    return torch.randperm(total_slots, device=device, dtype=torch.int64)[:num_tokens]


def _pack_slot_mapping(
    slot_mapping: torch.Tensor, starts: list[int], ends: list[int]
) -> torch.Tensor:
    return torch.cat(
        [slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)],
        dim=0,
    )


def _make_metadata(
    *,
    model_name: str,
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
) -> LMCacheMetadata:
    """Create metadata for a synthetic vLLM MUSA KV cache."""
    return LMCacheMetadata(
        model_name=model_name,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(num_layers, 2, num_tokens, num_heads, head_size),
    )


def _patch_musa_connector_attrs(
    monkeypatch: pytest.MonkeyPatch,
    conn: VLLMPagedMemMUSAConnectorV2,
    *,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    engine_kv_format: lmc_ops.EngineKVFormat = (
        lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    ),
) -> None:
    """Patch connector layout discovery so transfer logic can run on CPU."""

    def _initialize_attributes(_kv_caches: list[torch.Tensor]) -> None:
        conn.device = torch.device("cpu")
        conn.engine_kv_format = engine_kv_format
        conn.num_layers = num_layers
        conn.num_blocks = num_blocks
        conn.block_size = block_size
        conn.page_buffer_size = num_blocks * block_size
        conn.hidden_dim_size = num_heads * head_size
        conn.head_size = head_size
        conn.use_mla = False
        conn.dtype = torch.bfloat16
        conn.num_heads = num_heads
        conn._attributes_initialized = True

    monkeypatch.setattr(conn, "_initialize_attributes", _initialize_attributes)


class _FakeMUSAStream:
    """Minimal stream object used by CPU-only MUSA generator tests."""

    def wait_stream(self, _stream: object) -> None:
        return None


class _FakeMUSA:
    """Minimal ``torch.musa`` facade for CPU-only generator contract tests."""

    def Stream(self) -> _FakeMUSAStream:
        return _FakeMUSAStream()

    def current_stream(self) -> _FakeMUSAStream:
        return _FakeMUSAStream()

    def stream(self, _stream: object) -> Any:
        return nullcontext()


def test_musa_connector_to_gpu_skips_vllm_cached_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``vllm_cached_tokens`` skips prefix slots before torch ``index_copy_``."""
    num_layers = 1
    num_blocks = 3
    block_size = 4
    num_heads = 1
    head_size = 2
    hidden_dim = num_heads * head_size
    start = 4
    end = 8
    vllm_cached_tokens = 6
    skipped = vllm_cached_tokens - start

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_test_vllm_cached_tokens",
            num_layers=num_layers,
            num_tokens=end - start,
            num_heads=num_heads,
            head_size=head_size,
        ),
    )
    _patch_musa_connector_attrs(
        monkeypatch,
        conn,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
    )

    kvcaches_dst = [
        torch.zeros(
            2,
            num_blocks,
            block_size,
            num_heads,
            head_size,
            dtype=torch.bfloat16,
        )
    ]
    memory_tensor = torch.arange(
        2 * num_layers * (end - start) * hidden_dim,
        dtype=torch.float32,
    ).reshape(2, num_layers, end - start, hidden_dim)
    memory_tensor = memory_tensor.to(torch.bfloat16)
    memory_obj = cast(
        MemoryObj,
        SimpleNamespace(
            tensor=memory_tensor,
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        ),
    )
    slot_mapping = torch.tensor(
        [-1, -1, -1, -1, -1, -1, 6, 7],
        dtype=torch.long,
    )

    conn.to_gpu(
        memory_obj,
        start=start,
        end=end,
        slot_mapping=slot_mapping,
        kvcaches=kvcaches_dst,
        vllm_cached_tokens=vllm_cached_tokens,
    )

    flat_k = kvcaches_dst[0][0].reshape(num_blocks * block_size, hidden_dim)
    flat_v = kvcaches_dst[0][1].reshape(num_blocks * block_size, hidden_dim)
    target_slots = slot_mapping[vllm_cached_tokens:end]

    assert torch.equal(flat_k[target_slots], memory_tensor[0, 0, skipped:])
    assert torch.equal(flat_v[target_slots], memory_tensor[1, 0, skipped:])
    assert torch.count_nonzero(flat_k[-1]) == 0
    assert torch.count_nonzero(flat_v[-1]) == 0


def test_musa_connector_rejects_unsupported_kv_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported MUSA KV layouts fail with a connector-level message."""
    num_layers = 1
    num_blocks = 3
    block_size = 4
    num_heads = 1
    head_size = 2
    hidden_dim = num_heads * head_size

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_test_unsupported_layout",
            num_layers=num_layers,
            num_tokens=block_size,
            num_heads=num_heads,
            head_size=head_size,
        ),
    )
    _patch_musa_connector_attrs(
        monkeypatch,
        conn,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        engine_kv_format=lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
    )

    kvcaches_dst = [
        torch.zeros(
            num_blocks,
            2,
            block_size,
            num_heads,
            head_size,
            dtype=torch.bfloat16,
        )
    ]
    memory_obj = cast(
        MemoryObj,
        SimpleNamespace(
            tensor=torch.zeros(
                2,
                num_layers,
                block_size,
                hidden_dim,
                dtype=torch.bfloat16,
            ),
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        ),
    )

    with pytest.raises(ValueError, match="VLLMPagedMemMUSAConnectorV2 supports"):
        conn.to_gpu(
            memory_obj,
            start=0,
            end=block_size,
            slot_mapping=torch.arange(block_size, dtype=torch.long),
            kvcaches=kvcaches_dst,
        )


def test_musa_connector_to_gpu_uses_native_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connector skips torch scatter when the native adapter completes."""
    num_layers = 1
    num_blocks = 2
    block_size = 4
    num_heads = 1
    head_size = 2
    hidden_dim = num_heads * head_size
    calls: list[str] = []

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_native_to_gpu",
            num_layers=num_layers,
            num_tokens=block_size,
            num_heads=num_heads,
            head_size=head_size,
        ),
    )
    _patch_musa_connector_attrs(
        monkeypatch,
        conn,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
    )

    def _native_to_gpu(**_kwargs: object) -> bool:
        calls.append("native_to_gpu")
        return True

    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.musa_connectors.try_native_to_gpu",
        _native_to_gpu,
    )

    memory_obj = cast(
        MemoryObj,
        SimpleNamespace(
            tensor=torch.ones(2, num_layers, block_size, hidden_dim),
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        ),
    )
    kvcaches_dst = [
        torch.zeros(2, num_blocks, block_size, num_heads, head_size),
    ]

    conn.to_gpu(
        memory_obj,
        start=0,
        end=block_size,
        slot_mapping=torch.arange(block_size),
        kvcaches=kvcaches_dst,
    )

    assert calls == ["native_to_gpu"]
    assert torch.count_nonzero(kvcaches_dst[0]) == 0


def test_musa_connector_to_gpu_falls_back_when_native_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connector keeps Stage1 torch behavior when native returns False."""
    num_layers = 1
    num_blocks = 2
    block_size = 4
    num_heads = 1
    head_size = 2
    hidden_dim = num_heads * head_size

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_native_fallback",
            num_layers=num_layers,
            num_tokens=block_size,
            num_heads=num_heads,
            head_size=head_size,
        ),
    )
    _patch_musa_connector_attrs(
        monkeypatch,
        conn,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
    )
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.musa_connectors.try_native_to_gpu",
        lambda **_kwargs: False,
    )

    memory_tensor = torch.arange(
        2 * num_layers * block_size * hidden_dim,
        dtype=torch.float32,
    ).reshape(2, num_layers, block_size, hidden_dim)
    memory_obj = cast(
        MemoryObj,
        SimpleNamespace(
            tensor=memory_tensor,
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        ),
    )
    kvcaches_dst = [
        torch.zeros(2, num_blocks, block_size, num_heads, head_size),
    ]

    conn.to_gpu(
        memory_obj,
        start=0,
        end=block_size,
        slot_mapping=torch.arange(block_size),
        kvcaches=kvcaches_dst,
    )

    flat_k = kvcaches_dst[0][0].reshape(num_blocks * block_size, hidden_dim)
    assert torch.equal(flat_k[:block_size], memory_tensor[0, 0])


def test_vllm_layerwise_musa_connector_constructs_without_torch_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layerwise construction must not require a MUSA-enabled torch build."""
    monkeypatch.delattr(torch, "musa", raising=False)

    conn = VLLMPagedMemLayerwiseMUSAConnector(
        hidden_dim_size=64,
        num_layers=2,
        use_musa=False,
        chunk_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert tuple(conn.get_shape(num_tokens=3)) == (3, 2, 64)


def test_musa_connector_from_gpu_uses_native_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connector skips torch gather when the native adapter completes."""
    num_layers = 1
    num_blocks = 2
    block_size = 4
    num_heads = 1
    head_size = 2
    hidden_dim = num_heads * head_size
    calls: list[str] = []

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_native_from_gpu",
            num_layers=num_layers,
            num_tokens=block_size,
            num_heads=num_heads,
            head_size=head_size,
        ),
    )
    _patch_musa_connector_attrs(
        monkeypatch,
        conn,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
    )

    def _native_from_gpu(**_kwargs: object) -> bool:
        calls.append("native_from_gpu")
        return True

    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.musa_connectors.try_native_from_gpu",
        _native_from_gpu,
    )

    memory_obj = cast(
        MemoryObj,
        SimpleNamespace(
            tensor=torch.zeros(2, num_layers, block_size, hidden_dim),
            metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD),
        ),
    )
    kvcaches_src = [
        torch.ones(2, num_blocks, block_size, num_heads, head_size),
    ]

    conn.from_gpu(
        memory_obj,
        start=0,
        end=block_size,
        slot_mapping=torch.arange(block_size),
        kvcaches=kvcaches_src,
    )

    assert calls == ["native_from_gpu"]
    assert torch.count_nonzero(memory_obj.tensor) == 0


def test_vllm_layerwise_musa_batched_to_gpu_accepts_empty_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty layerwise batches keep the generator contract without cat([])."""
    monkeypatch.setattr(torch, "musa", _FakeMUSA(), raising=False)

    num_layers = 2
    conn = VLLMPagedMemLayerwiseMUSAConnector(
        hidden_dim_size=4,
        num_layers=num_layers,
        use_musa=False,
        chunk_size=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    consumer = conn.batched_to_gpu(
        starts=[],
        ends=[],
        slot_mapping=torch.empty(0, dtype=torch.long),
        sync=True,
        kvcaches=[torch.empty(0) for _ in range(num_layers)],
    )

    for _ in range(num_layers + 2):
        next(consumer)
    with pytest.raises(StopIteration):
        next(consumer)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_musa_connector_roundtrip_non_layerwise(use_gpu: bool):
    """Round-trip from_gpu -> to_gpu on the non-layerwise MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 4
    block_size = 16
    head_size = 64
    num_tokens = 32

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )

    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobj = pin_alloc.allocate(
        torch.Size([2, num_layers, num_tokens, hidden_dim_actual]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
    )

    meta = LMCacheMetadata(
        model_name="musa_test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            num_layers,
            2,
            num_tokens,
            num_heads_actual,
            head_size_actual,
        ),
    )
    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        meta,
        use_gpu=use_gpu,
        device=device,
    )

    try:
        conn.from_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for t in kvcaches_dst:
            t.zero_()

        conn.to_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches_dst,
        )

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        memobj.ref_count_down()
        pin_alloc.close()


def test_musa_connector_to_gpu_accepts_cpu_slot_mapping() -> None:
    """Round-trip with CPU ``slot_mapping`` and MUSA KV cache tensors."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 4
    block_size = 16
    head_size = 64
    num_tokens = 32

    kvcaches_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )
    _, _, num_heads_actual, head_size_actual = kvcaches_src[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    slot_mapping_cpu = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=num_tokens,
        device=torch.device("cpu"),
    )
    slot_mapping_musa = slot_mapping_cpu.to(device)

    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(
            model_name="musa_test_cpu_slot_mapping",
            num_layers=num_layers,
            num_tokens=num_tokens,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        ),
        use_gpu=False,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobj = pin_alloc.allocate(
        torch.Size([2, num_layers, num_tokens, hidden_dim_actual]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
    )

    try:
        conn.from_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping_cpu,
            kvcaches=kvcaches_src,
        )

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        conn.to_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping_cpu,
            kvcaches=kvcaches_dst,
        )

        check_paged_kv_cache_equal(
            kvcaches_src,
            kvcaches_dst,
            slot_mapping_musa,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        memobj.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_gpu", [False, True])
def test_musa_connector_roundtrip_layerwise(use_gpu: bool):
    """Round-trip batched_from_gpu -> batched_to_gpu on layerwise MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 4
    num_blocks = 8
    block_size = 16
    head_size = 64
    num_tokens = 64

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )

    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    meta = LMCacheMetadata(
        model_name="musa_test_layerwise",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            num_layers,
            2,
            num_tokens,
            num_heads_actual,
            head_size_actual,
        ),
    )

    conn = VLLMPagedMemLayerwiseMUSAConnector.from_metadata(
        meta,
        use_musa=use_gpu,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 256)

    memobjs_by_layer = [
        [
            pin_alloc.allocate(
                torch.Size([num_tokens, 2, hidden_dim_actual]),
                torch.bfloat16,
                MemoryFormat.KV_T2D,
            )
        ]
        for _ in range(num_layers)
    ]

    try:
        gen = conn.batched_from_gpu(
            memobjs_by_layer,
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches,
        )

        for _ in range(num_layers + 1):
            next(gen)

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for t in kvcaches_dst:
            t.zero_()

        gen2 = conn.batched_to_gpu(
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches_dst,
        )

        next(gen2)
        for layer_id in range(num_layers):
            gen2.send(memobjs_by_layer[layer_id])

        next(gen2)

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for layer in memobjs_by_layer:
            for m in layer:
                m.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_gpu", [False, True])
def test_musa_connector_roundtrip_non_layerwise_multi_chunk(
    use_gpu: bool,
) -> None:
    """Non-layerwise multi-chunk round-trip on MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 2
    num_blocks = 6
    block_size = 8
    head_size = 64
    total_tokens = 32

    starts = [0, 7, 19]
    ends = [4, 13, 25]

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )
    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )
    packed_slot_mapping = _pack_slot_mapping(slot_mapping, starts, ends)

    meta = LMCacheMetadata(
        model_name="musa_test_non_layerwise_multi_chunk",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            num_layers,
            2,
            total_tokens,
            num_heads_actual,
            head_size_actual,
        ),
    )
    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        meta,
        use_gpu=use_gpu,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobjs = []
    try:
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            memobj = pin_alloc.allocate(
                torch.Size([2, num_layers, n, hidden_dim_actual]),
                torch.bfloat16,
                MemoryFormat.KV_2LTD,
            )
            conn.from_gpu(
                memobj,
                start=s,
                end=e,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches,
            )
            memobjs.append((s, e, memobj))

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        for s, e, memobj in memobjs:
            conn.to_gpu(
                memobj,
                start=s,
                end=e,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches_dst,
            )

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            packed_slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for _, _, memobj in memobjs:
            memobj.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_musa", [False, True])
def test_musa_connector_roundtrip_layerwise_multi_chunk(
    use_musa: bool,
) -> None:
    """Layerwise multi-chunk round-trip on MUSA connector."""
    _skip_if_no_musa()
    device = torch.device("musa:0")

    num_layers = 4
    num_blocks = 8
    block_size = 8
    head_size = 64
    total_tokens = 40

    starts = [0, 9, 21]
    ends = [5, 15, 30]

    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=num_layers,
        head_size=head_size,
        device=device,
    )

    _, _, num_heads_actual, head_size_actual = kvcaches[0][0].shape
    hidden_dim_actual = num_heads_actual * head_size_actual

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )
    packed_slot_mapping = _pack_slot_mapping(slot_mapping, starts, ends)

    meta = LMCacheMetadata(
        model_name="musa_test_layerwise_multi_chunk",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            num_layers,
            2,
            total_tokens,
            num_heads_actual,
            head_size_actual,
        ),
    )
    conn = VLLMPagedMemLayerwiseMUSAConnector.from_metadata(
        meta,
        use_musa=use_musa,
        device=device,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 128)
    memobjs_by_layer = []
    for _ in range(num_layers):
        per_layer = []
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            per_layer.append(
                pin_alloc.allocate(
                    torch.Size([n, 2, hidden_dim_actual]),
                    torch.bfloat16,
                    MemoryFormat.KV_T2D,
                )
            )
        memobjs_by_layer.append(per_layer)

    try:
        producer = conn.batched_from_gpu(
            memobjs_by_layer,
            starts=starts,
            ends=ends,
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches,
        )
        for _ in range(num_layers + 1):
            next(producer)

        if use_musa:
            assert conn.gpu_buffer_allocator is not None
        else:
            assert conn.gpu_buffer_allocator is None

        kvcaches_dst = generate_kv_cache_paged_list_tensors(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            head_size=head_size_actual,
            device=device,
        )
        for layer in kvcaches_dst:
            layer.zero_()

        consumer = conn.batched_to_gpu(
            starts=starts,
            ends=ends,
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches_dst,
        )
        next(consumer)
        for layer_id in range(num_layers):
            consumer.send(memobjs_by_layer[layer_id])
        next(consumer)

        check_paged_kv_cache_equal(
            kvcaches,
            kvcaches_dst,
            packed_slot_mapping,
            num_heads=num_heads_actual,
            head_size=head_size_actual,
        )
    finally:
        for layer in memobjs_by_layer:
            for memobj in layer:
                memobj.ref_count_down()
        pin_alloc.close()
