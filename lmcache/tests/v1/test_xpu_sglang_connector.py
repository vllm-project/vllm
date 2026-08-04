# SPDX-License-Identifier: Apache-2.0
# tests/v1/test_xpu_sglang_connector.py
#
# Unit tests for SGLang XPU connectors (SGLangXPUConnector and
# SGLangLayerwiseXPUConnector). Mirrors the structure of
# test_xpu_connector.py which covers the vLLM XPU connectors.

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.gpu_connector import xpu_connectors
from lmcache.v1.gpu_connector.xpu_connectors import (
    SGLangLayerwiseXPUConnector,
    SGLangXPUConnector,
)
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from tests.v1.utils import (
    check_paged_kv_cache_equal_with_mla,
    check_sglang_paged_kv_cache_equal,
    generate_sglang_kv_cache_paged_list_tensors,
)

pytestmark = pytest.mark.skipif(
    torch_device_type != "xpu",
    reason="XPU-only tests",
)


def _current_xpu_device() -> torch.device:
    return torch.device(torch_device_type, torch.xpu.current_device())


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


def _check_kv_equal(
    kvcaches_src, kvcaches_dst, slot_mapping, num_heads, head_size, use_mla
):
    """Dispatch to the correct comparison helper based on MLA mode."""
    if use_mla:
        check_paged_kv_cache_equal_with_mla(
            kvcaches_src, kvcaches_dst, slot_mapping, head_size
        )
    else:
        check_sglang_paged_kv_cache_equal(
            kvcaches_src,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
        )


def _zero_kvcaches(kvcaches, use_mla):
    """Zero out all tensors in the kvcaches structure."""
    if use_mla:
        for t in kvcaches:
            t.zero_()
    else:
        for kv_list in kvcaches:
            for t in kv_list:
                t.zero_()


def _as_flat_sglang_mha(kvcaches):
    """Convert [[k_list], [v_list]] to flat [k0, ..., kN, v0, ..., vN]."""
    return list(kvcaches[0]) + list(kvcaches[1])


def _flat_to_nested_sglang_mha(kvcaches_flat, num_layers: int):
    return [kvcaches_flat[:num_layers], kvcaches_flat[num_layers:]]


# --------------------------------------------------------------------------- #
# Non-layerwise (SGLangXPUConnector)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("use_xpu", [False, True])
@pytest.mark.parametrize("use_mla", [False, True])
def test_sglang_xpu_connector_roundtrip(use_xpu: bool, use_mla: bool):
    """Roundtrip: XPU kvcaches -> CPU memobj -> XPU kvcaches_dst, then compare."""
    device = _current_xpu_device()

    num_layers = 2
    num_blocks = 4
    block_size = 16
    num_heads = 1 if use_mla else 8
    head_size = 64
    num_tokens = 32
    hidden_dim = num_heads * head_size

    kvcaches = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
    )

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    conn = SGLangXPUConnector(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_xpu=use_xpu,
        chunk_size=num_tokens,
        dtype=torch.bfloat16,
        device=device,
        use_mla=use_mla,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    shape = conn.get_shape(num_tokens)
    fmt = MemoryFormat.KV_MLA_FMT if use_mla else MemoryFormat.KV_2LTD
    memobj = pin_alloc.allocate(shape, torch.bfloat16, fmt)

    try:
        # XPU -> CPU
        conn.from_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )

        # CPU -> XPU into fresh caches
        kvcaches_dst = generate_sglang_kv_cache_paged_list_tensors(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
            device=device,
        )
        _zero_kvcaches(kvcaches_dst, use_mla)

        conn.to_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches_dst,
        )

        _check_kv_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
        )
    finally:
        memobj.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_xpu", [False, True])
@pytest.mark.parametrize("use_mla", [False, True])
def test_sglang_xpu_connector_roundtrip_multi_chunk(use_xpu: bool, use_mla: bool):
    """Multi-chunk roundtrip with non-contiguous token ranges."""
    device = _current_xpu_device()

    num_layers = 2
    num_blocks = 6
    block_size = 8
    num_heads = 1 if use_mla else 8
    head_size = 64
    total_tokens = 32
    hidden_dim = num_heads * head_size

    starts = [0, 7, 19]
    ends = [4, 13, 25]

    kvcaches = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
    )

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )
    packed_slot_mapping = _pack_slot_mapping(slot_mapping, starts, ends)

    conn = SGLangXPUConnector(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_xpu=use_xpu,
        chunk_size=total_tokens,
        dtype=torch.bfloat16,
        device=device,
        use_mla=use_mla,
    )

    fmt = MemoryFormat.KV_MLA_FMT if use_mla else MemoryFormat.KV_2LTD
    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobjs = []
    try:
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            shape = conn.get_shape(n)
            memobj = pin_alloc.allocate(shape, torch.bfloat16, fmt)
            conn.from_gpu(
                memobj,
                start=s,
                end=e,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches,
            )
            memobjs.append((s, e, memobj))

        kvcaches_dst = generate_sglang_kv_cache_paged_list_tensors(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
            device=device,
        )
        _zero_kvcaches(kvcaches_dst, use_mla)

        for s, e, memobj in memobjs:
            conn.to_gpu(
                memobj,
                start=s,
                end=e,
                slot_mapping=slot_mapping,
                kvcaches=kvcaches_dst,
            )

        _check_kv_equal(
            kvcaches,
            kvcaches_dst,
            packed_slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
        )
    finally:
        for _, _, memobj in memobjs:
            memobj.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_xpu", [False, True])
def test_sglang_xpu_connector_roundtrip_flat_mha_kvcaches(use_xpu: bool):
    """Non-layerwise SGLang can pass flat MHA kvcaches; ensure roundtrip works."""
    device = _current_xpu_device()

    num_layers = 2
    num_blocks = 4
    block_size = 16
    num_heads = 8
    head_size = 64
    num_tokens = 32
    hidden_dim = num_heads * head_size

    nested_src = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
    )
    kvcaches_src = _as_flat_sglang_mha(nested_src)

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    conn = SGLangXPUConnector(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_xpu=use_xpu,
        chunk_size=num_tokens,
        dtype=torch.bfloat16,
        device=device,
        use_mla=False,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 64)
    memobj = pin_alloc.allocate(
        conn.get_shape(num_tokens), torch.bfloat16, MemoryFormat.KV_2LTD
    )

    try:
        conn.from_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches_src,
        )

        nested_dst = generate_sglang_kv_cache_paged_list_tensors(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=False,
            device=device,
        )
        kvcaches_dst = _as_flat_sglang_mha(nested_dst)
        for t in kvcaches_dst:
            t.zero_()

        conn.to_gpu(
            memobj,
            start=0,
            end=num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches_dst,
        )

        _check_kv_equal(
            _flat_to_nested_sglang_mha(kvcaches_src, num_layers),
            _flat_to_nested_sglang_mha(kvcaches_dst, num_layers),
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=False,
        )
    finally:
        memobj.ref_count_down()
        pin_alloc.close()


# --------------------------------------------------------------------------- #
# Layerwise (SGLangLayerwiseXPUConnector)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("use_xpu", [False, True])
@pytest.mark.parametrize("use_mla", [False, True])
def test_sglang_xpu_connector_roundtrip_layerwise(use_xpu: bool, use_mla: bool):
    """Layerwise roundtrip using generator protocol."""
    device = _current_xpu_device()

    num_layers = 4
    num_blocks = 8
    block_size = 16
    num_heads = 1 if use_mla else 8
    head_size = 64
    num_tokens = 64
    hidden_dim = num_heads * head_size

    kvcaches = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
    )

    total_slots = num_blocks * block_size
    slot_mapping = _make_unique_slot_mapping(
        total_slots=total_slots, num_tokens=num_tokens, device=device
    )

    conn = SGLangLayerwiseXPUConnector(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_xpu=use_xpu,
        chunk_size=num_tokens,
        dtype=torch.bfloat16,
        device=device,
        use_mla=use_mla,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 256)

    # Per-layer list-of-chunks (1 chunk per layer)
    mem_fmt = MemoryFormat.KV_MLA_FMT if use_mla else MemoryFormat.KV_T2D
    memobjs_by_layer = [
        [
            pin_alloc.allocate(
                conn.get_shape(num_tokens),
                torch.bfloat16,
                mem_fmt,
            )
        ]
        for _ in range(num_layers)
    ]

    try:
        # XPU -> CPU (layerwise generator): yields num_layers + 1 times
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

        # CPU -> XPU into fresh caches
        kvcaches_dst = generate_sglang_kv_cache_paged_list_tensors(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
            device=device,
        )
        _zero_kvcaches(kvcaches_dst, use_mla)

        gen2 = conn.batched_to_gpu(
            starts=[0],
            ends=[num_tokens],
            slot_mapping=slot_mapping,
            sync=True,
            kvcaches=kvcaches_dst,
        )

        next(gen2)  # initial yield, expects send()
        for layer_id in range(num_layers):
            gen2.send(memobjs_by_layer[layer_id])
        next(gen2)  # final yield

        _check_kv_equal(
            kvcaches,
            kvcaches_dst,
            slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
        )
    finally:
        for layer in memobjs_by_layer:
            for m in layer:
                m.ref_count_down()
        pin_alloc.close()


@pytest.mark.parametrize("use_xpu", [False, True])
@pytest.mark.parametrize("use_mla", [False, True])
def test_sglang_xpu_connector_roundtrip_layerwise_multi_chunk(
    use_xpu: bool, use_mla: bool
):
    """Layerwise multi-chunk roundtrip."""
    device = _current_xpu_device()

    num_layers = 4
    num_blocks = 8
    block_size = 8
    num_heads = 1 if use_mla else 8
    head_size = 64
    total_tokens = 40
    hidden_dim = num_heads * head_size

    starts = [0, 9, 21]
    ends = [5, 15, 30]

    kvcaches = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=use_mla,
        device=device,
    )

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )
    packed_slot_mapping = _pack_slot_mapping(slot_mapping, starts, ends)

    conn = SGLangLayerwiseXPUConnector(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_xpu=use_xpu,
        chunk_size=total_tokens,
        dtype=torch.bfloat16,
        device=device,
        use_mla=use_mla,
    )

    mem_fmt = MemoryFormat.KV_MLA_FMT if use_mla else MemoryFormat.KV_T2D
    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 128)
    memobjs_by_layer = []
    for _ in range(num_layers):
        per_layer = []
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            per_layer.append(
                pin_alloc.allocate(
                    conn.get_shape(n),
                    torch.bfloat16,
                    mem_fmt,
                )
            )
        memobjs_by_layer.append(per_layer)

    try:
        # XPU -> CPU
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

        # CPU -> XPU
        kvcaches_dst = generate_sglang_kv_cache_paged_list_tensors(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
            device=device,
        )
        _zero_kvcaches(kvcaches_dst, use_mla)

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

        _check_kv_equal(
            kvcaches,
            kvcaches_dst,
            packed_slot_mapping,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=use_mla,
        )
    finally:
        for layer in memobjs_by_layer:
            for memobj in layer:
                memobj.ref_count_down()
        pin_alloc.close()


def test_sglang_layerwise_uses_kernel_transfers(monkeypatch):
    """Ensure layerwise SGLang path dispatches through kernel transfer ops."""
    device = _current_xpu_device()

    num_layers = 3
    num_blocks = 6
    block_size = 8
    num_heads = 8
    head_size = 64
    hidden_dim = num_heads * head_size
    total_tokens = 24
    starts = [0, 7]
    ends = [4, 12]
    num_chunks = len(starts)

    kvcaches = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=False,
        device=device,
    )

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )

    conn = SGLangLayerwiseXPUConnector(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_xpu=True,
        chunk_size=total_tokens,
        dtype=torch.bfloat16,
        device=device,
        use_mla=False,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 128)
    memobjs_by_layer = []
    for _ in range(num_layers):
        per_layer = []
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            per_layer.append(
                pin_alloc.allocate(
                    torch.Size([n, 2, hidden_dim]),
                    torch.bfloat16,
                    MemoryFormat.KV_T2D,
                )
            )
        memobjs_by_layer.append(per_layer)

    calls: list[tuple[object, bool]] = []
    orig_single_layer_kv_transfer_sgl = (
        xpu_connectors.lmc_ops.single_layer_kv_transfer_sgl
    )

    def _recording_single_layer_kv_transfer_sgl(
        src,
        sgl_k,
        sgl_v,
        slot_map,
        direction,
        token_major=False,
    ):
        calls.append((direction, token_major))
        return orig_single_layer_kv_transfer_sgl(
            src,
            sgl_k,
            sgl_v,
            slot_map,
            direction,
            token_major,
        )

    monkeypatch.setattr(
        xpu_connectors.lmc_ops,
        "single_layer_kv_transfer_sgl",
        _recording_single_layer_kv_transfer_sgl,
    )

    try:
        # D2H path
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

        # H2D path into fresh caches
        kvcaches_dst = generate_sglang_kv_cache_paged_list_tensors(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=False,
            device=device,
        )
        _zero_kvcaches(kvcaches_dst, use_mla=False)

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

        expected_calls_per_direction = num_layers * num_chunks
        d2h_calls = [
            c for c in calls if c[0] == xpu_connectors.lmc_ops.TransferDirection.D2H
        ]
        h2d_calls = [
            c for c in calls if c[0] == xpu_connectors.lmc_ops.TransferDirection.H2D
        ]

        assert len(d2h_calls) == expected_calls_per_direction
        assert len(h2d_calls) == expected_calls_per_direction
        assert all(token_major for _, token_major in calls)
    finally:
        for layer in memobjs_by_layer:
            for memobj in layer:
                memobj.ref_count_down()
        pin_alloc.close()


def test_sglang_layerwise_uses_mla_kernel_transfers(monkeypatch):
    """Ensure MLA layerwise path dispatches through single_layer_kv_transfer."""
    device = _current_xpu_device()

    num_layers = 3
    num_blocks = 6
    block_size = 8
    num_heads = 1
    head_size = 64
    hidden_dim = num_heads * head_size
    total_tokens = 24
    starts = [0, 7]
    ends = [4, 12]
    num_chunks = len(starts)

    kvcaches = generate_sglang_kv_cache_paged_list_tensors(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        use_mla=True,
        device=device,
    )

    slot_mapping = _make_unique_slot_mapping(
        total_slots=num_blocks * block_size,
        num_tokens=total_tokens,
        device=device,
    )

    conn = SGLangLayerwiseXPUConnector(
        hidden_dim_size=hidden_dim,
        num_layers=num_layers,
        use_xpu=True,
        chunk_size=total_tokens,
        dtype=torch.bfloat16,
        device=device,
        use_mla=True,
    )

    pin_alloc = PinMemoryAllocator(size=1024 * 1024 * 128)
    memobjs_by_layer = []
    for _ in range(num_layers):
        per_layer = []
        for s, e in zip(starts, ends, strict=False):
            n = e - s
            per_layer.append(
                pin_alloc.allocate(
                    conn.get_shape(n),
                    torch.bfloat16,
                    MemoryFormat.KV_MLA_FMT,
                )
            )
        memobjs_by_layer.append(per_layer)

    single_layer_calls: list[tuple[object, bool]] = []
    sgl_calls: list[tuple[object, bool]] = []
    orig_single_layer_kv_transfer = xpu_connectors.lmc_ops.single_layer_kv_transfer
    orig_single_layer_kv_transfer_sgl = (
        xpu_connectors.lmc_ops.single_layer_kv_transfer_sgl
    )

    def _recording_single_layer_kv_transfer(
        src,
        dst,
        slot_map,
        direction,
        engine_kv_format,
        token_major=False,
    ):
        single_layer_calls.append((direction, token_major))
        return orig_single_layer_kv_transfer(
            src,
            dst,
            slot_map,
            direction,
            engine_kv_format,
            token_major,
        )

    def _recording_single_layer_kv_transfer_sgl(
        src,
        sgl_k,
        sgl_v,
        slot_map,
        direction,
        token_major=False,
    ):
        sgl_calls.append((direction, token_major))
        return orig_single_layer_kv_transfer_sgl(
            src,
            sgl_k,
            sgl_v,
            slot_map,
            direction,
            token_major,
        )

    monkeypatch.setattr(
        xpu_connectors.lmc_ops,
        "single_layer_kv_transfer",
        _recording_single_layer_kv_transfer,
    )
    monkeypatch.setattr(
        xpu_connectors.lmc_ops,
        "single_layer_kv_transfer_sgl",
        _recording_single_layer_kv_transfer_sgl,
    )

    try:
        # D2H path
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

        # H2D path into fresh caches
        kvcaches_dst = generate_sglang_kv_cache_paged_list_tensors(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=num_heads,
            head_size=head_size,
            use_mla=True,
            device=device,
        )
        _zero_kvcaches(kvcaches_dst, use_mla=True)

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

        expected_calls_per_direction = num_layers * num_chunks
        d2h_calls = [
            c
            for c in single_layer_calls
            if c[0] == xpu_connectors.lmc_ops.TransferDirection.D2H
        ]
        h2d_calls = [
            c
            for c in single_layer_calls
            if c[0] == xpu_connectors.lmc_ops.TransferDirection.H2D
        ]

        assert len(d2h_calls) == expected_calls_per_direction
        assert len(h2d_calls) == expected_calls_per_direction
        assert all(token_major for _, token_major in single_layer_calls)
        assert len(sgl_calls) == 0
    finally:
        for layer in memobjs_by_layer:
            for memobj in layer:
                memobj.ref_count_down()
        pin_alloc.close()
