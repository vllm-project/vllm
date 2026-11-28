# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for host <-> device KV block copies used by host-buffer transfers."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.distributed.kv_transfer.kv_connector.utils import (
    _try_copy_kv_blocks_batched,
    copy_kv_blocks,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")

NUM_LAYERS = 3
NUM_BLOCKS = 32
BLOCK_SHAPE = (2, 16, 4, 8)  # (K/V, block_size, num_heads, head_size)

# Non-contiguous ids, and src ids differ from dst ids.
SRC_IDS = [3, 0, 17, 9, 31]
DST_IDS = [8, 21, 1, 30, 12]
UNTOUCHED = [i for i in range(NUM_BLOCKS) if i not in DST_IDS]


def _caches(device: str, pin: bool = False) -> dict[str, torch.Tensor]:
    return {
        f"layer.{i}": torch.zeros(
            (NUM_BLOCKS, *BLOCK_SHAPE),
            dtype=torch.bfloat16,
            device=device,
            pin_memory=pin,
        )
        for i in range(NUM_LAYERS)
    }


def _random_like(caches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn(t.shape, dtype=t.dtype, pin_memory=True)
        for name, t in caches.items()
    }


def _stall_current_stream() -> None:
    # Hold the stream back so that a copy which returns without waiting for
    # it is caught reading/writing too early.
    torch.cuda._sleep(50_000_000)


@pytest.fixture
def batch_calls(monkeypatch):
    calls = []
    real = ops.swap_blocks_batch

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(ops, "swap_blocks_batch", spy)
    return calls


@pytest.fixture(params=["default_stream", "side_stream"])
def stream(request):
    # swap_blocks_batch takes the batched driver call only on a non-default
    # stream (serving uses a dedicated one) and per-copy memcpys on the
    # legacy default stream; cover both.
    if request.param == "default_stream":
        s = torch.cuda.default_stream()
    else:
        s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        yield s


@pytest.mark.parametrize("pin", [True, False])
def test_d2h_copies_selected_blocks(pin, stream, batch_calls):
    device = _caches("cuda")
    host = _caches("cpu", pin=pin)
    ref = _random_like(device)
    # Enqueue the source writes behind a stall on the current stream without
    # syncing, so a copy that is not ordered after them, or returns before
    # finishing, leaves stale data in the host buffer.
    _stall_current_stream()
    for name in device:
        device[name].copy_(ref[name], non_blocking=True)

    copy_kv_blocks(device, host, SRC_IDS, DST_IDS, "d2h")

    # Read the host buffer immediately: the copy must be complete on return.
    for name in device:
        assert torch.equal(host[name][DST_IDS], ref[name][SRC_IDS])
        assert torch.count_nonzero(host[name][UNTOUCHED]) == 0
    # Pinned host buffers take the batched path; pageable ones fall back.
    assert len(batch_calls) == (1 if pin else 0)
    if pin:
        src_ptrs, dst_ptrs, sizes = batch_calls[0][:3]
        assert len(src_ptrs) == len(dst_ptrs) == len(sizes) == NUM_LAYERS * len(SRC_IDS)


@pytest.mark.parametrize("pin", [True, False])
def test_h2d_copies_selected_blocks(pin, stream, batch_calls):
    host = _caches("cpu", pin=pin)
    device = _caches("cuda")
    ref = _random_like(host)
    for name in host:
        host[name].copy_(ref[name])
    _stall_current_stream()

    copy_kv_blocks(host, device, SRC_IDS, DST_IDS, "h2d")

    if pin:
        # The batched path returns with the current stream drained.
        assert stream.query()
    for name in host:
        assert torch.equal(device[name][DST_IDS].cpu(), ref[name][SRC_IDS])
        assert torch.count_nonzero(device[name][UNTOUCHED]) == 0
    assert len(batch_calls) == (1 if pin else 0)


def test_non_contiguous_buffers_fall_back(batch_calls):
    # A permuted view keeps the block dim first but is not contiguous.
    device = {name: t.permute(0, 1, 3, 2, 4) for name, t in _caches("cuda").items()}
    # empty_like keeps the permuted strides, so the pinned host side is
    # non-contiguous too.
    host = {
        name: torch.empty_like(t, device="cpu", pin_memory=True).zero_()
        for name, t in device.items()
    }
    assert not any(t.is_contiguous() for t in host.values())
    for t in device.values():
        t.copy_(torch.randn(t.shape, dtype=t.dtype))

    copy_kv_blocks(device, host, SRC_IDS, DST_IDS, "d2h")

    for name in device:
        assert torch.equal(host[name][DST_IDS], device[name][SRC_IDS].cpu())
    assert batch_calls == []


def test_mismatched_block_bytes_fall_back(batch_calls):
    device = _caches("cuda")
    host = _caches("cpu", pin=True)
    # One layer with a different per-block size (e.g. a hybrid model group).
    device["layer.0"] = torch.zeros(
        (NUM_BLOCKS, 2, 16, 2, 8), dtype=torch.bfloat16, device="cuda"
    )
    host["layer.0"] = torch.zeros(
        (NUM_BLOCKS, 2, 16, 2, 8), dtype=torch.bfloat16, pin_memory=True
    )
    for t in device.values():
        t.copy_(torch.randn(t.shape, dtype=t.dtype))

    copy_kv_blocks(device, host, SRC_IDS, DST_IDS, "d2h")

    for name in device:
        assert torch.equal(host[name][DST_IDS], device[name][SRC_IDS].cpu())
    assert batch_calls == []


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_registered_pageable_buffer_takes_batched_path(direction, batch_calls):
    # The NIXL host buffer is allocated pageable and page-locked later, when
    # the backend registers it; mirror that with cudaHostRegister.
    device = _caches("cuda")
    host = _caches("cpu", pin=False)
    src, dst = (device, host) if direction == "d2h" else (host, device)
    cudart = torch.cuda.cudart()
    registered = []
    try:
        for t in host.values():
            assert cudart.cudaHostRegister(t.data_ptr(), t.nbytes, 0) == 0
            registered.append(t)
        assert all(t.is_pinned() for t in host.values())
        for t in src.values():
            t.copy_(torch.randn(t.shape, dtype=t.dtype))

        copy_kv_blocks(src, dst, SRC_IDS, DST_IDS, direction)

        for name in src:
            assert torch.equal(dst[name][DST_IDS].cpu(), src[name][SRC_IDS].cpu())
        assert len(batch_calls) == 1
    finally:
        for t in registered:
            assert cudart.cudaHostUnregister(t.data_ptr()) == 0


def test_overlapping_destinations_fall_back(batch_calls):
    # Layers sharing a KV cache map to one device tensor but have separate
    # host buffers; the per-layer path writes them in order, last one wins.
    host = _caches("cpu", pin=True)
    for t in host.values():
        t.copy_(torch.randn(t.shape, dtype=t.dtype))
    shared = torch.zeros(
        (NUM_BLOCKS, *BLOCK_SHAPE), dtype=torch.bfloat16, device="cuda"
    )
    device = {name: shared for name in host}

    copy_kv_blocks(host, device, SRC_IDS, DST_IDS, "h2d")

    last = host[f"layer.{NUM_LAYERS - 1}"]
    assert torch.equal(shared[DST_IDS].cpu(), last[SRC_IDS])
    assert batch_calls == []


@pytest.mark.parametrize("bad_id", [-1, NUM_BLOCKS])
def test_out_of_range_block_ids_fall_back(bad_id, batch_calls):
    # The batched path would turn these into addresses outside the buffers;
    # leave them to tensor indexing, which wraps -1 and raises on the rest.
    device = _caches("cuda")
    host = _caches("cpu", pin=True)
    ids = [*SRC_IDS[:-1], bad_id]

    assert not _try_copy_kv_blocks_batched(device, host, ids, DST_IDS)
    assert not _try_copy_kv_blocks_batched(device, host, SRC_IDS, ids)
    assert batch_calls == []


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_tensors_on_non_current_device(direction, batch_calls):
    device = _caches("cuda:1")
    host = _caches("cpu", pin=True)
    src, dst = (device, host) if direction == "d2h" else (host, device)
    ref = _random_like(src)
    if direction == "h2d":
        for name in host:
            host[name].copy_(ref[name])
    # Use non-default streams on both GPUs, as serving does: the legacy
    # default streams would order a copy issued on the wrong device anyway.
    prev_stream = torch.cuda.current_stream(1)
    torch.cuda.set_stream(torch.cuda.Stream(device=1))
    try:
        # Enqueue work on cuda:1 behind a stall: the source writes for d2h,
        # and for h2d a write to the destination that the copy must land
        # after. A copy ordered on the current device's stream gets it wrong.
        with torch.cuda.device(1):
            _stall_current_stream()
            for name in device:
                if direction == "d2h":
                    device[name].copy_(ref[name], non_blocking=True)
                else:
                    device[name].zero_()

        with torch.cuda.device(0), torch.cuda.stream(torch.cuda.Stream(device=0)):
            copy_kv_blocks(src, dst, SRC_IDS, DST_IDS, direction)

        for name in src:
            assert torch.equal(dst[name][DST_IDS].cpu(), ref[name][SRC_IDS])
        assert len(batch_calls) == 1
    finally:
        torch.cuda.set_stream(prev_stream)
