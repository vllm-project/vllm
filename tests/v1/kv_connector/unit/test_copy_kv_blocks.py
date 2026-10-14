# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for host <-> device KV block copies used by host-buffer transfers."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
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


def test_registered_pageable_buffer_takes_batched_path(batch_calls):
    # The NIXL host buffer is allocated pageable and page-locked later, when
    # the UCX backend registers it; mirror that with cudaHostRegister.
    device = _caches("cuda")
    host = _caches("cpu", pin=False)
    cudart = torch.cuda.cudart()
    for t in host.values():
        assert cudart.cudaHostRegister(t.data_ptr(), t.nbytes, 0) == 0
    try:
        assert all(t.is_pinned() for t in host.values())
        for t in device.values():
            t.copy_(torch.randn(t.shape, dtype=t.dtype))

        copy_kv_blocks(device, host, SRC_IDS, DST_IDS, "d2h")

        for name in device:
            assert torch.equal(host[name][DST_IDS], device[name][SRC_IDS].cpu())
        assert len(batch_calls) == 1
    finally:
        for t in host.values():
            cudart.cudaHostUnregister(t.data_ptr())
