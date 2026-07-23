# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the pooled CUDA-IPC multimodal tensor transport (``cuda_ipc``).

The transport keeps GPU-resident feature tensors on the device and ships only a
small metadata proxy from the API-server (frontend) process to each
tensor-parallel worker, which materializes the tensor itself. Opening a CUDA IPC
handle in the same process that exported it is not allowed, so the round-trip is
exercised across a spawned child process (mirroring the real frontend/worker
topology).
"""

import contextlib
import multiprocessing as mp

import pytest
import torch
import torch.multiprocessing as torch_mp

from vllm.multimodal.gpu_ipc_memory import (
    CudaIpcPoolProxy,
    CudaIpcTensorSender,
    cuda_ipc_provider,
    is_cuda_ipc_proxy,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda_ipc transport requires CUDA"
)


@pytest.fixture(scope="module", autouse=True)
def _spawn():
    with contextlib.suppress(RuntimeError):
        torch_mp.set_start_method("spawn", force=True)
    yield


def test_cpu_tensor_is_rejected():
    """CPU tensors must be rejected so the encoder falls back to host copy."""
    sender = CudaIpcTensorSender(pool_bytes=64 << 20, tp_size=1)
    sender.new_message()
    assert sender(torch.zeros(4)) is None


def test_pool_full_falls_back():
    """A tensor larger than the pool yields None (graceful host fallback)."""
    sender = CudaIpcTensorSender(pool_bytes=1 << 10, tp_size=1)
    sender.new_message()
    assert sender(torch.zeros(4096, dtype=torch.float32, device="cuda:0")) is None


def test_encode_returns_plain_metadata_no_pickle():
    """The out-of-band payload is msgpack-native (no embedded pickle blob)."""
    sender = CudaIpcTensorSender(pool_bytes=64 << 20, tp_size=1)
    sender.new_message()
    meta = sender(torch.arange(32, dtype=torch.bfloat16, device="cuda:0"))
    assert isinstance(meta, dict) and "cuda_ipc" in meta
    inner = meta["cuda_ipc"]
    assert set(inner) == {"handle", "flags_shm", "slot", "offset", "nbytes"}
    # handle is a list of ints/bytes/bool only (serializable without pickle).
    assert all(isinstance(x, (int, bytes, bool)) for x in inner["handle"])


def _reconstruct_in_child(proxy: CudaIpcPoolProxy, device_index: int, out: mp.Queue):
    try:
        torch.accelerator.set_device_index(device_index)
        t = proxy.reconstruct(device_index)
        out.put(("ok", t.cpu(), int(t.get_device())))
    except Exception as e:  # noqa: BLE001
        out.put(("err", repr(e), -1))


def _roundtrip_proxy(
    tensor: torch.Tensor, tp_size: int
) -> tuple[CudaIpcTensorSender, CudaIpcPoolProxy]:
    """Sender -> OOB metadata -> provider -> deferred proxy (frontend side)."""
    sender = CudaIpcTensorSender(pool_bytes=64 << 20, tp_size=tp_size)
    sender.new_message()
    meta = sender(tensor)
    assert meta is not None, "expected the tensor to be pooled"
    dtype = str(tensor.dtype).removeprefix("torch.")
    proxy = cuda_ipc_provider(dtype, tuple(tensor.shape), meta)
    assert is_cuda_ipc_proxy(proxy)
    return sender, proxy


@pytest.mark.parametrize("device_index", [0])
def test_roundtrip_same_device(device_index: int):
    """Deferred proxy reconstructs to an equal tensor in a worker process."""
    x = torch.arange(4 * 8, dtype=torch.bfloat16, device=f"cuda:{device_index}")
    x = x.reshape(4, 8)
    sender, proxy = _roundtrip_proxy(x, tp_size=1)

    out: mp.Queue = torch_mp.get_context("spawn").Queue()
    p = torch_mp.get_context("spawn").Process(
        target=_reconstruct_in_child, args=(proxy, device_index, out)
    )
    p.start()
    status, payload, got_dev = out.get(timeout=60)
    p.join(timeout=30)

    assert status == "ok", payload
    assert got_dev == device_index
    torch.testing.assert_close(payload, x.cpu())
    sender._pool.close()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2, reason="cross-device P2P needs >=2 GPUs"
)
def test_roundtrip_cross_device_p2p():
    """A worker on a different GPU reads the pool over P2P (rank!=0 path)."""
    x = torch.arange(64, dtype=torch.float16, device="cuda:0").reshape(8, 8)
    sender, proxy = _roundtrip_proxy(x, tp_size=2)

    out: mp.Queue = torch_mp.get_context("spawn").Queue()
    p = torch_mp.get_context("spawn").Process(
        target=_reconstruct_in_child, args=(proxy, 1, out)
    )
    p.start()
    status, payload, got_dev = out.get(timeout=60)
    p.join(timeout=30)

    assert status == "ok", payload
    assert got_dev == 1
    torch.testing.assert_close(payload, x.cpu())
    sender._pool.close()


def test_tp_size_over_max_falls_back():
    """Beyond the supported single-node width the pool declines (host fallback).

    The recycling cells are indexed per consumer device; a wider world would
    alias cells, so ``_SenderPool`` refuses to start and the encoder falls back
    to the regular host-copy transport instead of recycling slices too early.
    """
    from vllm.multimodal import gpu_ipc_memory

    sender = CudaIpcTensorSender(
        pool_bytes=64 << 20, tp_size=gpu_ipc_memory._MAX_TP + 1
    )
    sender.new_message()
    assert sender(torch.zeros(4, device="cuda:0")) is None
