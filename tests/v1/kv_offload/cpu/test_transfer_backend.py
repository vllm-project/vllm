# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload import transfer_backend
from vllm.v1.kv_offload.base import CanonicalKVCacheRef
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.gpu_worker import PlatformTransferBackend
from vllm.v1.kv_offload.transfer_backend import (
    KVTransferBackend,
    get_kv_transfer_backend,
    register_kv_transfer_backend,
    resolve_kv_transfer_backend,
    unregister_kv_transfer_backend,
)


@pytest.fixture(autouse=True)
def clean_registry():
    saved = dict(transfer_backend._REGISTRY)
    yield
    transfer_backend._REGISTRY.clear()
    transfer_backend._REGISTRY.update(saved)


class StubBackend(KVTransferBackend):
    def __init__(self):
        self.calls: list[str] = []

    def select_copy_op(
        self, layer_refs_per_group, gpu_to_cpu, host_memory_is_pinned=True
    ):
        self.calls.append("select_copy_op")
        return lambda *args, **kwargs: None

    def descriptor_ptr_dtype(self):
        return torch.uint64

    def validate_device_tensor(self, tensor):
        self.calls.append("validate_device_tensor")

    def host_register(self, ptr, num_bytes):
        self.calls.append("host_register")
        return True

    def host_unregister(self, ptr):
        self.calls.append("host_unregister")


def test_no_backend_on_unsupported_platform():
    if current_platform.is_cuda_alike() or current_platform.is_xpu():
        pytest.skip("built-in backend exists on this platform")
    assert get_kv_transfer_backend() is None
    with pytest.raises(RuntimeError, match="register a KVTransferBackend"):
        resolve_kv_transfer_backend()


def test_default_backend_on_cuda_alike(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    assert isinstance(get_kv_transfer_backend(), PlatformTransferBackend)


def test_default_backend_on_xpu(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    assert isinstance(get_kv_transfer_backend(), PlatformTransferBackend)


def test_registered_backend_takes_precedence():
    stub = StubBackend()
    register_kv_transfer_backend(current_platform.device_type, stub)
    assert get_kv_transfer_backend() is stub
    unregister_kv_transfer_backend(current_platform.device_type)
    unregister_kv_transfer_backend(current_platform.device_type)  # idempotent
    if current_platform.is_cuda_alike() or current_platform.is_xpu():
        assert get_kv_transfer_backend() is not None
    else:
        assert get_kv_transfer_backend() is None


def test_platform_backend_delegates_host_register(monkeypatch):
    calls: list[tuple[int, int]] = []

    def register(ptr: int, size: int) -> bool:
        calls.append((ptr, size))
        return True

    monkeypatch.setattr(gpu_worker, "host_register", register)
    unregistered: list[int] = []
    monkeypatch.setattr(gpu_worker, "host_unregister", unregistered.append)
    backend = PlatformTransferBackend()
    assert backend.host_register(0x1000, 64)
    backend.host_unregister(0x1000)
    assert calls == [(0x1000, 64)]
    assert unregistered == [0x1000]


def test_platform_backend_descriptor_dtype(monkeypatch):
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    assert PlatformTransferBackend().descriptor_ptr_dtype() == torch.int64
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    assert PlatformTransferBackend().descriptor_ptr_dtype() == torch.uint64


def test_platform_backend_validate_device_tensor():
    backend = PlatformTransferBackend()

    class FakeTensor:
        is_cuda = True
        is_xpu = False

    backend.validate_device_tensor(FakeTensor())

    class CpuTensor:
        is_cuda = False
        is_xpu = False

    with pytest.raises(AssertionError):
        backend.validate_device_tensor(CpuTensor())


def test_platform_backend_select_copy_op(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(gpu_worker, "_select_swap_blocks_fn", lambda *a, **k: sentinel)
    assert (
        PlatformTransferBackend().select_copy_op(
            [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=512)]], gpu_to_cpu=False
        )
        is sentinel
    )


def test_handler_uses_injected_backend():
    """A registered/injected backend replaces every device-bound operation."""
    stub = StubBackend()
    gpu_tensor = torch.empty((4, 512), dtype=torch.int8)
    cpu_tensor = torch.empty((4, 512), dtype=torch.int8)
    refs = [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=512)]]
    handler = gpu_worker.SingleDirectionOffloadingHandler(
        gpu_tensors=[gpu_tensor],
        cpu_tensors=[cpu_tensor],
        blocks_per_chunk=1,
        layer_refs_per_group=refs,
        gpu_to_cpu=False,
        backend=stub,
    )
    assert handler._swap_blocks_batch is not None
    assert "select_copy_op" in stub.calls
    assert "validate_device_tensor" in stub.calls


def test_worker_resolves_backend_via_registry(monkeypatch):
    stub = StubBackend()
    monkeypatch.setattr(
        transfer_backend,
        "get_kv_transfer_backend",
        lambda: stub,
    )
    from vllm.v1.kv_offload.transfer_backend import resolve_kv_transfer_backend

    assert resolve_kv_transfer_backend() is stub
