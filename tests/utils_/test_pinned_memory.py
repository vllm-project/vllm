# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils import pinned_memory
from vllm.utils.pinned_memory import empty_pinned
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

requires_cuda_alike = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="exact pinning is CUDA/ROCm only"
)

# torch's caching host allocator would pin 128 MiB for this.
JUST_OVER_A_POWER_OF_TWO = (64 << 20) + 4096


def _caching_allocator_bytes() -> int:
    return torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)


@requires_cuda_alike
def test_pins_outside_the_caching_allocator():
    before = _caching_allocator_bytes()
    host = empty_pinned(JUST_OVER_A_POWER_OF_TWO, torch.uint8)

    assert host.is_pinned()
    assert host.untyped_storage().nbytes() == JUST_OVER_A_POWER_OF_TWO
    assert _caching_allocator_bytes() == before


@requires_cuda_alike
@pytest.mark.parametrize(
    ("size", "stride"),
    [((7,), None), ((3, 5), None), ((4, 6), (1, 4)), ((0, 4), None)],
    ids=["vector", "matrix", "column-major", "empty"],
)
def test_round_trips_through_the_device(size, stride):
    dtype = torch.bfloat16
    host = empty_pinned(size, dtype, stride=stride)
    assert host.shape == size
    if stride is not None:
        assert host.stride() == stride

    device = current_platform.device_type
    source = torch.randint(-100, 100, size, device=device).to(dtype)
    host.copy_(source, non_blocking=True)
    torch.accelerator.synchronize()
    assert torch.equal(host.to(device), source)


@requires_cuda_alike
def test_the_device_sees_the_same_memory_through_uva():
    host = empty_pinned(1024, torch.int32)
    host.copy_(torch.arange(1024, dtype=torch.int32))
    view = get_accelerator_view_from_cpu_tensor(host)
    assert torch.equal(view.cpu(), host)

    host[0] = -1
    assert view[0].item() == -1


@requires_cuda_alike
def test_memory_outlives_the_tensor_while_a_view_remains():
    host = empty_pinned(256, torch.int32)
    host.copy_(torch.arange(256, dtype=torch.int32))
    tail = host[128:]
    del host
    gc.collect()

    assert torch.equal(tail, torch.arange(128, 256, dtype=torch.int32))


def test_falls_back_to_pageable_memory_without_pinning(monkeypatch):
    monkeypatch.setattr(pinned_memory, "is_pin_memory_available", lambda: False)
    host = empty_pinned((2, 3), torch.float16, stride=(1, 2))

    assert not host.is_pinned()
    assert host.shape == (2, 3)
    assert host.stride() == (1, 2)


@requires_cuda_alike
@pytest.mark.parametrize("register", [False, True])
def test_both_paths_pin_exactly_and_round_trip(register):
    """Allocation and registration both pin the requested size, not more."""
    before = _caching_allocator_bytes()
    host = empty_pinned(JUST_OVER_A_POWER_OF_TWO, torch.uint8, register=register)

    assert host.is_pinned()
    assert host.untyped_storage().nbytes() == JUST_OVER_A_POWER_OF_TWO
    assert _caching_allocator_bytes() == before

    device = current_platform.device_type
    source = torch.randint(0, 255, (JUST_OVER_A_POWER_OF_TWO,), device=device).to(
        torch.uint8
    )
    host.copy_(source, non_blocking=True)
    torch.accelerator.synchronize()
    assert torch.equal(host.to(device), source)


@requires_cuda_alike
def test_registered_memory_is_released_with_its_last_view():
    """Registering a range that is still registered fails, so this must not."""
    for _ in range(3):
        host = empty_pinned(1 << 20, torch.uint8, register=True)
        host.fill_(7)
        assert int(host.sum().item()) == 7 * (1 << 20)
        del host
        gc.collect()


@requires_cuda_alike
def test_the_returned_view_outlives_the_tensor_it_was_taken_from():
    """The caller gets a view; the backing allocation owns the release.

    A finalizer hung off the intermediate tensor instead would unregister the
    range as soon as empty_pinned() returned, while the caller still held it.
    """
    host = empty_pinned((4, 1 << 18), torch.uint8, register=True)
    gc.collect()

    source = torch.full((4, 1 << 18), 3, dtype=torch.uint8, device="cuda")
    host.copy_(source, non_blocking=True)
    torch.accelerator.synchronize()
    assert torch.equal(host.to("cuda"), source)


@requires_cuda_alike
def test_raises_by_default_and_falls_back_only_when_asked(monkeypatch, caplog):
    def boom(nbytes):
        raise RuntimeError("no page-locked memory left")

    monkeypatch.setattr(pinned_memory, "_alloc", boom)
    with pytest.raises(RuntimeError, match="no page-locked memory left"):
        empty_pinned(1 << 20, torch.uint8)

    host = empty_pinned(1 << 20, torch.uint8, fallback=True)
    assert host.is_pinned()
    assert "falling back" in caplog.text
