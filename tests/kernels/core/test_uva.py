# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_cpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cpu_tensor[0, 0] = 1
    cpu_tensor[2, 3] = 2
    cpu_tensor[4, 5] = -1

    cuda_view.mul_(2)
    assert cuda_view[0, 0] == 2
    assert cuda_view[2, 3] == 4
    assert cuda_view[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_gpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"

    assert cuda_view[0, 0] == 0
    assert cuda_view[2, 3] == 0
    assert cuda_view[4, 5] == 0

    cuda_view[0, 0] = 1
    cuda_view[2, 3] = 2
    cuda_view[4, 5] = -1
    cuda_view.mul_(2)

    assert cpu_tensor[0, 0] == 2
    assert cpu_tensor[2, 3] == 4
    assert cpu_tensor[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_write_after_map_coherence(device):
    """The V2 model runner's UVA buffers depend on a live alias between the
    CPU tensor and the device view: per-step writes to the CPU side must be
    visible through the device view without any additional copy.

    This test verifies that contract by writing a non-trivial pattern
    *after* the device view is created and reading it back through the GPU.
    A failure here (stale zeros) would indicate the view is a detached
    snapshot — the exact defect that causes silent input corruption under
    GPU Confidential Computing when is_pinned() is falsely negative."""
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(8, dtype=torch.int32, device="cpu", pin_memory=True)
    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)

    pattern = torch.tensor(
        [10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32, device="cpu"
    )
    cpu_tensor[:] = pattern

    readback = gpu_view.cpu()
    assert torch.equal(readback, pattern), (
        f"UVA coherence broken: expected {pattern.tolist()}, got {readback.tolist()}"
    )


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_require_live_view_raises_on_unregistered_memory(device):
    """When require_live_view=True is passed and the CPU tensor is not pinned
    (i.e. the host memory is not registered for zero-copy), the C++ op should
    raise before entering the detached alloc+copy fallback."""
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(8, dtype=torch.int32, device="cpu")
    assert not cpu_tensor.is_pinned()

    with pytest.raises(RuntimeError, match="require_live_view"):
        get_accelerator_view_from_cpu_tensor(cpu_tensor, require_live_view=True)


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_require_live_view_succeeds_on_pinned_memory(device):
    """When require_live_view=True is passed and the CPU tensor IS pinned,
    the call should succeed and return a live alias."""
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(8, dtype=torch.int32, device="cpu", pin_memory=True)

    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_tensor, require_live_view=True)
    assert gpu_view.device.type == "cuda"

    cpu_tensor[0] = 42
    assert int(gpu_view[0].item()) == 42


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_detached_fallback_without_require_live_view(device):
    """Without require_live_view, an unregistered tensor should still produce
    a valid (detached) CUDA tensor via the alloc+copy path."""
    torch.set_default_device(device)
    cpu_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cpu")
    assert not cpu_tensor.is_pinned()

    cuda_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert cuda_view.device.type == "cuda"
    expected = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cpu")
    assert torch.equal(cuda_view.cpu(), expected)
