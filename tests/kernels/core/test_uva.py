# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

DEVICE_TYPE = current_platform.device_type
DEVICES = [
    f"{DEVICE_TYPE}:{i}"
    for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", DEVICES)
def test_cpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert gpu_view.device.type == DEVICE_TYPE

    assert gpu_view[0, 0] == 0
    assert gpu_view[2, 3] == 0
    assert gpu_view[4, 5] == 0

    cpu_tensor[0, 0] = 1
    cpu_tensor[2, 3] = 2
    cpu_tensor[4, 5] = -1

    gpu_view.mul_(2)
    assert gpu_view[0, 0] == 2
    assert gpu_view[2, 3] == 4
    assert gpu_view[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", DEVICES)
def test_gpu_write(device):
    torch.set_default_device(device)
    cpu_tensor = torch.zeros(10, 10, device="cpu", pin_memory=True, dtype=torch.int32)
    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert gpu_view.device.type == DEVICE_TYPE

    assert gpu_view[0, 0] == 0
    assert gpu_view[2, 3] == 0
    assert gpu_view[4, 5] == 0

    gpu_view[0, 0] = 1
    gpu_view[2, 3] = 2
    gpu_view[4, 5] = -1
    gpu_view.mul_(2)

    assert cpu_tensor[0, 0] == 2
    assert cpu_tensor[2, 3] == 4
    assert cpu_tensor[4, 5] == -2


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.parametrize("device", DEVICES)
def test_non_pinned_cpu_tensor(device):
    # Non-pinned CPU tensors are internally copied into a pinned buffer,
    # so the resulting gpu view reflects the values at creation time but
    # is decoupled from further writes to the original `cpu_tensor`.
    torch.set_default_device(device)
    cpu_tensor = torch.arange(100, dtype=torch.int32, device="cpu").view(10, 10)
    assert not cpu_tensor.is_pinned()
    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_tensor)
    assert gpu_view.device.type == DEVICE_TYPE

    assert gpu_view[0, 0] == 0
    assert gpu_view[2, 3] == 23
    assert gpu_view[9, 9] == 99

    # Writes to the original (unpinned) CPU tensor must not affect the view,
    # since a private pinned copy was made.
    cpu_tensor[0, 0] = -1
    assert gpu_view[0, 0] == 0

    # The view itself remains writable and independently usable.
    gpu_view.mul_(2)
    assert gpu_view[2, 3] == 46
    assert gpu_view[9, 9] == 198


@pytest.mark.skipif(not is_uva_available(), reason="UVA is not available.")
@pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU non-contiguous UVA test."
)
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_non_contiguous_strided_view(device, pinned):
    torch.set_default_device(device)
    # Simulate scale_kn: a [32, 16] contiguous tensor
    scale_kn = torch.arange(
        512, dtype=torch.float32, device="cpu", pin_memory=pinned
    ).view(32, 16)
    # Transposed view: shape [16, 32], non-contiguous stride (1, 16)
    cpu_view = scale_kn.t()
    assert cpu_view.shape == (16, 32)
    assert cpu_view.stride() == (1, 16)
    assert not cpu_view.is_contiguous()

    gpu_view = get_accelerator_view_from_cpu_tensor(cpu_view)
    assert gpu_view.device.type == DEVICE_TYPE
    assert gpu_view.shape == (16, 32)
    assert gpu_view.stride() == (1, 16)
    assert not gpu_view.is_contiguous()

    # Transposing tensor_view back should yield a contiguous [32, 16] tensor
    gpu_view_t = gpu_view.t()
    assert gpu_view_t.shape == (32, 16)
    assert gpu_view_t.is_contiguous()

    # Correctness check: values match original transposed CPU tensor
    assert torch.equal(gpu_view.cpu(), cpu_view)
