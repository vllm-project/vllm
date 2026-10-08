# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not is_pin_memory_available(),
    reason="needs pinned memory on CUDA/ROCm",
)
def test_cpu_params_come_back_pinned_at_their_exact_size():
    # A CPU-offloaded weight of 33 MiB. torch's caching host allocator would
    # pin 64 MiB for it, for as long as the model lives.
    module = nn.Module()
    module.weight = nn.Parameter(
        torch.arange(33 << 19, dtype=torch.float32).to(torch.bfloat16),
        requires_grad=False,
    )
    expected = module.weight.detach().clone()
    before = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)

    with device_loading_context(module, torch.device(current_platform.device_type)):
        assert module.weight.device.type != "cpu"

    assert module.weight.device.type == "cpu"
    assert module.weight.is_pinned()
    torch.testing.assert_close(module.weight, expected)
    cached = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
    assert cached - before < module.weight.nbytes
