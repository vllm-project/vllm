# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.model_executor.offloader import UVAOffloader
from vllm.model_executor.utils import replace_parameter
from vllm.utils.platform_utils import is_uva_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_uva_available(),
    reason="requires CUDA with UVA",
)

NUMEL = 2 * 1024 * 1024


def _uva_offloaded_layer() -> nn.Module:
    layer = nn.Module()
    layer.weight = nn.Parameter(
        torch.randn(NUMEL, device="cuda", dtype=torch.bfloat16), requires_grad=False
    )
    UVAOffloader(cpu_offload_max_bytes=1 << 40).wrap_modules(iter([layer]))
    assert layer.weight._vllm_is_uva_offloaded
    return layer


# "fits" is copied into the host storage; "grows" (2x bytes) gets .data rebound
# to the new device tensor while the Parameter keeps its UVA marker.
@pytest.mark.parametrize(
    "new_dtype", [torch.bfloat16, torch.float32], ids=["fits", "grows"]
)
def test_repacked_uva_param_is_reoffloaded(new_dtype: torch.dtype):
    layer = _uva_offloaded_layer()
    param = layer.weight
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()

    new = torch.randn(NUMEL, device="cuda", dtype=new_dtype)
    expected = new.cpu()
    with device_loading_context(layer, torch.device("cuda")):
        replace_parameter(layer, "weight", new)
    del new
    torch.cuda.synchronize()

    assert layer.weight is param
    assert param._vllm_is_uva_offloaded
    assert param.dtype == new_dtype
    torch.testing.assert_close(param.cpu(), expected)
    # The repacked weight must not stay resident on the device.
    assert torch.cuda.memory_allocated() <= allocated_before
