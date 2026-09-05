# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="NVFP4 custom op requires CUDA"
)
def test_nvfp4_quantize_128x4_opcheck() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("NVFP4 b12x is only supported on SM120+")
    if not has_flashinfer():
        pytest.skip("FlashInfer is unavailable")

    device = torch.device("cuda")
    activations = torch.randn((128, 128), dtype=torch.bfloat16, device=device)
    global_scale = torch.ones((), dtype=torch.float32, device=device)
    opcheck(
        torch.ops.vllm.flashinfer_nvfp4_quantize_128x4.default,
        (activations, global_scale),
    )
