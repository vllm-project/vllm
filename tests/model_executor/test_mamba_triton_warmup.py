# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.warmup.mamba_triton_warmup import _warm_batch_memcpy_kernel
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_mamba_batch_memcpy_kernel_compiles_on_gpu() -> None:
    _warm_batch_memcpy_kernel(torch.device("cuda"))
    torch.accelerator.synchronize("cuda")
