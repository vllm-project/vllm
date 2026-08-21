# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
from vllm.model_executor.kernels.linear.nvfp4.pytorch import (
    TorchNvFp4LinearKernel,
)


def test_torch_backend_selects_nvfp4_kernel() -> None:
    with (
        patch(
            "vllm.model_executor.kernels.linear._get_linear_backend",
            return_value="torch",
        ),
        patch.object(
            TorchNvFp4LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
    ):
        kernel = init_nvfp4_linear_kernel()

    assert isinstance(kernel, TorchNvFp4LinearKernel)
