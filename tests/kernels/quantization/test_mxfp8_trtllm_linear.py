# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest

from vllm.model_executor.kernels.linear import (
    FlashInferTrtllmMxfp8LinearKernel,
    init_mxfp8_linear_kernel,
)
from vllm.platforms import PlatformEnum

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@patch("vllm.model_executor.kernels.linear.current_platform")
def test_mxfp8_trtllm_backend_selects_trtllm_kernel(platform_mock) -> None:
    platform_mock._enum = PlatformEnum.CUDA

    with (
        patch(
            "vllm.model_executor.kernels.linear._get_linear_backend",
            return_value="flashinfer_trtllm",
        ),
        patch.object(
            FlashInferTrtllmMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
    ):
        kernel = init_mxfp8_linear_kernel()

    assert isinstance(kernel, FlashInferTrtllmMxfp8LinearKernel)
