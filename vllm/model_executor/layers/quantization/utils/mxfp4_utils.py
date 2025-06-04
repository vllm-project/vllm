# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

OCP_MX_BLOCK_SIZE = 32


def dequant_mxfp4(x: torch.Tensor, scale: torch.Tensor,
                  float_dtype: torch.dtype) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                          "MX-FP4 models. Please install it with `pip install "
                          "amd-quark`.") from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def quant_dequant_mxfp4(x: torch.Tensor,
                        scale_calculation_mode: str = "even") -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                          "MX-FP4 models. Please install it with `pip install "
                          "amd-quark`.") from err

    return mx.qdq_mxfp4(x, scale_calculation_mode)
