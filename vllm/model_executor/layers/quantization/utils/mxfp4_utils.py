# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.utils import direct_register_custom_op

OCP_MX_BLOCK_SIZE = 32


def _dequant_mxfp4(x: torch.Tensor, scale: torch.Tensor,
                   float_dtype: torch.dtype) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                          "MX-FP4 models. Please install it with `pip install "
                          "amd-quark`.") from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def _dequant_mxfp4_fake(x: torch.Tensor, scale: torch.Tensor,
                        float_dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], x.shape[-1] * 2),
                       dtype=float_dtype,
                       device=x.device)


def _quant_dequant_mxfp4(x: torch.Tensor,
                         scale_calculation_mode: str = "even") -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("The package `amd-quark` is required to use "
                          "MX-FP4 models. Please install it with `pip install "
                          "amd-quark`.") from err

    return mx.qdq_mxfp4(x, scale_calculation_mode)


def _quant_dequant_mxfp4_fake(x: torch.Tensor,
                              scale_calculation_mode: str = "even"
                              ) -> torch.Tensor:
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="dequant_mxfp4",
        op_func=_dequant_mxfp4,
        mutates_args=[],
        fake_impl=_dequant_mxfp4_fake,
    )
    dequant_mxfp4 = torch.ops.vllm.dequant_mxfp4
except AttributeError as error:
    raise error

try:
    direct_register_custom_op(
        op_name="quant_dequant_mxfp4",
        op_func=_quant_dequant_mxfp4,
        mutates_args=[],
        fake_impl=_quant_dequant_mxfp4_fake,
    )
    quant_dequant_mxfp4 = torch.ops.vllm.quant_dequant_mxfp4
except AttributeError as error:
    raise error
