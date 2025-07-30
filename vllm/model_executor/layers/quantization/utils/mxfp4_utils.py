# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

OCP_MX_BLOCK_SIZE = 32


def _swizzle_mxfp4(quant_tensor, scale, num_warps):
    """ weight swizzle for mxfp4 moe, used for OAI mxfp4 kernel
    """
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1)
    scale_layout, scale_layout_opts = (
        layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1,
                                                        num_warps=num_warps))
    if current_platform.is_cuda() and \
        torch.cuda.get_device_capability()[0] == 10:
        constraints = {
            "is_persistent": True,
            "epilogue_subtile": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(wrap_torch_tensor(quant_tensor, dtype=FP4),
                                  value_layout, **value_layout_opts)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout,
                           **scale_layout_opts)
    return quant_tensor, InFlexData(), scale


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
