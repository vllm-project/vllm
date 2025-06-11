# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.utils import direct_register_custom_op

OCP_MX_BLOCK_SIZE = 32


def quantize(w, dtype, dev, **opt):
    """ Downcast weights to various precision, used for OAI mxfp4 kernel
    """
    from triton_kernels.matmul_ogs import MicroscalingCtx
    from triton_kernels.numerics import InFlexData
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
    from triton_kernels.target_info import get_cdna_version
    if dtype == "bf16":
        wq = w.to(torch.bfloat16).transpose(-1,
                                            -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), MicroscalingCtx()
    elif dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 \
            else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), \
                   MicroscalingCtx()
    else:
        assert dtype == "mx4", f"{dtype=}"
        swizzle_mx_scale = opt.get("swizzle_mx_scale")
        swizzle_mx_value = opt.get("swizzle_mx_value")
        swizzle_axis = 2 if swizzle_mx_scale else None
        w = w.to(torch.bfloat16)
        w, mx_scales, weight_scale_shape = downcast_to_mxfp(
            w,
            torch.uint8,
            axis=1,
            swizzle_axis=swizzle_axis,
            swizzle_scale=swizzle_mx_scale,
            swizzle_value=swizzle_mx_value)
        return w, InFlexData(), MicroscalingCtx(
            weight_scale=mx_scales,
            swizzle_scale=swizzle_mx_scale,
            swizzle_value=swizzle_mx_value,
            actual_weight_scale_shape=weight_scale_shape)


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
