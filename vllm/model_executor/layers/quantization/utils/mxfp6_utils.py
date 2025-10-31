# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_BLOCK_SIZE
from vllm.utils.torch_utils import direct_register_custom_op


def _quant_dequant_mxfp6(
    x: torch.Tensor,
    quant_dtype: str,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            fake_quantize_fp4_fp6_per_group_with_scale,
        )
        from quark.torch.quantization.utils import even_round, reshape_to_blocks
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP6 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    axis = -1
    block_x = reshape_to_blocks(x, OCP_MX_BLOCK_SIZE, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    amax = amax.squeeze(-1)

    # TODO: there are other rounding strategies supported in quark and in the
    # config.json that we do not check for here!
    if scale_calculation_mode != "even":
        raise NotImplementedError(
            f"Scale calculation mode {scale_calculation_mode} is not yet "
            "supported in MX-FP6 quantization"
        )
    scale = even_round(amax, quant_dtype)

    # Apply dequantize(quantize(x)).
    x = fake_quantize_fp4_fp6_per_group_with_scale(
        x,
        scale.to(x.device),
        axis=axis,
        group_size=OCP_MX_BLOCK_SIZE,
        quant_dtype=quant_dtype,
    )

    return x


def _quant_dequant_mxfp6_fake(
    x: torch.Tensor,
    quant_dtype: str,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    return torch.empty_like(x)


def _dequant_mxfp6(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype, quant_dtype: str
) -> torch.Tensor:
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            dequantize_fp4_fp6_per_group,
        )
        from quark.torch.utils.pack import create_pack_method
    except ImportError as e:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP6 models. Please install it with `pip install "
            "amd-quark`."
        ) from e

    pack_method = create_pack_method(None, dtype=quant_dtype)
    unpacked_x = pack_method.unpack(x, reorder=False)

    scale = 2 ** (scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

    # TODO: `dequantize_fp4_fp6_per_group` and `prepare_inputs_per_group`
    # always return fp32.
    return dequantize_fp4_fp6_per_group(
        unpacked_x,
        scale,
        axis=-1,
        group_size=OCP_MX_BLOCK_SIZE,
        quant_dtype=quant_dtype,
    ).to(float_dtype)


def _dequant_mxfp6_fake(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype, quant_dtype: str
) -> torch.Tensor:
    assert (x.shape[-1] * 4) % 3 == 0
    return torch.empty(
        (*x.shape[:-1], (x.shape[-1] * 4) // 3), dtype=float_dtype, device=x.device
    )


# Protect these operations into a torch custom op to avoid errors as
# torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
# Explanation: Dynamo does not know how to trace the builtin
# `kernel_ext.PyCapsule.dq_uint8_mxfp4_to_half.` This function is either a
# Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python
# extension (perhaps created with pybind).
# TODO: Make sure there is no way to avoid having these functions
# marked as skipped by dynamo.
try:
    direct_register_custom_op(
        op_name="quant_dequant_mxfp6",
        op_func=_quant_dequant_mxfp6,
        mutates_args=[],
        fake_impl=_quant_dequant_mxfp6_fake,
    )
except AttributeError as error:
    raise error


# Expose keyword arguments.
def quant_dequant_mxfp6(
    x: torch.Tensor,
    quant_dtype: str,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    return torch.ops.vllm.quant_dequant_mxfp6(x, quant_dtype, scale_calculation_mode)


try:
    direct_register_custom_op(
        op_name="dequant_mxfp6",
        op_func=_dequant_mxfp6,
        mutates_args=[],
        fake_impl=_dequant_mxfp6_fake,
    )
except AttributeError as error:
    raise error


def dequant_mxfp6(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype, quant_dtype: str
) -> torch.Tensor:
    return torch.ops.vllm.dequant_mxfp6(x, scale, float_dtype, quant_dtype)
