# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MXFP4 weight swizzling for the OAI Triton kernel.

``_swizzle_mxfp4`` pre-transforms an MXFP4 weight + its E8M0 block scale
into the layout ``triton_kernels.matmul_ogs.matmul_ogs`` expects on the
current device. Layout selection is delegated to ``triton_kernels``'s own
``make_default_matmul_mxfp4_w_layout`` / ``make_default_matmul_mxfp4_w_scale_layout``
helpers so this module stays free of per-device branching.
"""

from vllm.logger import init_logger
from vllm.utils.import_utils import has_triton_kernels

logger = init_logger(__name__)


def _swizzle_mxfp4(quant_tensor, scale, num_warps=8):
    """Weight + scale swizzle for MXFP4 MoE, used for the OAI Triton kernel.

    Returns a triple ``(quant_tensor, InFlexData, scale)`` shaped for
    ``triton_kernels.matmul_ogs.matmul_ogs``.
    """
    assert has_triton_kernels()
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1
    )
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps
    )

    # Transpose so the quantization axis is on dim1.
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    return quant_tensor, InFlexData(), scale
