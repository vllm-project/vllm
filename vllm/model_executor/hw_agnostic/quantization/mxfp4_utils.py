# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MXFP4 weight swizzling for the OAI Triton kernel.

``_swizzle_mxfp4`` pre-transforms an MXFP4 weight + its E8M0 block scale
into the layout ``triton_kernels.matmul_ogs.matmul_ogs`` expects on the
current device. The device branches select between the strided default,
the CUDA Hopper / Blackwell layouts, and the ROCm CDNA4 / gfx950 scale
layouts -- this mirrors the per-device layout choices the kernel itself
makes at runtime, so the pre-swizzle and the kernel's load layout agree.
"""

from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.torch_utils import is_torch_equal_or_newer

logger = init_logger(__name__)


def should_use_cdna4_mx_scale_swizzle() -> bool:
    """Whether to use the CDNA4 swizzled scale layout for MXFP4 on gfx950.

    CDNA4 swizzle requires BLOCK_K % 256 == 0; at TP >= 4 ``matmul_ogs``
    picks BK<256 tiles for the smaller per-rank shapes, so swizzle must be
    off there to keep the weight-load layout consistent with whatever the
    kernel selects at runtime.
    """
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950() and get_tensor_model_parallel_world_size() <= 2


def _swizzle_mxfp4(quant_tensor, scale, num_warps=8):
    """Weight + scale swizzle for MXFP4 MoE, used for the OAI Triton kernel.

    Returns a triple ``(quant_tensor, InFlexData, scale)`` shaped for
    ``triton_kernels.matmul_ogs.matmul_ogs``.
    """
    assert has_triton_kernels()
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    from triton_kernels.tensor_details.layout import StridedLayout

    value_layout_opts: dict[str, Any] = {}
    scale_layout_opts: dict[str, Any] = {}

    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability(90)
        and not is_torch_equal_or_newer("2.8.1")
    ):
        logger.warning_once(
            "Mxfp4 on hopper is running on torch < 2.8.1, "
            "this causes swizzling to be disabled, which may "
            "cause performance degradation. Please upgrade to torch nightly"
        )
        value_layout = StridedLayout
        scale_layout = StridedLayout
    elif current_platform.is_rocm():
        value_layout = StridedLayout
        if should_use_cdna4_mx_scale_swizzle():
            try:
                # triton < 3.6
                from triton_kernels.tensor_details.layout import GFX950MXScaleLayout

                scale_layout = GFX950MXScaleLayout
            except ImportError:
                # triton >= 3.6
                from triton_kernels.tensor_details.layout import CDNA4MXScaleLayout

                scale_layout = CDNA4MXScaleLayout
        else:
            scale_layout = StridedLayout
    else:
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
            mx_axis=1
        )
        scale_layout, scale_layout_opts = (
            layout.make_default_matmul_mxfp4_w_scale_layout(
                mx_axis=1, num_warps=num_warps
            )
        )

    if current_platform.is_cuda():
        if current_platform.is_device_capability(90):
            constraints = {"split_k": 1}
            opt_flags.update_opt_flags_constraints(constraints)
        elif current_platform.is_device_capability_family(100):
            constraints = {"is_persistent": True, "epilogue_subtile": 1}
            opt_flags.update_opt_flags_constraints(constraints)

    # Transpose so the quantization axis is on dim1.
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    return quant_tensor, InFlexData(), scale
