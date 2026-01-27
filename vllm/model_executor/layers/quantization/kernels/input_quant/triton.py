# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .InputQuantKernel import InputQuantConfig, InputQuantKernel
from .pytorch import PytorchInputQuantKernel


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    # Ensure offset calculations use int64 to prevent overflow
    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (
        row_g_id.to(tl.int64) * group_size
    )
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    # Ensure offset calculation uses int64 for y_s_ptr
    y_s_ptr_offset = (scale_col.to(tl.int64) * y_s_col_stride) + scale_row.to(tl.int64)
    y_s_ptr += y_s_ptr_offset

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8_triton(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool = False,
    tma_aligned_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of per-token-group FP8 quantization.

    This function uses Triton kernels for per-token-group quantization.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
            is supported for now.
        column_major_scales: Outputs scales in column major.
        out_q: Optional output tensor. If not provided, function will create.
        use_ue8m0: Whether to use UE8M0 format for scales.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    fp8_min, fp8_max = get_fp8_min_max()

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty(x.shape, device=x.device, dtype=dtype)

    # Allocate the scale tensor in either row- or column-major format.
    if column_major_scales:
        shape = (x.shape[-1] // group_size,) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=use_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M,)](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=use_ue8m0,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s


class TritonInputQuantKernel(InputQuantKernel[InputQuantConfig]):
    @classmethod
    def is_supported(cls):
        return current_platform.is_cuda_alike(), "Not supported on Non cuda platfrom"

    @classmethod
    def can_implement(cls, config: InputQuantConfig):
        if config.group_shape.is_per_group() and config.static:
            return False, "triton per group kernel does not support static quantization"

        if config.group_shape.is_per_tensor():
            return False, "triton per tensor kernel is not implement per tensor"

        if config.group_shape.is_per_token():
            return False, "triton per token kernel is not implement per token"

        return True, ""

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type[InputQuantKernel[InputQuantConfig]]]:
        return [PytorchInputQuantKernel]

    def apply_group_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return per_token_group_quant_fp8_triton(
            x,
            self.group_size,
            column_major_scales=self.is_column_major_scales,
            use_ue8m0=self.use_ue8m0,
        )

    def apply_per_token_per_tensor_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Currently there is no per_tensor, per_token triton kernel implementation.
        raise NotImplementedError
