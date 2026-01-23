# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import functools
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    get_fp8_min_max,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
    all_close_1d,
    per_tensor_dequantize,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    fp8_gemm_nt,
    get_tma_aligned_size,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
    should_use_deepgemm_for_fp8_linear,
    transform_sf_into_required_layout,
)
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    is_flashinfer_fp8_blockscale_gemm_supported,
    should_use_flashinfer_for_blockscale_fp8_gemm,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def is_fp8(x: torch.dtype | torch.Tensor) -> bool:
    if isinstance(x, torch.Tensor):
        x = x.dtype
    return x == torch.float8_e4m3fn or x == torch.float8_e4m3fnuz


# We need to pass in the is_hopper flag as argument because the function
# current_platform.is_device_capability() is not supported by Torch compiler.
def cutlass_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return ops.cutlass_scaled_mm(
        A,
        B.T,
        out_dtype=output_dtype,
        scale_a=As,
        scale_b=Bs.T,
    )


# TODO we should be able to change the type of block_size to GroupShape
# after we resolve GroupShape compilation issue
# https://github.com/vllm-project/vllm/issues/25270
def _w8a8_triton_block_scaled_mm_func(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return w8a8_triton_block_scaled_mm(
        qx, weight, x_scale, weight_scale, block_size, output_dtype
    )


def _w8a8_triton_block_scaled_mm_fake(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (qx.size(0), weight.size(0)), dtype=output_dtype, device=qx.device
    )


direct_register_custom_op(
    "w8a8_triton_block_scaled_mm_func",
    _w8a8_triton_block_scaled_mm_func,
    fake_impl=_w8a8_triton_block_scaled_mm_fake,
)


def _padded_cutlass(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    pad_multiple = 4
    dim = qx.shape[0]
    padded = (
        dim if dim % pad_multiple == 0 else dim + pad_multiple - (dim % pad_multiple)
    )

    has_pad = padded > dim

    if has_pad:
        padded_shape = [padded, *qx.shape[1:]]
        padded_qx = torch.zeros(padded_shape, device=qx.device, dtype=qx.dtype)
        padded_qx[0 : qx.shape[0], ...].copy_(qx)

        padded_x_scale_shape = [*x_scale.shape[1:], padded]
        padded_x_scale = torch.ones(
            padded_x_scale_shape, device=x_scale.device, dtype=x_scale.dtype
        ).permute(-1, -2)
        padded_x_scale[0 : x_scale.shape[0], ...].copy_(x_scale)

        output = cutlass_scaled_mm(
            padded_qx, weight, padded_x_scale, weight_scale, block_size, output_dtype
        )
        return output[0 : qx.shape[0], ...]
    else:
        return cutlass_scaled_mm(
            qx, weight, x_scale, weight_scale, block_size, output_dtype
        )


def _padded_cutlass_fake(
    qx: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (qx.size(0), weight.size(0)), dtype=output_dtype, device=qx.device
    )


direct_register_custom_op(
    "padded_cutlass",
    _padded_cutlass,
    fake_impl=_padded_cutlass_fake,
)


def _fp8_gemm_nt_op(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
    )


def _fp8_gemm_nt_op_fake(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    use_deep_gemm_e8m0: bool,
) -> None:
    return None


direct_register_custom_op(
    "fp8_gemm_nt_op",
    _fp8_gemm_nt_op,
    mutates_args=["output"],
    fake_impl=_fp8_gemm_nt_op_fake,
)


def _triton_per_token_group_quant_fp8_impl(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return per_token_group_quant_fp8(
        x, group_size, column_major_scales=False, use_ue8m0=False
    )


def _triton_per_token_group_quant_fp8_fake(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=current_platform.fp8_dtype(), device=x.device)
    out_bs = torch.empty(
        (
            M,
            (N + group_size - 1) // group_size,
        ),
        dtype=torch.float32,
        device=x.device,
    )
    return x_fp8, out_bs


direct_register_custom_op(
    "triton_per_token_group_quant_fp8",
    _triton_per_token_group_quant_fp8_impl,
    fake_impl=_triton_per_token_group_quant_fp8_fake,
)


def _flashinfer_fp8_blockscale_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    """
    Conditional FlashInfer FP8 blockscale GEMM with batch-size-dependent selection.

    This function switches between two optimized kernels based on the input batch size:
    - For small batches (M < 32): Uses FlashInfer's DeepGEMM swapAB optimization.
    - For larger batches (M >= 32): Uses the official DeepGEMM kernel.

    The conditional logic must use torch.cond() instead of a simple if-else statement
    to maintain compatibility with torch.compile graph compilation.

    This batch-size-dependent selection is essential for maintaining model accuracy.
    Benchmarks on GSM8K show a significant accuracy gap (88% vs 95%) for DeepSeek-V3.1
    when using FlashInfer's DeepGEMM on M>=32. The M < 32 strategy fixes the accurracy
    drop.

    Args:
        input: Input tensor of shape (batch_size, input_dim) in FP8 format
        weight: Weight tensor of shape (output_dim, input_dim) in FP8 format
        weight_scale: Scale factors for weight quantization (per-group)
        group_size: Quantization group size for the weight tensor
        use_deep_gemm_e8m0: Whether to use the E8M0 format in DeepGEMM quantization

    Returns:
        Output tensor of shape (batch_size, output_dim) in bfloat16 format
    """

    def run_flashinfer_deepgemm_swapAB(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return flashinfer_fp8_blockscale_gemm(
            input=input,
            weight=weight,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
        )

    def run_deepgemm(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        q_input, input_scale = per_token_group_quant_fp8(
            input,
            group_size=group_size,
            column_major_scales=True,
            use_ue8m0=use_deep_gemm_e8m0,
        )
        output = torch.empty(
            (q_input.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=q_input.device,
        )
        fp8_gemm_nt(
            (q_input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )
        return output

    condition = input.shape[0] < 32

    # PyTorch's torch.compile cannot handle input-dependent control flow in standard
    # Python conditionals. torch.cond() explicitly registers both code paths in the
    # computation graph, allowing torch.compile to capture both branches.
    # without torch.cond, the M < 32 condition won't be able to be captured by torch
    # compile
    return torch.cond(
        condition,
        run_flashinfer_deepgemm_swapAB,
        run_deepgemm,
        (input, weight, weight_scale),
    )


def _flashinfer_fp8_blockscale_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    """
    Required fake/meta implementation for torch.compile graph tracing.
    """
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "flashinfer_fp8_blockscale_gemm",
    _flashinfer_fp8_blockscale_gemm_impl,
    fake_impl=_flashinfer_fp8_blockscale_gemm_fake,
)


# TODO fix ROCm->Triton custom path:
#  https://github.com/vllm-project/vllm/issues/14397
class W8A8BlockFp8LinearOp:
    """
    This class executes a Blocked FP8 linear layer using cutlass if supported
    and torch.scaled_mm otherwise.
    """

    def __init__(
        self,
        weight_group_shape: GroupShape,
        act_quant_group_shape: GroupShape,
        cutlass_block_fp8_supported: bool = CUTLASS_BLOCK_FP8_SUPPORTED,
        use_aiter_and_is_supported: bool = False,
    ):
        self.weight_group_shape = weight_group_shape
        self.act_quant_group_shape = act_quant_group_shape
        self.is_deep_gemm_supported = is_deep_gemm_supported()
        self.is_hopper = current_platform.is_device_capability(90)
        self.use_deep_gemm_e8m0 = is_deep_gemm_e8m0_used()
        self.is_flashinfer_supported = is_flashinfer_fp8_blockscale_gemm_supported()

        # Get the correct blockscale mul and input quant operations.
        # We can't use _dispatch_w8a8_blockscale_op to figure out if we want
        # to use deepgemm because we don't know the shape of weights (and
        # whether deepgemm supports it) at the init time.
        self.w8a8_blockscale_op, self.input_quant_op = (
            self._dispatch_w8a8_blockscale_op(
                cutlass_block_fp8_supported, use_aiter_and_is_supported
            )
        )
        self.deepgemm_input_quant_op = (
            QuantFP8(
                False,
                self.act_quant_group_shape,
                column_major_scales=True,
                tma_aligned_scales=True,
                use_ue8m0=self.use_deep_gemm_e8m0,
            )
            if self.is_deep_gemm_supported
            else None
        )

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input_scale is None
        # View input as 2D matrix for fp8 methods
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[0]]
        output_dtype = input.dtype

        if should_use_flashinfer_for_blockscale_fp8_gemm(
            self.is_flashinfer_supported, output_dtype, input_2d, weight
        ) and should_use_deepgemm_for_fp8_linear(
            output_dtype, weight, self.is_deep_gemm_supported
        ):
            output = self._run_flashinfer(input_2d, weight, weight_scale)

        elif should_use_deepgemm_for_fp8_linear(
            output_dtype, weight, self.is_deep_gemm_supported
        ):
            output = self._run_deepgemm(input_2d, weight, weight_scale)
        else:
            output = self.w8a8_blockscale_op(
                input_2d, weight, weight_scale, input_scale
            )

        if bias is not None:
            output = output + bias
        return output.to(dtype=input.dtype).view(*output_shape)

    def _run_deepgemm(
        self,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        if DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0:
            q_input, input_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                input_2d,
                group_size=self.act_quant_group_shape.col,
                use_ue8m0=True,
            )
        else:
            assert self.deepgemm_input_quant_op is not None
            q_input, input_scale = self.deepgemm_input_quant_op(input_2d)
        output = torch.empty(
            (q_input.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=q_input.device,
        )
        torch.ops.vllm.fp8_gemm_nt_op(
            q_input, input_scale, weight, weight_scale, output, self.use_deep_gemm_e8m0
        )
        return output

    def _run_cutlass(
        self,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input_scale is None
        assert self.input_quant_op is not None
        q_input, input_scale = self.input_quant_op(input_2d)
        if self.is_hopper:
            return torch.ops.vllm.padded_cutlass(
                q_input,
                weight,
                input_scale,
                weight_scale,
                list(self.weight_group_shape),
                input_2d.dtype,
            )
        else:
            return cutlass_scaled_mm(
                q_input,
                weight,
                input_scale,
                weight_scale,
                list(self.weight_group_shape),
                input_2d.dtype,
            )

    def _run_aiter(
        self,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.act_quant_group_shape == GroupShape(1, 128)

        n, k = weight.shape

        use_triton = (
            not current_platform.is_fp8_fnuz()
            and rocm_aiter_ops.is_triton_gemm_w8a8_tuned(n, k)
        )

        if use_triton:
            gemm_a8w8_blockscale_op = rocm_aiter_ops.triton_gemm_a8w8_blockscale
        else:
            gemm_a8w8_blockscale_op = rocm_aiter_ops.gemm_a8w8_blockscale

        if input_scale is not None:
            q_input = input_2d
        elif use_triton:
            q_input, input_scale = torch.ops.vllm.triton_per_token_group_quant_fp8(
                input_2d,
                self.act_quant_group_shape.col,
            )
        else:
            q_input, input_scale = rocm_aiter_ops.group_fp8_quant(
                input_2d, self.act_quant_group_shape.col
            )

        return gemm_a8w8_blockscale_op(
            q_input,
            weight,
            input_scale,
            weight_scale,
            list(self.weight_group_shape),
            output_dtype=input_2d.dtype,
        )

    def _run_triton(
        self,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert input_scale is None
        assert self.input_quant_op is not None
        q_input, input_scale = self.input_quant_op(input_2d)
        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
            q_input,
            weight,
            input_scale,
            weight_scale,
            list(self.weight_group_shape),
            input_2d.dtype,
        )

    def _run_flashinfer(
        self,
        input_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run FlashInfer FP8 block-scale GEMM.

        This backend uses TensorRT-LLM's FP8 block-scale GEMM kernels
        and supports FP8+FP8 (W8A8 full quantization) on SM90+ (Hopper).
        """
        # Now call FlashInfer with BF16 input + FP8 weight, input will be
        # quantized with FlashInfer kernel (W8A8)
        output = torch.ops.vllm.flashinfer_fp8_blockscale_gemm(
            input=input_2d,  # BF16 input
            weight=weight,  # FP8 weight
            weight_scale=weight_scale,  # Weight scales
            group_size=self.act_quant_group_shape.col,
            use_deep_gemm_e8m0=self.use_deep_gemm_e8m0,
        )
        return output

    def _dispatch_w8a8_blockscale_op(
        self,
        use_cutlass: bool,
        use_aiter_and_is_supported: bool,
    ) -> tuple[
        Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ],
            torch.Tensor,
        ],
        QuantFP8 | None,
    ]:
        if use_cutlass:
            return self._run_cutlass, (
                QuantFP8(
                    False,
                    self.act_quant_group_shape,
                    column_major_scales=True,
                    use_ue8m0=False,
                )
            )
        if use_aiter_and_is_supported:
            return self._run_aiter, None
        return self._run_triton, (
            QuantFP8(
                False,
                self.act_quant_group_shape,
                column_major_scales=False,
                use_ue8m0=False,
            )
        )


def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values "
    "with tensor-wise quantization."""
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


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
def _silu_mul_per_token_group_quant_fp8_colmajor(
    y_ptr,  # [M, N]
    y_q_ptr,  # [M, N // 2]
    y_s_ptr,  # [M, (N // 2) // GROUP_SIZE]
    M,  # num tokens
    N,  # intermediate size
    # Stride
    y_s_col_stride: tl.int64,
    # Information for float8
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    # Meta-parameters
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # TODO(varun) : Add expert_ids so we may early-exit no-op thread blocks.
    """
    Each thread block (BLOCK_N) computes [BLOCK_M, GROUP_SIZE] act-mul outputs. Then
    the thread block quantizes the [BLOCK_M, GROUP_SIZE] block of values and fills
    the outputs tensors at the right positions.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    N_2 = N // 2

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    if m_offset >= M:
        return

    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)

    base_y_ptr = y_ptr + m_offset * N + n_offset

    act_in_ptrs = base_y_ptr + offs_m[:, None] * N + offs_n[None, :]

    act_in = tl.load(act_in_ptrs)
    mul_in = tl.load(act_in_ptrs + N_2)

    # silu & mul
    act_in = act_in.to(tl.float32)
    one_f32 = tl.cast(1, tl.float32)
    silu_out = (act_in / (one_f32 + tl.exp(-act_in))).to(y_ptr.dtype.element_ty)
    y = (silu_out * mul_in).to(tl.float32)

    # quant
    _absmax = tl.maximum(tl.max(tl.abs(y), axis=1), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_s = tl.reshape(y_s, (BLOCK_M, 1))
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    # store y_q
    base_y_q_ptr = y_q_ptr + m_offset * N_2 + n_offset
    y_q_ptrs = base_y_q_ptr + offs_m[:, None] * N_2 + offs_n[None, :]
    tl.store(y_q_ptrs, y_q)

    # store y_s
    group_id = n_offset // GROUP_SIZE
    base_y_s_ptr = y_s_ptr + group_id * y_s_col_stride + m_offset
    y_s_ptrs = base_y_s_ptr + offs_m
    y_s = tl.reshape(y_s, (BLOCK_M,))
    tl.store(y_s_ptrs, y_s)


def silu_mul_per_token_group_quant_fp8_colmajor(
    input: torch.Tensor,  # [M, N]
    output: torch.Tensor | None = None,  # [M, N // 2]
    use_ue8m0: bool | None = None,
    eps: float = 1e-10,
):
    """
    silu+mul + block-fp8 quant with group size 128.
    """
    GROUP_SIZE = 128
    assert input.ndim == 2
    if output is not None:
        assert output.ndim == 2
    assert input.size(0) % GROUP_SIZE == 0
    assert input.size(1) % (GROUP_SIZE * 2) == 0

    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()

    M, N = input.size()
    N_2 = N // 2

    fp8_dtype = current_platform.fp8_dtype()
    if output is None:
        output = torch.empty((M, N_2), dtype=fp8_dtype, device=input.device)

    output_scales = torch.empty(
        ((N_2 // GROUP_SIZE), M), dtype=torch.float32, device=input.device
    ).transpose(0, 1)

    BLOCK_M = 8
    BLOCK_N = GROUP_SIZE
    assert M % BLOCK_M == 0
    assert N_2 % BLOCK_N == 0

    # Using the default value (240.0) from pytorch will cause accuracy
    # issue on dynamic quantization models. Here use 224.0 for fnuz on ROCm
    # platforms that use the torch.float8_e4m3fnuz dtype.
    finfo = torch.finfo(fp8_dtype)
    fp8_min = -224.0 if current_platform.is_fp8_fnuz() else finfo.min
    fp8_max = 224.0 if current_platform.is_fp8_fnuz() else finfo.max

    # Force even division so we can avoid edgecases within the kernel.
    assert M % BLOCK_M == 0
    assert N_2 % BLOCK_N == 0
    grid = (M // BLOCK_M, N_2 // BLOCK_N)

    _silu_mul_per_token_group_quant_fp8_colmajor[grid](
        input,
        output,
        output_scales,
        M,
        N,
        output_scales.stride(-1),
        eps,
        fp8_min,
        fp8_max,
        use_ue8m0,
        GROUP_SIZE,
        BLOCK_M,
        BLOCK_N,
    )

    return output, output_scales


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


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
        column_major_scales: Outputs scales in column major.
        tma_aligned_scales: Outputs scales in TMA-aligned layout.
        out_q: Optional output tensor. If not provided, function will create.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()
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
        if tma_aligned_scales:
            m = x.shape[-2]
            sf_k = x.shape[-1] // group_size
            tma_aligned_m = get_tma_aligned_size(m, 4)
            shape = x.shape[:-2] + (m, sf_k)
            stride = (
                (1, tma_aligned_m)
                if x.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            x_s = torch.empty_strided(
                shape, stride, device=x.device, dtype=torch.float32
            )
        else:
            shape = x.shape[:-2] + (x.shape[-1] // group_size, x.shape[-2])
            x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(
                -1, -2
            )
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # prefer CUDA kernel if available
    # TODO(bnell): this causes some fp8 moe test to fail.
    if current_platform.is_cuda() and x.is_contiguous():
        torch.ops._C.per_token_group_fp8_quant(
            x, x_q, x_s, group_size, eps, fp8_min, fp8_max, use_ue8m0
        )
        return x_q, x_s

    # TRITON FALLBACK
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


def per_token_group_quant_fp8_packed_for_deepgemm(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    use_ue8m0: bool | None = None,
    out_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 per-token-group quantization for DeepGEMM.

    Returns:
        (x_q, x_s_packed)
            x_q: FP8 activations, same shape as `x`.
            x_s_packed: Int32 tensor with logical shape
                        [mn, ceil(num_groups_per_row / 4)], laid out with
                        TMA-aligned stride along the packed-K dimension
    """
    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    # for DeepGEMM UE8M0-packed layout we *require* UE8M0 scales.
    assert use_ue8m0, (
        "per_token_group_quant_fp8_packed_for_deepgemm requires UE8M0 scales."
    )

    dtype = current_platform.fp8_dtype()
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    # compute DeepGEMM-style packed scale tensor shape.
    hidden_dim = x.shape[-1]
    mn = x.numel() // hidden_dim
    num_groups_per_row = hidden_dim // group_size
    k_num_packed_sf_k = (num_groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4

    x_s_packed = torch.empty_strided(
        (mn, k_num_packed_sf_k),
        (1, tma_aligned_mn),
        device=x.device,
        dtype=torch.int32,
    )

    # CUDA kernel path only (DeepGEMM + E8M0 is CUDA-specific).
    assert current_platform.is_cuda(), (
        "per_token_group_quant_fp8_packed_for_deepgemm is only valid on CUDA "
        "platforms using DeepGEMM."
    )

    x_contiguous = x.contiguous()
    if out_q is not None:
        x_q_local = out_q
    else:
        x_q_local = torch.empty_like(x_contiguous, device=x.device, dtype=dtype)

    torch.ops._C.per_token_group_fp8_quant_packed(
        x_contiguous,
        x_q_local,
        x_s_packed,
        group_size,
        eps,
        fp8_min,
        fp8_max,
    )

    # return a tensor with the original logical shape.
    x_q = x_q_local.view_as(x)
    return x_q, x_s_packed


@triton.jit
def _w8a8_triton_block_scaled_mm(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and
    store the result in output tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.
    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n},{block_k}].json"  # noqa: E501

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for W8A8 Block FP8 kernel.",
                config_file_path,
            )
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        "Using default W8A8 Block FP8 kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        config_file_path,
    )
    return None


def w8a8_triton_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization.
    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should
        be 2-dim, e.g., [128, 128].
        output_dytpe: The dtype of the returned tensor.
    Returns:
        torch.Tensor: The result of matmul.
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    configs = get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])
    if configs:
        # Get the optimal config if there is one
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default config
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_size[0]
        # BLOCK_SIZE_K must be divisible by block_size[1]
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return C


def requant_weight_ue8m0_inplace(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: Sequence[int] = (128, 128),
) -> None:
    """Re-quantise *weight* so that its per-block scaling factors are in the
    UE8M0 (power-of-two) format expected by the new DeepGEMM kernels inplace.

    Args:
        weight: Block-quantised weight tensor stored in `torch.float8_e4m3fn`.
            Expected shape `(..., M, K)`.
        weight_scale: Corresponding per-block scale tensor (`torch.float32`)
            with shape `(..., M // block_size[0], K // block_size[1])`.
        block_size: 2-element iterable `[block_m, block_k]` describing the
            block quantisation granularity.
    """
    if weight.numel() == 0:
        return

    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"Expected *weight* to be torch.float8_e4m3fn, got {weight.dtype} instead."
        )

    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    block_m, block_k = int(block_size[0]), int(block_size[1])

    # Flatten leading dimensions so we can iterate over the last two dims.
    leading_shape = weight.shape[:-2]
    if len(leading_shape) == 0:
        w_view = weight.unsqueeze(0)
        s_view = weight_scale.unsqueeze(0)
    else:
        w_view = weight.reshape(-1, weight.shape[-2], weight.shape[-1])
        s_view = weight_scale.reshape(-1, *weight_scale.shape[-2:])

    num_mats = w_view.size(0)
    for idx in range(num_mats):
        w_q = w_view[idx]
        s_old = s_view[idx]

        # De-quantise with the *old* scaling factors (float32).
        m_cur, k_cur = w_q.shape
        s_float = s_old.to(torch.float32)
        # Expand scales along rows and cols by block size, then crop.
        s_exp_r = torch.repeat_interleave(s_float, block_m, dim=0)
        s_exp = torch.repeat_interleave(s_exp_r, block_k, dim=1)
        s_exp = s_exp[:m_cur, :k_cur]
        w_dq = w_q.to(torch.float32) * s_exp
        # Re-quantise using power-of-two scaling (UE8M0).
        w_requant, s_requant = per_block_cast_to_fp8(
            w_dq, [block_m, block_k], use_ue8m0=True
        )

        # Write back the results in-place.
        w_q.copy_(w_requant)
        s_old.copy_(s_requant)


def deepgemm_post_process_fp8_weight_block(
    wq: torch.Tensor, ws: torch.Tensor, quant_block_shape: tuple[int], use_e8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    assert wq.dtype == torch.float8_e4m3fn, (
        "Expected quantized tensor dtype "
        f"to be torch.float8_e4m3fn, got {wq.dtype} instead."
    )
    assert ws.dtype == torch.float32, (
        f"Expected tensor scales dtype to be torch.float32, got {ws.dtype} instead"
    )

    if use_e8m0:
        requant_weight_ue8m0_inplace(wq, ws, block_size=quant_block_shape)

    original_ndim = wq.ndim
    if wq.ndim == 2:
        assert ws.ndim == 2
        wq = wq.unsqueeze(0)
        ws = ws.unsqueeze(0)

    # From https://github.com/deepseek-ai/DeepGEMM/blob/c9f8b34dcdacc20aa746b786f983492c51072870/csrc/utils/layout.hpp#L46
    recipe = (1, 128, 128)

    # Ref : https://github.com/deepseek-ai/DeepGEMM/blob/c9f8b34dcdacc20aa746b786f983492c51072870/csrc/apis/gemm.hpp
    # DeepGemm uses the `transform_sf_into_required_layout` function to
    # represent scales in the correct format.
    dg_ws = transform_sf_into_required_layout(
        sf=ws,
        mn=wq.size(1),
        k=wq.size(2),
        recipe=recipe,
        num_groups=wq.size(0),
        # is the scale factors for A in (Refers to the argument A in A @ B).
        # Weights are B.
        is_sfa=False,
    )

    if original_ndim == 2:
        wq = wq.squeeze(0)
        dg_ws = dg_ws.squeeze(0)

    return wq, dg_ws


def prepare_fp8_moe_layer_for_deepgemm(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: tuple[int],
):
    w13, w13_scale = deepgemm_post_process_fp8_weight_block(
        wq=w13,
        ws=w13_scale,
        quant_block_shape=block_shape,
        use_e8m0=is_deep_gemm_e8m0_used(),
    )
    w2, w2_scale = deepgemm_post_process_fp8_weight_block(
        wq=w2,
        ws=w2_scale,
        quant_block_shape=block_shape,
        use_e8m0=is_deep_gemm_e8m0_used(),
    )

    return w13, w2, w13_scale, w2_scale


def _maybe_pad_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pad the weight tensor. This is an optimization on ROCm platform, which
    can benefit from tensors located far enough from one another in memory"""
    if (
        envs.VLLM_ROCM_FP8_PADDING
        and current_platform.is_rocm()
        and weight.stride(-1) == 1
        and (weight.stride(-2) * weight.element_size()) % 512 == 0
    ):
        num_pad = 256 // weight.element_size()
        import torch.nn.functional as F

        weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
        torch.cuda.empty_cache()
    return weight


def validate_fp8_block_shape(
    layer: torch.nn.Module,
    input_size: int,
    output_size: int,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    block_size: list[int],
) -> None:
    """Validate block quantization shapes for tensor parallelism."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    if getattr(layer, "allow_fp8_block_shape_mismatch", False):
        logger.debug(
            "Skipping FP8 block shape validation for layer %s due to detected"
            " mismatch allowance.",
            getattr(layer, "prefix", "<unknown>"),
        )
        return

    tp_size = getattr(layer, "tp_size", get_tensor_model_parallel_world_size())
    block_n, block_k = block_size[0], block_size[1]

    # Required by row parallel
    if (
        tp_size > 1
        and input_size // input_size_per_partition == tp_size
        and input_size_per_partition % block_k != 0
    ):
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition} "
            f"is not divisible by weight quantization block_k = {block_k}."
        )

    # Required by column parallel or enabling merged weights
    is_tp_split = tp_size > 1 and output_size // sum(output_partition_sizes) == tp_size
    is_merged_gemm = len(output_partition_sizes) > 1
    if is_tp_split or is_merged_gemm:
        sizes_to_check = output_partition_sizes
        if not is_tp_split and is_merged_gemm:
            # In case of merged matrices, we allow the last
            # matrix to not be a multiple of block size
            sizes_to_check = output_partition_sizes[:-1]
        for output_partition_size in sizes_to_check:
            if output_partition_size % block_n != 0:
                raise ValueError(
                    f"Weight output_partition_size = "
                    f"{output_partition_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )


def create_fp8_weight_parameter(
    output_size_per_partition: int,
    input_size_per_partition: int,
    weight_loader: Callable | None,
) -> torch.nn.Parameter:
    """Create FP8 weight parameter."""
    from vllm.model_executor.parameter import ModelWeightParameter

    return ModelWeightParameter(
        data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn,
        ),
        input_dim=1,
        output_dim=0,
        weight_loader=weight_loader,
    )


def create_fp8_scale_parameter(
    parameter_type: torch.nn.Parameter,
    output_partition_sizes: list[int],
    input_size_per_partition: int,
    block_size: list[int] | None,
    weight_loader: Callable | None,
) -> torch.nn.Parameter:
    """Create scale parameter based on quantization strategy."""
    if parameter_type == ChannelQuantScaleParameter:
        scale = parameter_type(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
    elif parameter_type == BlockQuantScaleParameter:
        assert block_size is not None
        block_n, block_k = block_size[0], block_size[1]
        output_size_per_partition = sum(output_partition_sizes)
        scale = parameter_type(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
    elif parameter_type == PerTensorScaleParameter:
        scale = parameter_type(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
    else:
        raise ValueError(f"Unknown parameter type: {parameter_type}")

    scale[:] = torch.finfo(torch.float32).min
    return scale


def create_fp8_input_scale(
    output_partition_sizes: list[int], weight_loader: Callable | None
) -> torch.nn.Parameter:
    """Create input scale parameter for static activation quantization."""
    from vllm.model_executor.parameter import PerTensorScaleParameter

    scale = PerTensorScaleParameter(
        data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
        weight_loader=weight_loader,
    )
    scale[:] = torch.finfo(torch.float32).min
    return scale


def process_fp8_weight_tensor_strategy(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_widths: list[int],
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Process weights for tensor-wise quantization strategy."""
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        normalize_e4m3fn_to_e4m3fnuz,
        requantize_with_max_scale,
    )

    if current_platform.is_fp8_fnuz():
        weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
            weight=weight, weight_scale=weight_scale, input_scale=input_scale
        )

    # Requantize with max scale
    weight_scale, weight = requantize_with_max_scale(
        weight=weight,
        weight_scale=weight_scale,
        logical_widths=logical_widths,
    )

    weight = _maybe_pad_fp8_weight(weight)
    return weight, weight_scale, input_scale


def process_fp8_weight_channel_strategy(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Process weights for channel-wise quantization strategy."""
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        normalize_e4m3fn_to_e4m3fnuz,
    )

    if current_platform.is_fp8_fnuz():
        weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
            weight=weight, weight_scale=weight_scale, input_scale=input_scale
        )

    return weight, weight_scale, input_scale


def process_fp8_weight_block_strategy(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process weights for block-wise quantization strategy."""
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        normalize_e4m3fn_to_e4m3fnuz,
    )

    if current_platform.is_fp8_fnuz():
        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
            weight=weight, weight_scale=weight_scale
        )

    weight = _maybe_pad_fp8_weight(weight)
    return weight, weight_scale


def maybe_post_process_fp8_weight_block(layer: torch.nn.Module):
    assert layer.weight_block_size is not None

    from vllm.utils.deep_gemm import (
        is_deep_gemm_e8m0_used,
        should_use_deepgemm_for_fp8_linear,
    )

    # On Blackwell or Hopper, if E8M0 for DeepGemm is used, we need to
    # requantize the weight and input to the specific scale
    # at the same time.
    should_use_deepgemm = should_use_deepgemm_for_fp8_linear(
        layer.orig_dtype, layer.weight
    )
    if should_use_deepgemm:
        scale_attr = (
            "weight_scale_inv" if hasattr(layer, "weight_scale_inv") else "weight_scale"
        )
        dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=layer.weight.data,
            ws=getattr(layer, scale_attr).data,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=is_deep_gemm_e8m0_used(),
        )
        replace_parameter(layer, "weight", dg_weight)
        replace_parameter(layer, scale_attr, dg_weight_scale)


def process_fp8_weight_tensor_strategy_moe(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    shard_size: int,
    num_experts: int,
    is_act_and_mul: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process moe weights for tensor-wise quantization strategy."""
    max_scales = weight_scales.max(dim=1).values

    # For w1 case (i.e. not w13): there is already just one scale per expert.
    if not is_act_and_mul:
        assert weight_scales.shape[1] == 1
        # One scale per expert
        assert max_scales.shape == (num_experts,)
        return weight, max_scales

    # For w13 case (common): require single scale for w13 per expert, but
    # on disk there is a scale for w1 and w3. Use the max to requantize.
    for expert_id in range(num_experts):
        start = 0
        for shard_id in range(2):
            dq_weight = per_tensor_dequantize(
                weight[expert_id][start : start + shard_size, :],
                weight_scales[expert_id][shard_id],
            )
            weight[expert_id][start : start + shard_size, :], _ = ops.scaled_fp8_quant(
                dq_weight, max_scales[expert_id]
            )
            start += shard_size
    return weight, max_scales


def process_fp8_input_tensor_strategy_moe(
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process moe input scales for tensor-wise quantization strategy."""

    if not all_close_1d(w13_input_scale) or not all_close_1d(w2_input_scale):
        logger.info_once(
            "Found input_scales that are not equal for "
            "fp8 MoE layer. Using the maximum across experts "
            "for each layer."
        )

    return w13_input_scale.max(), w2_input_scale.max()
