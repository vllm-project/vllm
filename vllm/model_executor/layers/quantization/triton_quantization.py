import torch
import triton
import triton.language as tl
from vllm.triton_utils import tl, triton

@triton.jit
def silu_and_mul_per_block_quant_kernel(
    output_ptr,        # [num_tokens, hidden_size]
    scales_ptr,        # [num_tokens, num_groups]
    input_ptr,         # [num_tokens, hidden_size * 2]
    scale_ub_ptr,      # Optional pointer to a single float
    hidden_size,       
    stride_input_m: tl.int64,    # input.stride(0)
    stride_output_m: tl.int64,   # output.stride(0)
    stride_scale_m: tl.int64,    # scales.stride(0)
    stride_scale_g: tl.int64,    # scales.stride(1)
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_SCALE_UB: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_input_ptr  = input_ptr + row_idx * stride_input_m
    row_output_ptr = output_ptr + row_idx * stride_output_m
    row_scale_ptr  = scales_ptr + row_idx * stride_scale_m

    # Optional scale_ub
    if HAS_SCALE_UB:
        ub_val = tl.load(scale_ub_ptr)
    else:
        ub_val = 1e10

    # Iterate groups
    for group_idx in tl.range(0, hidden_size, group_size):
        offsets = group_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size

        gate = tl.load(row_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up   = tl.load(row_input_ptr + offsets + hidden_size, mask=mask, other=0.0).to(tl.float32)
        silu_mul = gate * tl.sigmoid(gate) * up

        # Calc group scale
        max_val = tl.max(tl.abs(silu_mul), axis=0)

        scale = tl.minimum(max_val / 448.0, ub_val)
        scale = tl.maximum(scale, 1e-10)

        # Store scale by group
        g_id = group_idx // group_size
        tl.store(row_scale_ptr + g_id * stride_scale_g, scale)

        # Quant
        silu_mul_quant = (silu_mul / scale).to(output_ptr.dtype.element_ty)
        tl.store(row_output_ptr + offsets, silu_mul_quant, mask=mask)

def silu_and_mul_per_block_quant_triton(
    input: torch.Tensor,
    group_size: int,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = input.size(0)
    hidden_size = input.size(1) // 2
    num_groups = hidden_size // group_size

    # Allocate output tensors
    output = torch.empty((num_tokens, hidden_size), device=input.device, dtype=quant_dtype)
    
    if is_scale_transposed:
        scales = torch.empty((num_groups, num_tokens), device=input.device, dtype=torch.float32)
        scales_stride_m = scales.stride(1)
        scales_stride_g = scales.stride(0)
    else:
        scales = torch.empty((num_tokens, num_groups), device=input.device, dtype=torch.float32)
        scales_stride_m = scales.stride(0)
        scales_stride_g = scales.stride(1)

    grid = (num_tokens,)
    BLOCK_SIZE = triton.next_power_of_2(group_size)

    silu_and_mul_per_block_quant_kernel[grid](
        output, scales, input, scale_ub,
        hidden_size,
        input.stride(0), output.stride(0),
        scales_stride_m, scales_stride_g,
        group_size, BLOCK_SIZE,
        HAS_SCALE_UB=(scale_ub is not None),
    )

    return output, scales
