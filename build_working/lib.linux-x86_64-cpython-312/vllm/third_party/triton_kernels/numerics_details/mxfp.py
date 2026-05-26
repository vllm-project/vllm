# isort: off
# fmt: off
from enum import Enum
import triton
import torch
import torch.nn.functional as F
from .mxfp_details._upcast_from_mxfp import _upcast_from_mxfp
from .mxfp_details._downcast_to_mxfp import _downcast_to_mxfp, MXFP_BLOCK_SIZE, _quantize_mxfp8_fn

# -----------------------------------------------------------------------------
#                      Dequantization / Quantization Utilities
# -----------------------------------------------------------------------------


class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1


def downcast_to_mxfp(src_tensor: torch.Tensor, out_quant_type: torch.dtype, axis: int,
                     DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP):
    """
         Convert the src weights to mx format. The src weight is quantized along the axis dimension.

         If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
         Note that this means the k_dim of the tensor will be half of the logical k_dim.

         If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored
         in their respective formats.
    """
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    # downcast
    src_tensor = src_tensor.transpose(axis, src_tensor.ndim - 1)
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = out_quant_type in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert is_fp4 or is_fp8
    divisor = 2 if is_fp4 else 1
    L = src_tensor.shape[-1]
    if is_fp4:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"
    out_shape = src_tensor.shape[:-1] + (L // divisor, )
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, MXFP_BLOCK_SIZE), )

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)

    if src_tensor.numel() > 0:
        kernel_src_tensor = src_tensor.reshape(-1, src_tensor.shape[-1])
        kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        kernel_scale = out_scale.view(-1, out_scale.shape[-1])

        BLOCK_OUT_DIM = 128
        BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value
        grid_out = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
        grid_quant = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

        _downcast_to_mxfp[(grid_out, grid_quant)](kernel_quant_tensor, *kernel_quant_tensor.stride(), kernel_scale,
                                                *kernel_scale.stride(), kernel_src_tensor, *kernel_src_tensor.stride(),
                                                *kernel_src_tensor.shape, BLOCK_OUT_DIM, BLOCK_QUANT_DIM,
                                                DEQUANT_SCALE_ROUNDING_MODE.value, num_warps=8)

    out_quant_tensor = out_quant_tensor.transpose(axis, src_tensor.ndim - 1)
    out_scale = out_scale.transpose(axis, src_tensor.ndim - 1)
    return out_quant_tensor, out_scale


def upcast_from_mxfp(tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype, axis: int):
    """
    Upcasts an mxfp (packed) weight tensor back to float16 or bfloat16.

    The function assumes that the tensors were quantized along the given axis.
    It permutes the tensor so that the quantized axis is last, reshapes to 2D,
    launches the Triton upcast kernel, and then unpermutes back to the original order.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    assert tensor.ndim == scale.ndim, (f"Weight and scale must have the same number of dimensions. "
                                       f"Got {tensor.ndim=} and {scale.ndim=}")
    # dtype checks
    assert tensor.dtype in {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}, \
        f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert target_dtype in (torch.float16, torch.bfloat16, torch.float32), f"Invalid output dtype {target_dtype=}"
    # upcast
    logical_quant_dim = tensor.shape[axis] * (2 if tensor.dtype == torch.uint8 else 1)
    tensor = tensor.transpose(axis, tensor.ndim - 1).contiguous()
    scale = scale.transpose(axis, scale.ndim - 1).contiguous()
    out = torch.empty((*tensor.shape[:-1], logical_quant_dim), dtype=target_dtype, device=tensor.device)
    reshaped_out = out.view(-1, out.shape[-1])
    reshaped_tensor = tensor.view(-1, tensor.shape[-1])
    reshaped_scale = scale.view(-1, scale.shape[-1])
    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value
    blocks_out_dim = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)
    _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](reshaped_out, *reshaped_out.stride(), reshaped_scale,
                                                          *reshaped_scale.stride(), reshaped_tensor,
                                                          *reshaped_tensor.stride(), *reshaped_out.shape, BLOCK_OUT_DIM,
                                                          BLOCK_QUANT_DIM, num_warps=8)
    out = out.transpose(axis, scale.ndim - 1).contiguous()
    return out


# ------------


def right_shift_unsigned(x, shift):
    # CUDA torch does not support bit ops on uint32, so we need to mask to get unsigned right shift
    return (x >> shift) & ((1 << (32 - shift)) - 1)


def get_max_quant_val(dtype: torch.dtype):
    d = {torch.uint8: 6.0, torch.float8_e5m2: 57344.0, torch.float8_e4m3fn: 448.0}
    assert dtype in d
    return d[dtype]


def downcast_to_mxfp_torch(src_tensor: torch.Tensor, out_quant_type: torch.dtype, axis: int,
                           DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP):
    """
    Converts the src tensor to the output format specified by out_quant_type.
      axis: The axis along which the tensors are contiguous and quantization is applied.
      DEQUANT_SCALE_ROUNDING_MODE: 0 for ROUND_UP, 1 for ROUND_DOWN.

    Returns:
      out_quant_tensor: Quantized tensor in mx format.
         • For mxfp8, the output has the same shape as src_tensor.
         • For mxfp4, the size along the axis is halved, and the tensor is returned as a torch.uint8.
      scale: Scale tensor (stored as uint8) computed per group of 32 elements along the axis.
             Its shape is the same as src_tensor except that the axis is replaced by ceil(L/32),
             where L is the original length along that axis.
    """
    # This should probably be packed into its own tiny class
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    assert src_tensor.dtype in {torch.float32, torch.bfloat16,
                                torch.float16}, f"Invalid input tensor dtype {src_tensor.dtype}"

    axis = axis if axis >= 0 else axis + ndim
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = "float8" in str(out_quant_type)
    assert is_fp4 or is_fp8, f"Invalid input tensor dtype {out_quant_type}"

    device = src_tensor.device

    # For mxfp4 conversion, we assume the contiguous axis length is even.
    if is_fp4:
        axis_shape = src_tensor.size(axis)
        assert axis_shape % 2 == 0, "For mxfp4 conversion the contiguous axis length must be even."

    # Permute the tensor so that the contiguous axis becomes the last dimension.
    src = src_tensor.transpose(axis, src_tensor.ndim - 1).to(torch.float32)
    axis_shape = src.shape[-1]

    # Pad the axis to be divisible by 32, in case it is not.
    next_multiple = triton.cdiv(axis_shape, MXFP_BLOCK_SIZE) * MXFP_BLOCK_SIZE
    pad_amount = next_multiple - axis_shape
    padded_src = F.pad(src, (0, pad_amount))
    valid_mask = F.pad(torch.ones_like(src, dtype=torch.bool), (0, pad_amount))
    padded_axis_shape = padded_src.size(-1)  # now divisible by 32

    # --- Compute per-group maximums for scale ---
    # Set padded entries to -1 so they don’t affect the max.
    abs_f = torch.abs(padded_src)
    abs_f = torch.where(valid_mask, abs_f, torch.tensor(-1.0, device=device, dtype=padded_src.dtype))
    # Reshape the last dimension into groups of 32.
    new_shape = padded_src.shape[:-1] + (padded_axis_shape // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE)
    abs_groups = abs_f.view(*new_shape)
    # Compute maximum along the group dimension (of size 32).
    max_val, _ = abs_groups.max(dim=-1, keepdim=True)

    # Choose a max quantization value depending on type.
    max_quant_val = get_max_quant_val(out_quant_type)
    dequant_scale = max_val / max_quant_val  # shape: (..., padded_axis_shape//32, 1)

    # Convert to int to round the FP32 scale, prior to quantization!
    ds_int = dequant_scale.view(torch.int32)
    if DEQUANT_SCALE_ROUNDING_MODE == DequantScaleRoundingMode.ROUND_UP:
        ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
    else:
        ds_int_rounded = ds_int & 0x7F800000
    # Reinterpret back as float32.
    dequant_scale_rounded = ds_int_rounded.view(torch.float32)

    # Compute the quantization scale.
    quant_scale = torch.where(dequant_scale_rounded == 0, torch.tensor(0.0, device=device), 1.0 / dequant_scale_rounded)

    # Quantize the tensor
    orig_padded_shape = padded_src.shape
    padded_src_groups = padded_src.view(*new_shape)
    quant_tensor = padded_src_groups * quant_scale
    # Reshape back to the original shape and trim padding
    quant_tensor = quant_tensor.view(orig_padded_shape)
    quant_tensor = quant_tensor[..., :axis_shape]

    # Finally, convert the quantized tensor to the target format
    if is_fp8:
        # Conversion must use satfinite PTX, so clamp before the conversion in torch to emulate this behavior
        quant_tensor = torch.clamp(quant_tensor, -max_quant_val, max_quant_val)
        out_weight = quant_tensor.to(out_quant_type)
    else:
        assert is_fp4, f"Invalid output quantization type {out_quant_type}"
        # For mxfp4, perform bit-level manipulation and pack two 4-bit values per uint8.
        # First, reinterpret the quantized tensor bits.
        q_int = quant_tensor.contiguous().view(torch.int32)
        # Extract sign, exponent, and mantissa.
        signs = q_int & 0x80000000
        exponents = right_shift_unsigned(q_int, 23) & 0xFF
        mantissas = q_int & 0x7FFFFF

        E8_BIAS = 127
        E2_BIAS = 1
        # Adjust mantissas for subnormals.
        mantissas = torch.where(exponents < E8_BIAS, (0x400000 | right_shift_unsigned(mantissas, 1)) >>
                                (E8_BIAS - exponents - 1), mantissas)
        exponents = torch.maximum(exponents, torch.tensor(E8_BIAS - E2_BIAS, device=device)) - (E8_BIAS - E2_BIAS)
        e2m1_tmp = right_shift_unsigned(((exponents << 2) | right_shift_unsigned(mantissas, 21)) + 1, 1)
        e2m1_tmp = torch.minimum(e2m1_tmp, torch.tensor(0x7, device=device))
        e2m1_value = (right_shift_unsigned(signs, 28) | e2m1_tmp).to(torch.uint8)  # shape: (..., even_axis_shape)

        # Pack pairs of 4-bit values along the last dimension.
        e2m1_value = e2m1_value.view(*e2m1_value.shape[:-1], axis_shape // 2, 2)
        evens = e2m1_value[..., 0]
        odds = e2m1_value[..., 1]
        out_weight = evens | (odds << 4)  # shape: (..., axis_shape//2)

    # --- Process and output the scale ---
    dq_scale = (ds_int_rounded.view(*dequant_scale.shape) >> 23).to(torch.uint8)  # shape: (..., axis_shape//32, 1)
    dq_scale = dq_scale.squeeze(-1)
    out_weight = out_weight.transpose(axis, src_tensor.ndim - 1)
    dq_scale = dq_scale.transpose(axis, src_tensor.ndim - 1)
    return out_weight, dq_scale


def cvt_e2m1_to_fp32(input_tensor):
    assert input_tensor.dtype == torch.uint8

    input_tensor = input_tensor.to(torch.int32)
    evens = input_tensor & 0xF
    odds = (input_tensor >> 4) & 0xF

    vals = [0.0, 0.5, 1, 1.5, 2, 3, 4, 6]
    outputs = torch.tensor(vals, dtype=torch.float32, device=input_tensor.device)
    outputs = torch.cat([outputs, -outputs])

    even_floats = outputs[evens]
    odd_floats = outputs[odds]
    output_tensor = torch.stack([even_floats, odd_floats], dim=-1)
    output_tensor = output_tensor.view(*input_tensor.shape[:-1], -1)
    return output_tensor


def upcast_from_mxfp_torch(tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype, axis: int):
    """
    Converts the mxfp4/mxfp8 tensor to the target format specified by target_dtype.
      axis: The axis along which dequantization is applied.

    Returns:
      out_weight: Tensor in the target format.
    """

    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    is_fp8 = tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.float8_e5m2
    assert is_fp8 or tensor.dtype == torch.uint8, f"Invalid input quantization type {tensor.dtype}"

    # Permute the tensor and scale so that the quantization axis becomes the last dimension
    axis = axis if axis >= 0 else axis + ndim
    scale = scale.transpose(axis, scale.ndim - 1)
    tensor = tensor.transpose(axis, tensor.ndim - 1)

    dq_scale = (scale.to(torch.int32) << 23).view(torch.float32)  # Shift to the exponent and bitcast to fp32
    if tensor.dtype == torch.uint8:
        fp32_tensor = cvt_e2m1_to_fp32(tensor)
    else:
        fp32_tensor = tensor.to(torch.float32)

    logical_quant_dim = tensor.shape[-1] * (2 if tensor.dtype == torch.uint8 else 1)
    axis_shape = fp32_tensor.size(-1)
    padded_axis_shape = triton.cdiv(logical_quant_dim, MXFP_BLOCK_SIZE) * MXFP_BLOCK_SIZE
    pad_size = padded_axis_shape - axis_shape
    padded_tensor = F.pad(fp32_tensor, (0, pad_size))

    new_axis_shape = padded_tensor.shape[-1]
    new_shape = padded_tensor.shape[:-1] + (new_axis_shape // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE)
    padded_tensor = padded_tensor.view(*new_shape)
    dq_scale_padded = dq_scale.unsqueeze(-1)  # shape: [..., ceil(axis_shape/32), 1]
    out_padded = padded_tensor * dq_scale_padded

    # Flatten back and remove the padded tail
    out_padded = out_padded.view(*fp32_tensor.shape[:-1], new_axis_shape)
    out_tensor = out_padded[..., :axis_shape]

    out_tensor = out_tensor.to(target_dtype).contiguous()
    out_tensor = out_tensor.transpose(axis, tensor.ndim - 1)

    return out_tensor


quantize_mxfp8_fn = _quantize_mxfp8_fn
