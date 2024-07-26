from typing import Optional, Tuple, Union

import torch


def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')

def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype,
                                scale_ub: Optional[torch.tensor] = None) \
        -> Tuple[torch.tensor, torch.tensor]:

    assert quant_dtype in [torch.int8, torch.float8_e4m3fn]
    if scale_ub is not None:
        assert quant_dtype == torch.float8_e4m3fn

    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
            else torch.finfo(quant_dtype)
    qtype_max = as_float32_tensor(qtype_traits.max)
    s_1 = as_float32_tensor(1.0)
    s_512 = as_float32_tensor(512.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    if scale_ub is not None:
        x_token_max = x_token_max.clamp(max=scale_ub)
    scales = (x_token_max / qtype_max)[:, None]

    # Quant
    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        torch_out = as_float32_tensor(x) * iscales
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits.min,
                                    qtype_traits.max).to(quant_dtype)
    else:
        assert quant_dtype == torch.float8_e4m3fn
        min_scaling_factor = s_1 / (qtype_max * s_512)
        scales = scales.clamp(min=min_scaling_factor)
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.clamp(qtype_traits.min,
                                    qtype_traits.max).to(quant_dtype)

    return torch_out, scales


# The int8 version is very similar. Incorporate the int8 version, like in
# ref_dynamic_per_token_quant, when we have a dynamic_per_tensor int8 quant
# kernel
def ref_dynamic_per_tensor_fp8_quant(x: torch.tensor) \
                    -> Tuple[torch.tensor, torch.tensor]:

    fp8_traits = torch.finfo(torch.float8_e4m3fn)
    fp8_max = as_float32_tensor(fp8_traits.max)
    one = as_float32_tensor(1.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    x_max = as_float32_tensor(x.abs().max())
    ref_scale = x_max / fp8_max
    ref_iscale = one / ref_scale
    ref_out = (as_float32_tensor(x) * ref_iscale).clamp(
        fp8_traits.min, fp8_traits.max).to(dtype=torch.float8_e4m3fn)
    return ref_out, ref_scale
