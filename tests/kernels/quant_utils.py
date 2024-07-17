import torch
from typing import Tuple

def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype) \
        -> Tuple[torch.tensor, torch.tensor]:

    assert quant_dtype in [torch.int8, torch.float8_e4m3fn]
    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
            else torch.finfo(quant_dtype)

    # For fp8, inorder to match the cuda kernel output, we have to do the same operations
    # to prevent rounding errors.

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = x_token_max.to(dtype=torch.float32)
    scales = x_token_max / torch.as_tensor([qtype_traits.max], dtype=torch.float32, device='cuda')
    scales = scales[:, None]

    # Quant
    iscales = torch.as_tensor([qtype_traits.max], dtype=torch.float32, device='cuda') / x_token_max
    iscales = iscales[:, None]
    torch_out = (x.to(dtype=torch.float32) * iscales).to(device="cuda", dtype=torch.float32)
    torch_out = torch_out.round() if quant_dtype == torch.int8 else torch_out
    torch_out = torch_out.clamp(qtype_traits.min, qtype_traits.max).to(quant_dtype)

    return torch_out, scales

# The int8 version is very similar. Incorporate the int8 version, like in
# ref_dynamic_per_token_quant, when we have a dynamic_per_tensor int8 quant
# kernel
def ref_dynamic_per_tensor_fp8_quant(x: torch.tensor) \
                    -> Tuple[torch.tensor, torch.tensor]:

    fp8_traits = torch.finfo(torch.float8_e4m3fn)
    fp8_max = torch.as_tensor([fp8_traits.max], dtype=torch.float32, device='cuda') 
    one = torch.as_tensor([1.0], dtype=torch.float32, device='cuda')

    # For fp8, inorder to match the cuda kernel output, we have to do the same operations
    # to prevent rounding errors.

    x_max = x.abs().max().to(dtype=torch.float32) 
    ref_scale = x_max / fp8_max 
    ref_iscale =  one / ref_scale 
    ref_out = (x.to(dtype=torch.float32) * ref_iscale).clamp(fp8_traits.min, fp8_traits.max).to(dtype=torch.float8_e4m3fn)
    return ref_out, ref_scale
