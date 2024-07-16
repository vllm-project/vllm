import torch
from typing import Tuple

def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype) \
        -> Tuple[torch.tensor, torch.tensor]:

    assert quant_dtype in [torch.int8, torch.float8_e4m3fn]
    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
            else torch.finfo(quant_dtype)

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = x_token_max.to(dtype=torch.float32)
    scales = (x_token_max / float(qtype_traits.max))[:, None].to(device="cuda",
                                                      dtype=torch.float32)
    # Quant
    torch_out = (x / scales).round().clamp(qtype_traits.min, qtype_traits.max).to(quant_dtype)

    return torch_out, scales
