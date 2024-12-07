from typing import Callable, Optional, Type, Tuple

import torch

import vllm._custom_ops as ops


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def rand_int8(shape: tuple, device: str = "cuda"):
    return to_int8(torch.rand(shape, device=device) * 255 - 128)


def prune_to_2_4(tensor):
    # Reshape tensor to [N, 4] where N is number of groups of 4
    original_shape = tensor.shape
    reshaped = tensor.reshape(-1, 4)
    
    # Get indices of top 2 absolute values in each group of 4
    _, indices = torch.topk(torch.abs(reshaped), k=2, dim=1)
    
    # Create binary mask
    mask = torch.zeros_like(reshaped)
    mask.scatter_(dim=1, index=indices, src=torch.ones_like(indices, dtype=mask.dtype))
    
    # Apply mask and reshape back
    pruned = reshaped * mask

    # Turn all -0.0 to 0.0
    pruned[pruned == -0.0] = 0.0

    return pruned.reshape(original_shape)


def make_rand_tensors(dtype: torch.dtype, m: int, n: int,
                             k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    # # Initialize a to all ones
    # a = torch.ones((m, k), device='cuda')
    # # Initialize b to all ones
    # b = torch.ones((n, k), device='cuda')

    b = prune_to_2_4(b.t()).t()

    if dtype == torch.int8:
        a, b = to_int8(a), to_int8(b)
    elif dtype == torch.float8_e4m3fn:
        a, b = to_fp8(a), to_fp8(b)
    else:
        raise ValueError("unsupported dtype")

    b_compressed, e = ops.cutlass_compress_entry(b.t())

    # Compressed B, Metadata, Original A, B
    return b_compressed, e, a, b


def baseline_scaled_mm(a: torch.Tensor,
                       b: torch.Tensor,
                       scale_a: torch.Tensor,
                       scale_b: torch.Tensor,
                       out_dtype: Type[torch.dtype],
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    output = (scale_a * (scale_b * (torch.mm(
        a.to(dtype=torch.float32), b.to(dtype=torch.float32))))).to(out_dtype)
    if bias is not None:
        output = output + bias

    return output


def autogen_scaled_mm_fp8_gemm_test(
        fn: Callable,
        m: int,
        n: int,
        k: int,
        per_token_act_quant: bool,
        per_out_channel_weight_quant: bool,
        out_dtype: Type[torch.dtype] = torch.bfloat16,
        device: str = "cuda"):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = torch.randn((m, k), device=device)
    b = torch.randn((n, k), device=device).t()

    b = prune_to_2_4(b.t()).t()

    a, b = to_fp8(a), to_fp8(b)

    b_compressed, e = ops.cutlass_compress_entry(b.t())

    m_a_scales = m if per_token_act_quant else 1
    n_b_scales = n if per_out_channel_weight_quant else 1

    scale_a = (torch.randn((m_a_scales, 1), device=device,
                           dtype=torch.float32))
    scale_b = (torch.randn((1, n_b_scales), device=device,
                           dtype=torch.float32))

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    fn(out, b_compressed, e, a.t(), scale_a, scale_b)
    # TODO (varun) : cache baseline scaled_mm results so we dont recompute.
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype)
    return torch.allclose(out, baseline, rtol=1e-2, atol=5e-2)
