import torch

from .code_cache import CodeCache

# ruff: noqa: E501
add_residual_rms_norm_quant_name = "torch_P_ops_P__C_P_fused_add_rms_norm_float16_float16_float16_float_1e_05_torch_P_empty_like_float16_torch_P_ops_P__C_P_static_scaled_int8_quant_int8_float16_float32_fused"


def add_residual_rms_norm_quant(input: torch.Tensor, residual: torch.Tensor,
                                weight: torch.Tensor,
                                scale: torch.Tensor) -> torch.Tensor:

    tmp = torch.empty_like(input, dtype=torch.float32)
    out = torch.empty_like(input, dtype=torch.int8)
    torch.ops._C.add_residual_rms_norm_quant(out, input, residual, tmp, weight,
                                             scale, 1e-05)
    return out


def add_residual_rms_norm_quant_meta(input: torch.Tensor,
                                     residual: torch.Tensor,
                                     weight: torch.Tensor,
                                     scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(input, dtype=torch.int8)
    return out


def setup_fused_rms_norm(cc: CodeCache):
    namespace = "dogfood"
    ns_op = f"{namespace}::add_residual_rms_norm_quant"
    sig = "(Tensor hidden_states_158, Tensor! _, Tensor detach_63, Tensor x_scale_126) -> Tensor"
    torch.library.define(f"{ns_op}", sig)
    torch.library.impl(f"{ns_op}", "CUDA", func=add_residual_rms_norm_quant)
    torch.library.impl(f"{ns_op}",
                       "Meta",
                       func=add_residual_rms_norm_quant_meta)
    cc.add(add_residual_rms_norm_quant_name,
           torch.ops.dogfood.add_residual_rms_norm_quant)
