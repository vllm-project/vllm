import torch

from .code_cache import CodeCache

# ruff: noqa: E501
rms_norm_quant_name = "fused_rms_norm_quant"
rms_norm_quant_name_2 = "fused_rms_norm_quant_2"

register_fused_rms_norm = False

# def add_residual_rms_norm_quant(input: torch.Tensor,
#                                 weight: torch.Tensor,
#                                 scale: torch.Tensor) -> torch.Tensor:

#     tmp = torch.empty_like(input, dtype=torch.float32)
#     out = torch.empty_like(input, dtype=torch.int8)
#     torch.ops._C.add_residual_rms_norm_quant(out, input, residual, tmp, weight,
#                                              scale, 1e-05)
#     return out

# def add_residual_rms_norm_quant_meta(input: torch.Tensor,
#                                      residual: torch.Tensor,
#                                      weight: torch.Tensor,
#                                      scale: torch.Tensor) -> torch.Tensor:
#     out = torch.empty_like(input, dtype=torch.int8)
#     return out


def rms_norm_quant(input: torch.Tensor, weight: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    hidden_size = input.size(-1)
    num_tokens = input.numel() // hidden_size

    tmp = torch.empty((num_tokens, hidden_size),
                      dtype=torch.float32,
                      device="cuda")
    out = torch.empty((num_tokens, hidden_size),
                      dtype=torch.float8_e4m3fn,
                      device="cuda")
    torch.ops._C.rms_norm_quant(out, input, tmp, weight, scale, 1e-05)
    return out


def rms_norm_quant_meta(input: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor) -> torch.Tensor:
    hidden_size = input.size(-1)
    num_tokens = input.numel() / hidden_size

    return torch.empty((num_tokens, hidden_size),
                       dtype=torch.float8_e4m3fn,
                       device="cuda")


def rms_norm_quant_2(input: torch.Tensor, weight: torch.Tensor,
                     scale: torch.Tensor) -> torch.Tensor:
    hidden_size = input.size(-1)
    num_tokens = input.numel() // hidden_size

    tmp = torch.empty((num_tokens, hidden_size),
                      dtype=torch.float32,
                      device="cuda")
    out = torch.empty((num_tokens, hidden_size),
                      dtype=torch.float8_e4m3fn,
                      device="cuda")
    torch.ops._C.rms_norm_quant(out, input, tmp, weight, scale, 1e-05)
    second_output = torch.empty(out.size(0),
                                6144,
                                dtype=torch.float16,
                                device="cuda")
    return (out, second_output)


def rms_norm_quant_meta_2(input: torch.Tensor, weight: torch.Tensor,
                          scale: torch.Tensor) -> torch.Tensor:
    hidden_size = input.size(-1)
    num_tokens = input.numel() / hidden_size
    out = torch.empty((num_tokens, hidden_size),
                      dtype=torch.float8_e4m3fn,
                      device="cuda")
    second_output = torch.empty(out.size(0),
                                6144,
                                dtype=torch.float16,
                                device="cuda")

    return (out, second_output)


def setup_fused_rms_norm(cc: CodeCache):
    # global register_fused_rms_norm
    # if not register_fused_rms_norm:
    #     return
    # register_fused_rms_norm = True
    namespace = "dogfood"
    ns_op = f"{namespace}::rms_norm_quant"
    sig = "(Tensor output, Tensor detach, Tensor x_scale) -> Tensor"
    torch.library.define(f"{ns_op}", sig)
    torch.library.impl(f"{ns_op}", "CUDA", func=rms_norm_quant)
    torch.library.impl(f"{ns_op}", "Meta", func=rms_norm_quant_meta)
    cc.add(rms_norm_quant_name, torch.ops.dogfood.rms_norm_quant)


def setup_fused_rms_norm_2(cc: CodeCache):
    # global register_fused_rms_norm
    # if not register_fused_rms_norm:
    #     return
    # register_fused_rms_norm = True
    namespace = "dogfood"
    ns_op = f"{namespace}::rms_norm_quant_2"
    sig = "(Tensor output, Tensor detach, Tensor x_scale) -> (Tensor, Tensor)"
    torch.library.define(f"{ns_op}", sig)
    torch.library.impl(f"{ns_op}", "CUDA", func=rms_norm_quant_2)
    torch.library.impl(f"{ns_op}", "Meta", func=rms_norm_quant_meta_2)
    cc.add(rms_norm_quant_name_2, torch.ops.dogfood.rms_norm_quant_2)
