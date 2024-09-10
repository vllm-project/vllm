import torch

from .code_cache import CodeCache

# silu_mul_quant_name = "torch_P_ops_P__C_P_cutlass_scaled_mm_float16_float8_e4m3fn_float8_e4m3fn_float32_float32_None_torch_P_empty_T_int_8192_int_14336K_dtype_float16_K_device_D_cuda_0_torch_P_empty_T_int_8192_int_14336K_device_D_cuda_0_K_dtype_float8_e4m3fn_torch_P_ops_P__C_P_silu_and_mul_float16_float16_torch_P_ops_P__C_P_static_scaled_fp8_quant_float8_e4m3fn_float16_float32"  # noqa: E501
silu_mul_quant_name = "fused_silu_mul_quant"

register_silu_mul_quant = False


def silu_mul_quant(weight: torch.Tensor, weight_scale: torch.Tensor,
                   input_scale: torch.Tensor, input_: torch.Tensor,
                   output_scale: torch.Tensor):
    output = torch.empty((input_.size(0), weight.size(1)), 
                         dtype=torch.float16, 
                         device="cuda")
    torch.ops._C.cutlass_scaled_mm(output, input_, weight, input_scale,
                                   weight_scale, None)

    silu_mul_output = torch.empty((output.size(0), output.size(1) // 2),
                                  dtype=torch.float8_e4m3fn,
                                  device="cuda")
    tmp = torch.empty_like(silu_mul_output, dtype=torch.float32, device="cuda")
    torch.ops._C.silu_and_mul_quant(silu_mul_output, output, output_scale, tmp)
    return silu_mul_output


def silu_mul_quant_meta(weight: torch.Tensor, weight_scale: torch.Tensor,
                   input_scale: torch.Tensor, input_: torch.Tensor,
                   output_scale: torch.Tensor):
    full_output = torch.empty((input_.size(0), weight.size(1) // 2),
                              dtype=torch.float8_e4m3fn,
                              device="cuda")
    return full_output


def setup_silu_mul_quant(cc: CodeCache):
    # global register_silu_mul_quant
    # if not register_silu_mul_quant:
    #     return
    # register_silu_mul_quant = True
    namespace = "vllm"
    ns_op = f"{namespace}::silu_mul_quant"
    sig = ("(Tensor weight, "
           "Tensor weight_scale, "
           "Tensor x_scale_126, "
           "Tensor output,"
           "Tensor x_scale_127) "
           "-> Tensor")
    torch.library.define(f"{ns_op}", sig)
    torch.library.impl(f"{ns_op}", "CUDA", func=silu_mul_quant)
    torch.library.impl(f"{ns_op}", "Meta", func=silu_mul_quant_meta)
    cc.add(silu_mul_quant_name, torch.ops.vllm.silu_mul_quant)
