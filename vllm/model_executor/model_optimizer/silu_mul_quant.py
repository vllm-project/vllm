import torch
from .code_cache import CodeCache

silu_mul_quant_name = "torch_P_empty_T_int_3_int_1K_device_D_cuda_0_K_dtype_float32_torch_P_ops_P__C_P_cutlass_scaled_mm_float16_int8_int8_float32_float32_None__operator_P_getitem_float16_T_Ellipsis_S_None_14336_None__operator_P_getitem_float16_T_Ellipsis_S_14336_None_None_torch_P_nn_P_functional_P_silu_float16__operator_P_mul_float16_float16_torch_P_empty_like_float16K_dtype_int8_torch_P_ops_P__C_P_dynamic_scaled_int8_quant_int8_float16_float32_fused"


def silu_mul_quant(output: torch.Tensor, input: torch.Tensor,
                   weight: torch.Tensor, input_scale: torch.Tensor,
                   weight_scale: torch.Tensor):
    torch.ops._C.cutlass_scaled_mm(output, input, weight, input_scale,
                                   weight_scale, None)

    silu_mul_output = torch.empty((output.size(0), output.size(1) // 2),
                                  dtype=torch.int8,
                                  device="cuda")
    tmp = torch.empty_like(silu_mul_output, dtype=torch.float32)
    output_scale = torch.empty((output.size(0), 1),
                               dtype=torch.float32,
                               device="cuda")
    torch.ops._C.silu_and_mul_quant(silu_mul_output, output, output_scale, tmp)
    return (output_scale, silu_mul_output)


def silu_mul_quant_meta(output: torch.Tensor, input: torch.Tensor,
                        weight: torch.Tensor, input_scale: torch.Tensor,
                        weight_scale: torch.Tensor):
    full_output = torch.empty((output.size(0), output.size(1) // 2),
                              dtype=torch.int8,
                              device="cuda")
    output_scale = torch.empty((output.size(0), 1),
                               dtype=torch.float32,
                               device="cuda")
    return (output_scale, full_output)


def setup_silu_mul_quant(cc: CodeCache):
    namespace = "dogfood"
    ns_op = f"{namespace}::silu_mul_quant"
    sig = "(Tensor gate_up_31, Tensor x_q_126, Tensor l__self___layers_31_mlp_gate_up_proj_weight, Tensor x_scale_126, Tensor l__self___layers_31_mlp_gate_up_proj_weight_scale) -> (Tensor, Tensor)"
    torch.library.define(f"{ns_op}", sig)
    torch.library.impl(f"{ns_op}", "CUDA", func=silu_mul_quant)
    torch.library.impl(f"{ns_op}", "Meta", func=silu_mul_quant_meta)
    cc.add(silu_mul_quant_name, torch.ops.dogfood.silu_mul_quant)
