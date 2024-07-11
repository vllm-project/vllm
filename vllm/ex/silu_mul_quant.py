import torch
from .code_cache import CodeCache

silu_mul_quant_name = "torch_P_ops_P__C_P_cutlass_scaled_mm_float16_int8_int8_float32_float32_None__operator_P_getitem_float16_T_Ellipsis_S_None_14336_None__operator_P_getitem_float16_T_Ellipsis_S_14336_None_None_torch_P_nn_P_functional_P_silu_float16__operator_P_mul_float16_float16_torch_P_empty_like_float16_fused"
def silu_mul_quant(output: torch.Tensor,
                input: torch.Tensor, 
                weight: torch.Tensor, 
                input_scale: torch.Tensor, 
                weight_scale: torch.Tensor,
                output_scale: torch.Tensor):
    torch.ops._C.cutlass_scaled_mm(output, input, weight, input_scale, weight_scale, None)

    # needs to be n//2 which could be a problem?
    silu_mul_output = torch.empty((output.size(0), output.size(1)//2))
    tmp = torch.empty_like(silu_mul_output)
    torch.ops._C.silu_and_mul_quant(output, silu_mul_output, output_scale, tmp)
    return silu_mul_output

def silu_mul_quant_meta(output: torch.Tensor,
                input: torch.Tensor, 
                weight: torch.Tensor, 
                input_scale: torch.Tensor, 
                weight_scale: torch.Tensor,
                output_scale: torch.Tensor):
    return torch.empty((output.size(0), output.size(1)//2))


def setup_silu_mul_quant(cc: CodeCache):
    namespace = "dogfood"
    ns_op = f"{namespace}::silu_mul_quant"
    sig = f"(Tensor gate_up_21, Tensor x_q_86, Tensor l__self___layers_21_mlp_gate_up_proj_weight, Tensor x_scale_86, Tensor l__self___layers_21_mlp_gate_up_proj_weight_scale, Tensor x_scale_87) -> Tensor"
    torch.library.define(f"{ns_op}", sig)
    torch.library.impl(f"{ns_op}", "CUDA", func=silu_mul_quant)
    torch.library.impl(f"{ns_op}", "Meta", func=silu_mul_quant_meta)
    cc.add(silu_mul_quant_name, torch.ops.dogfood.silu_mul_quant)