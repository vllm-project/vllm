import torch
import torch.nn.functional as F
from safetensors import safe_open

from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.numerics import InFlexData
import triton_kernels.swiglu
from triton_kernels.matmul_ogs import (FnSpecs, FusedActivation,
                                       PrecisionConfig, matmul_ogs)
from triton_kernels.routing import (routing)
from triton_kernels.tensor_details import layout


zip_folder = "/data/xmo/yongye/vllm-os-mini/tests/kernels/moe/nan_debug"

st_files=f"{zip_folder}/model.pt"
act_files=f"{zip_folder}/norm_out.pt"
routing_logits_file=f"{zip_folder}/routing_out.pt"
device="cuda"

block = 2
# dim after padding
w1_in_dim_pad = 2880
w1_out_dim_pad = 5760
w2_in_dim_pad = 5760 // 2
w2_out_dim_pad = 2880

def swizzle_mxfp4(quant_tensor, scale):
    num_warps = 8 # if max_batched_tokens <= 512 else 8
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps)
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(wrap_torch_tensor(quant_tensor, dtype=FP4),
                                  value_layout, **value_layout_opts)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    import pdb; pdb.set_trace()
    return quant_tensor, InFlexData(), scale

def moe_forward(
    hidden_states,
    w1,
    w2,
    gating_output,
    topk,
    w1_bias,
    w2_bias,
    w1_precision,
    w2_precision
):
    routing_data, gather_idx, scatter_idx = None, None, None
    
    act = FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")),
        (1.702, None), 2)

    intermediate_cache1 = matmul_ogs(hidden_states,
                                     w1,
                                     w1_bias,
                                     routing_data,
                                     gather_indx=gather_idx,
                                     precision_config=w1_precision,
                                     gammas=None,
                                     fused_activation=act)

    intermediate_cache3 = matmul_ogs(
        intermediate_cache1,
        w2,
        w2_bias,
        routing_data,
        scatter_indx=scatter_idx,
        precision_config=w2_precision,
        gammas=routing_data.gate_scal)

    return intermediate_cache3

def test_close():
    # routing_logits = torch.load(routing_logits_file, map_location=device)
    # x_in = torch.load(act_files, map_location=device)
    # routing_logits = routing_logits.squeeze()
    
    # param_dict = torch.load(st_files, map_location=device)

    # w1_weight = param_dict["w1_weight"]     # [128, 5760, 1440]
    # w1_scale = param_dict["w1_scale"]       # [128, 5760, 90]
    # w1_bias = param_dict["w1_bias"].float() # [128, 5760]
    # w2_weight = param_dict["w2_weight"]     # [128, 2880, 1440]
    # w2_scale = param_dict["w2_scale"]       # [128, 2880, 90]
    # w2_bias = param_dict["w2_bias"].float() # [128, 2880]

    x_in = torch.rand((128, 64, 2880), dtype=torch.bfloat16, device=device)
    routing_logits = FileNotFoundError
    import pdb; pdb.set_trace()
    w1_weight = torch.rand((128, 5760, 1440), dtype=torch.bfloat16, device=device)
    w1_scale = torch.rand((128, 5760, 90), dtype=torch.bfloat16, device=device)
    w1_bias = torch.rand((128, 5760), dtype=torch.float32, device=device)
    w2_weight = torch.rand((128, 2880, 1440), dtype=torch.bfloat16, device=device)
    w2_scale = torch.rand((128, 2880, 90), dtype=torch.bfloat16, device=device)
    w2_bias = torch.rand((128, 2880), dtype=torch.float32, device=device)

    # padding weight tp enable hbm_swizzle
    w1_weight_pad = torch.zeros(
        128, w1_out_dim_pad, w1_in_dim_pad // 2,
        dtype=torch.uint8, device=device
    )
    w1_weight_pad[:, :w1_weight.shape[1], :w1_weight.shape[2]].copy_(w1_weight)

    w1_scale_pad = torch.zeros(
        128, w1_out_dim_pad, w1_in_dim_pad // 32,
        dtype=torch.uint8, device=device
    )
    w1_scale_pad[:, :w1_scale.shape[1], :w1_scale.shape[2]].copy_(w1_scale)

    w1_bias_pad = torch.zeros(128, w1_out_dim_pad, dtype=torch.float32, device=device)
    w1_bias_pad[:, :w1_bias.shape[1]].copy_(w1_bias)

    w2_weight_pad = torch.zeros(
        128, w2_out_dim_pad, w2_in_dim_pad // 2,
        dtype=torch.uint8, device=device
    )
    w2_weight_pad[:, :w2_weight.shape[1], :w2_weight.shape[2]].copy_(w2_weight)
    w2_scale_pad = torch.zeros(
        128, w2_out_dim_pad, w2_in_dim_pad // 32,
        dtype=torch.uint8, device=device
    )
    w2_scale_pad[:, :w2_scale.shape[1], :w2_scale.shape[2]].copy_(w2_scale)

    w2_bias_pad = torch.zeros(128, w2_out_dim_pad, dtype=torch.float32, device=device)
    w2_bias_pad[:, :w2_bias.shape[1]].copy_(w2_bias)

    w1_weight_pad, w1_flex_pad, w1_scale_pad = swizzle_mxfp4(
            w1_weight_pad, w1_scale_pad)
    w2_weight_pad, w2_flex_pad, w2_scale_pad = swizzle_mxfp4(w2_weight_pad,
                                                    w2_scale_pad)

    pc1 = PrecisionConfig(
            weight_scale=w1_scale_pad, flex_ctx=FlexCtx(rhs_data=w1_flex_pad))
    pc2 = PrecisionConfig(
            weight_scale=w2_scale_pad, flex_ctx=FlexCtx(rhs_data=w2_flex_pad))
    
    assert not torch.isnan(x_in).any(), "NaN detected in input"
    out_tri = moe_forward(
        hidden_states=x_in,
        w1=w1_weight_pad,
        w2=w2_weight_pad,
        gating_output=routing_logits,
        topk=4,
        w1_bias=w1_bias_pad,
        w2_bias=w2_bias_pad,
        w1_precision=pc1,
        w2_precision=pc2) 
    assert not torch.isnan(out_tri).any(), "NaN detected in output"
    
if __name__ == "__main__":
    test_close()

    


