import torch
import pytest
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_fp4_moe
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1fn.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

def swizzle_blockscale(scale: torch.tensor):
    # Pad and blockwise interleave weight_scale
    assert scale.ndim == 3
    E, M, K = scale.shape
    round_up_multiple = lambda x, m: (x + m - 1) // m * m
    M_padded = round_up_multiple(M, 128)
    K_padded = round_up_multiple(K, 4)
    padded_scale = torch.zeros((E, M_padded, K_padded), dtype=scale.dtype)
    padded_scale[:E, :M, :K] = scale
    experts, rows, cols = padded_scale.shape
    assert rows % 128 == 0
    assert cols % 4 == 0
    padded_scale = padded_scale.reshape(experts, rows // 128, 4, 32,
                                        cols // 4, 4)
    swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
    swizzled_scale = swizzled_scale.contiguous().cuda()
    return swizzled_scale.reshape(E, M, K)


kE2M1ToFloat = torch.tensor([
    0., 0.5, 1., 1.5, 
    2., 3., 4., 6.
], dtype=torch.float32)

def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape
    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F          # Lower nibbles
    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()
    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)
    # Device-aware lookup and sign application
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_to_dtype(tensor_fp4,
                        tensor_sf,
                        global_scale,
                        dtype,
                        device,
                        block_size=16):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype)


@pytest.mark.parametrize("m", [2, 16, 32, 64, 224])
@pytest.mark.parametrize("n", [2048])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("e", [32])
@pytest.mark.parametrize("topk", [4,6])
@torch.inference_mode()
def test_cutlass_fp4_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
):
    current_platform.seed_everything(7)
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):

        dtype = torch.half

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        quant_blocksize = 16
        w1_blockscale = torch.empty((e, 2 * n, k // quant_blocksize), 
                                    device="cuda",
                                    dtype=torch.float8_e4m3fn)
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        w2_blockscale = torch.empty((e, k, n // quant_blocksize), 
                                    device="cuda",
                                    dtype=torch.float8_e4m3fn)

        
        # n_b_scales = 2 * n if per_out_ch else 1
        # k_b_scales = k if per_out_ch else 1

        w1_q = torch.empty((e, 2 * n, k // 2),
                           device="cuda",
                           dtype=torch.uint8)
        w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
        
       
        w1_gs = torch.empty((e,),
                               device="cuda",
                               dtype=torch.float32)
        w2_gs = torch.empty((e,),
                               device="cuda",
                               dtype=torch.float32)

        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

        for expert in range(e):
            w1_amax = torch.abs(w1).max().to(torch.float32)
            w2_amax = torch.abs(w2).max().to(torch.float32)
            w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax 
            w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax 
            
            w1_q[expert], w1_blockscale[expert]= ops.scaled_fp4_quant(
                w1[expert], w1_gs[expert])
            

            w2_q[expert], w2_blockscale[expert]= ops.scaled_fp4_quant(
                w2[expert], w2_gs[expert])
            
        if False: 
          # not required here because weights are already swizzled 
          # from the quantization above.   
          w1_bs_swizzled = swizzle_blockscale(w1_blockscale)
          w2_bs_swizzled = swizzle_blockscale(w2_blockscale)
        
      
        ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
        ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
        c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
        
        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)
        
        
        cutlass_output = cutlass_fp4_moe(a=a, 
                                         w1_fp4=w1_q, 
                                         w1_blockscale=w1_blockscale,
                                         w1_tensorscale=w1_gs,
                                         w2_fp4=w2_q, 
                                         w2_blockscale=w2_blockscale,
                                         w2_tensorscale=w2_gs,
                                         topk_weights=topk_weights,
                                         topk_ids=topk_ids,
                                         m=m,n=n, k=k, e=e,
                                         gemm1_AB_strides=ab_strides1,
                                         gemm1_C_strides=c_strides1,
                                         gemm2_AB_strides=ab_strides2,
                                         gemm2_C_strides=c_strides2,
                                     )
        print(cutlass_output)
        
        # Reference check: 
        a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                          torch.amax(a.flatten(), dim=-1)).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)   
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_to_dtype(a_fp4,
                                         a_scale_interleaved,
                                         a_global_scale,
                                         dtype=a.dtype,
                                         device=a.device,
                                         block_size=quant_blocksize)

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0,e):
            w1_d[idx] = dequantize_to_dtype(w1_q[idx],
                                         w1_blockscale[idx],
                                         w1_gs[idx],
                                         dtype=w1.dtype,
                                         device=w1.device,
                                         block_size=quant_blocksize)
            w2_d[idx] = dequantize_to_dtype(w2_q[idx],
                                         w2_blockscale[idx],
                                         w2_gs[idx],
                                         dtype=w2.dtype,
                                         device=w2.device,
                                         block_size=quant_blocksize)

             
        torch_output = torch_moe(a_in_dtype, w1_d, w2_d, score, topk, None)
        print(torch_output)
        
        torch.testing.assert_close(torch_output,
                                   cutlass_output,
                                    atol=3.6e-2,
                                    rtol=1e-2)