# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional
import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (FLOAT4_E2M1_MAX,
                                                    FLOAT8_E4M3_MAX,
                                                    dequantize_nvfp4_to_dtype)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4
from vllm.model_executor.layers.fused_moe.fused_moe import (fused_experts,
                                                            fused_topk)
from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(reason="Nvfp4 Requires compute capability of 10 or above.",
                allow_module_level=True)

DEBUG = True

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 1024, 1536),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1024),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]

def run_ref_fp4_moe(a, w1_q, w2_q,
                        w1_blockscale, w2_blockscale,
                        w1_gs, w2_gs,
                        e, n, k,
                        quant_blocksize, dtype,
                        score, topk,
                        use_triton_ref=False,
                        topk_weights: Optional[torch.Tensor]=None,
                        topk_ids: Optional[torch.Tensor]=None):  # Only for testing purposes
    # a1_globalscale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
    #                    torch.abs(a.flatten()).max(dim=-1).values).to(torch.float32)
    a1_globalscale = torch.tensor(1.0, device=a.device, dtype=torch.float32)
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a1_globalscale)
    _, m_k = a_fp4.shape
    
    # This calculation is not correct, but it is only for testing purposes.
    # Here we are using scaling_factor=1.0 for all experts. If each expert has
    # a different scaling factor, we need to use a different global scale for 
    # each expert and dequantize individually.
    
    a_in_dtype = dequantize_nvfp4_to_dtype(a_fp4,
                                           a_scale_interleaved,
                                           a1_globalscale,
                                           dtype=a.dtype,
                                           device=a.device,
                                           block_size=quant_blocksize)

    w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
    w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

    for idx in range(0, e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(w1_q[idx],
                                              w1_blockscale[idx],
                                              w1_gs[idx],
                                              dtype=dtype,
                                              device=w1_q.device,
                                              block_size=quant_blocksize)
        w2_d[idx] = dequantize_nvfp4_to_dtype(w2_q[idx],
                                              w2_blockscale[idx],
                                              w2_gs[idx],
                                              dtype=dtype,
                                              device=w2_q.device,
                                              block_size=quant_blocksize)
    if use_triton_ref:
       return fused_experts(a_in_dtype, w1_d, w2_d, topk_weights, topk_ids) 
    else:
       return torch_moe(a_in_dtype, w1_d, w2_d, score, topk, None)

def convert_inputs_to_fp4(w1, w2, e, n, k, quant_blocksize, dtype):
    round_up = lambda x, y: (x + y - 1) // y * y
    sf_w1_2n = round_up(2 * n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty((e, sf_w1_2n, sf_w1_k),
                              device="cuda",
                              dtype=torch.float8_e4m3fn)

    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty((e, sf_w2_k, sf_w2_n),
                              device="cuda",
                              dtype=torch.float8_e4m3fn)

    w1_q = torch.empty((e, 2 * n, k // 2),
                     device="cuda",
                     dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e, ), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e, ), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1).max().to(torch.float32)
        w2_amax = torch.abs(w2).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_q[expert], w1_blockscale[expert] = ops.scaled_fp4_quant(
            w1[expert], w1_gs[expert])

        w2_q[expert], w2_blockscale[expert] = ops.scaled_fp4_quant(
            w2[expert], w2_gs[expert])

    return w1_q, w2_q, w1_blockscale, w2_blockscale, w1_gs, w2_gs

@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_cutlass_fp4_moe_no_graph(m: int, n: int, k: int, e: int, topk: int,
                                  dtype: torch.dtype):
    current_platform.seed_everything(7)
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        quant_blocksize = 16
        
        w1_q, w2_q, w1_blockscale, w2_blockscale, w1_gs, w2_gs = (
            convert_inputs_to_fp4(w1, w2, e, n, k, quant_blocksize, dtype))

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        a1_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)
        assert e > topk, "Number of experts must be greater than topk"
        topk_weights, topk_ids, _ = fused_topk(a, score, topk,
                                               renormalize=False)
        # strides for the cutlass moe_fp4 kernel
        ab_strides_13 = torch.full((e,),
                                  w1_q.shape[2] * 2,
                                  dtype=torch.int32, 
                                  device=w1_q.device)
        c_strides_13 = torch.full((e,),
                                 w1_q.shape[1],
                                 dtype=torch.int32,
                                 device=w1_q.device)
        ab_strides_2 = torch.full((e,),
                                 w2_q.shape[2] * 2,
                                 dtype=torch.int32,
                                 device=w2_q.device)
        c_strides_2 = torch.full((e,),
                                 w2_q.shape[1],
                                 dtype=torch.int32,
                                 device=w2_q.device)
        
        cutlass_output = cutlass_moe_fp4(
            a=a,
            a1_gscale=a1_gs,
            w1_fp4=w1_q,
            w1_blockscale=w1_blockscale,
            w1_alphas=(1 / w1_gs),
            a2_gscale=a2_gs,
            w2_fp4=w2_q,
            w2_blockscale=w2_blockscale,
            w2_alphas=(1 / w2_gs),
            ab_strides_13=ab_strides_13,
            ab_strides_2=ab_strides_2,
            c_strides_13=c_strides_13,
            c_strides_2=c_strides_2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            m=m,
            n=n,
            k=k,
            e=e,
            device=a.device,
        )
        # Reference check:
        torch_output = run_ref_fp4_moe(a, w1_q, w2_q,
                                        w1_blockscale, w2_blockscale,
                                        w1_gs, w2_gs, e, n, k,
                                        quant_blocksize, dtype,
                                        score, topk)
        torch.testing.assert_close(torch_output,
                                   cutlass_output,
                                   atol=1e-1,
                                   rtol=1e-1)

def run_with_expert_maps(num_experts: int, num_local_experts: int,
                         quant_blocksize: int, dtype: torch.dtype,
                         score: torch.Tensor, topk: int,
                        **cutlass_fp4_moe_kwargs):

   def slice_experts():
       slice_params = [
            "a1_gscale", "a2_gscale", 
            "w1_fp4", "w2_fp4", "w1_blockscale",
            "w2_blockscale", "w1_alphas", "w2_alphas"
       ]
       full_tensors = {
           k: v
           for k, v in cutlass_fp4_moe_kwargs.items()
           if k in slice_params and k in cutlass_fp4_moe_kwargs
       }

       for i in range(0, num_experts, num_local_experts):
           l, u = i, i + num_local_experts
           # make expert map
           expert_map = [-1] * num_experts
           expert_map[l:u] = list(range(num_local_experts))
           expert_map = torch.tensor(expert_map,
                                     dtype=torch.int32,
                                     device="cuda")

           # update cutlass moe arg with expert_map
           cutlass_fp4_moe_kwargs["expert_map"] = expert_map
           # update cutlass moe arg tensors
           for k, t in full_tensors.items():
              assert t.shape[0] == num_experts, "Tensor shape mismatch"
              cutlass_fp4_moe_kwargs[k] = t[l:u]
           cutlass_fp4_moe_kwargs["m"] = cutlass_fp4_moe_kwargs["a"].shape[0]
           cutlass_fp4_moe_kwargs["n"] = cutlass_fp4_moe_kwargs["w2_fp4"].shape[2] * 2
           cutlass_fp4_moe_kwargs["k"] = cutlass_fp4_moe_kwargs["a"].shape[1]
           cutlass_fp4_moe_kwargs["e"] = cutlass_fp4_moe_kwargs["w1_fp4"].shape[0]
           cutlass_fp4_moe_kwargs["apply_router_weight_on_input"] = False
           yield cutlass_fp4_moe_kwargs

   out_tensor = torch.zeros_like(cutlass_fp4_moe_kwargs["a"])
   for kwargs in slice_experts():
        cutlass_tensor = cutlass_moe_fp4(**kwargs)
        out_tensor = out_tensor + cutlass_tensor
   return out_tensor


def find_mismatches(actual, expected, atol, rtol):
    # Calculate divergence masks
    abs_diff = torch.abs(actual - expected)
    rel_diff = abs_diff / (torch.abs(expected) + 1e-8)  # Avoid division by zero
    
    # Find positions violating either tolerance
    mismatch_mask = (abs_diff > atol) | (rel_diff > rtol)
    mismatch_indices = torch.nonzero(mismatch_mask).tolist()
    
    return mismatch_indices, abs_diff, rel_diff


@pytest.mark.parametrize("m", [64])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("e", [16])
@pytest.mark.parametrize("topk", [1, 8])
@pytest.mark.parametrize("local_expert_size", [1, 2, 4, 8, 16])
def test_cutlass_moe_EP(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    local_expert_size: int,
):
    # current_platform.seed_everything(7)
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):
        dtype = torch.half
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        quant_blocksize = 16
        
        w1_fp4, w2_fp4, w1_blockscale, w2_blockscale, w1_gs, w2_gs = (
            convert_inputs_to_fp4(w1, w2, e, n, k, quant_blocksize, dtype))

        score = torch.randn((m, e), device="cuda", dtype=dtype)

        a1_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)

        assert e % local_expert_size == 0, "Cannot distribute experts evenly"

        topk_weights, topk_ids, _ = fused_topk(a, score, topk,
                                               renormalize=False)

        out_tensor = run_with_expert_maps(e, local_expert_size,
                                         quant_blocksize=quant_blocksize,
                                         dtype=dtype,
                                         score=score,
                                         topk=topk,
                                         a=a,
                                         a1_gscale=a1_gs,
                                         a2_gscale=a2_gs,
                                         w1_fp4=w1_fp4,
                                         w2_fp4=w2_fp4,
                                         w1_blockscale=w1_blockscale,
                                         w2_blockscale=w2_blockscale,
                                         w1_alphas=1/w1_gs,
                                         w2_alphas=1/w2_gs,
                                         topk_weights=topk_weights,
                                         topk_ids=topk_ids,
                                         device=a.device,
                                        
                                         )
        # Reference check:
        ref_output = run_ref_fp4_moe(a, w1_fp4, w2_fp4,
                                        w1_blockscale, w2_blockscale,
                                        w1_gs, w2_gs, e, n, k,
                                        quant_blocksize, dtype,
                                        score, topk,
                                        use_triton_ref=True,
                                        topk_weights=topk_weights,
                                        topk_ids=topk_ids)
        mismatch_indices, abs_diffs, rel_diffs = find_mismatches(out_tensor,
                                                                 ref_output,
                                                                 atol=1e-1,
                                                                 rtol=1e-1)
        with open('scratch/diff.txt', 'w') as f:
           # If abs_diffs is a 2D tensor (matrix)
            formatted_values = []
            for row in abs_diffs.tolist():
                if isinstance(row, list):
                    # Handle nested lists (2D case)
                    formatted_row = [f"{x:.2f}" for x in row]
                    formatted_values.append(' '.join(formatted_row))
                else:
                    # Handle single values (1D case)
                    formatted_values.append(f"{row:.2f}")
            f.write('\n'.join(formatted_values))
        
        print(f"Total mismatches: {len(mismatch_indices)}")
        print("mismatch indices:", mismatch_indices)
        torch.testing.assert_close(ref_output,
                                   out_tensor,
                                   atol=1e-1,
                                   rtol=1e-1)


if __name__ == "__main__":
    m, n, k = (2,3072,1024)
    e = 8
    topk = 1
    local_expert_size = 4
    test_cutlass_fp4_moe_no_graph(m=m, n=n, k=k, e=e, topk=topk, dtype=torch.half)
    # test_cutlass_moe_EP(m=m, n=n, k=k, e=e, topk=topk, local_expert_size=local_expert_size)
