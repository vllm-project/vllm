# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (FLOAT4_E2M1_MAX,
                                                    FLOAT8_E4M3_MAX,
                                                    dequantize_nvfp4_to_dtype)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import cutlass_moe_fp4
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(reason="Nvfp4 Requires compute capability of 10 or above.",
                allow_module_level=True)

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
        quant_blocksize = 16
        round_up = lambda x, y: (x + y - 1) // y * y
        sf_w1_2n = round_up(2 * n, 128)
        sf_w1_k = round_up(k // quant_blocksize, 4)
        w1_blockscale = torch.empty((e, sf_w1_2n, sf_w1_k),
                                    device="cuda",
                                    dtype=torch.float8_e4m3fn)

        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
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

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

        a1_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)

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
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            m=m,
            n=n,
            k=k,
            e=e,
            device=a.device,
        )

        # Reference check:
        a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                          torch.amax(a.flatten(), dim=-1)).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(a_fp4,
                                               a_scale_interleaved,
                                               a_global_scale,
                                               dtype=a.dtype,
                                               device=a.device,
                                               block_size=quant_blocksize)

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(w1_q[idx],
                                                  w1_blockscale[idx],
                                                  w1_gs[idx],
                                                  dtype=w1.dtype,
                                                  device=w1.device,
                                                  block_size=quant_blocksize)
            w2_d[idx] = dequantize_nvfp4_to_dtype(w2_q[idx],
                                                  w2_blockscale[idx],
                                                  w2_gs[idx],
                                                  dtype=w2.dtype,
                                                  device=w2.device,
                                                  block_size=quant_blocksize)

        torch_output = torch_moe(a_in_dtype, w1_d, w2_d, score, topk, None)

        torch.testing.assert_close(torch_output,
                                   cutlass_output,
                                   atol=1e-1,
                                   rtol=1e-1)


if __name__ == "__main__":
    test_cutlass_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)
