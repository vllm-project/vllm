# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin NVFP4 MoE (w4a16) coverage.

This is the kernel path `modelopt` NVFP4 MoE checkpoints serve through on
GPUs without native FP4 (e.g. SM120 workstation Blackwell); it previously had
no unit coverage (`test_nvfp4_moe.py` covers the CUTLASS experts only), which
let a wrong-activation config-plumbing bug ship undetected for SwiGLU-OAI
models such as MiniMax-M3.
"""

import pytest
import torch

from tests.kernels.utils import torch_moe
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_nvfp4_like,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

NVFP4_GROUP_SIZE = 16


def test_marlin_quant_config_forwards_swiglu_params():
    """The MARLIN branch must forward alpha/beta/clamp — MarlinExperts
    silently defaults to alpha=1.0 / beta=0.0 otherwise, which runs the
    wrong activation in every expert of SwiGLU-OAI models."""
    e, n_groups = 4, 8
    cfg = make_nvfp4_moe_quant_config(
        backend=NvFp4MoeBackend.MARLIN,
        w13_scale=torch.empty(e, 16, n_groups),
        w2_scale=torch.empty(e, 16, n_groups),
        w13_scale_2=torch.empty(e),
        w2_scale_2=torch.empty(e),
        a13_scale=torch.empty(e),
        a2_scale=torch.empty(e),
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )
    assert cfg.gemm1_clamp_limit == 7.0
    assert cfg.gemm1_alpha == 1.702
    assert cfg.gemm1_beta == 1.0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("m", [17, 257])
@pytest.mark.parametrize("n,k", [(256, 512), (512, 1024)])
@pytest.mark.parametrize("e,topk", [(8, 2), (1, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_marlin_nvfp4_moe(m, n, k, e, topk, dtype):
    torch.manual_seed(7)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w13 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    w13_ref, w13_q, w13_s, w13_gs = [], [], [], []
    w2_ref, w2_q, w2_s, w2_gs = [], [], [], []
    for i in range(e):
        ref, q, s, gs = rand_marlin_weight_nvfp4_like(w13[i], NVFP4_GROUP_SIZE)
        w13_ref.append(ref.T)
        w13_q.append(q)
        w13_s.append(s)
        w13_gs.append(gs)
        ref, q, s, gs = rand_marlin_weight_nvfp4_like(w2[i], NVFP4_GROUP_SIZE)
        w2_ref.append(ref.T)
        w2_q.append(q)
        w2_s.append(s)
        w2_gs.append(gs)

    w13_ref = torch.stack(w13_ref)
    w2_ref = torch.stack(w2_ref)
    w13_q = torch.stack(w13_q)
    w2_q = torch.stack(w2_q)
    w13_s = torch.stack(w13_s)
    w2_s = torch.stack(w2_s)
    w13_gs = torch.stack(w13_gs).view(e)
    w2_gs = torch.stack(w2_gs).view(e)

    with set_current_vllm_config(VllmConfig()):
        torch_output = torch_moe(a, w13_ref, w2_ref, score, topk)

        # Match torch_moe exactly: softmax over all experts, topk, no renorm.
        score_soft = torch.softmax(score.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(score_soft, topk)
        topk_ids = topk_ids.to(torch.int32)

        marlin_output = fused_marlin_moe(
            hidden_states=a,
            w1=w13_q,
            w2=w2_q,
            bias1=None,
            bias2=None,
            w1_scale=w13_s,
            w2_scale=w2_s,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=e,
            expert_map=None,
            global_scale1=w13_gs,
            global_scale2=w2_gs,
            g_idx1=None,
            g_idx2=None,
            input_global_scale1=None,
            input_global_scale2=None,
            sort_indices1=None,
            sort_indices2=None,
            w1_zeros=None,
            w2_zeros=None,
            input_dtype=dtype,
            is_k_full=True,
        )

    rel = (marlin_output - torch_output).abs().sum() / torch_output.abs().sum()
    assert rel.item() < 0.05, f"relative error {rel.item():.5f}"
