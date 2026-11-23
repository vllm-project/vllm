# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
    scales_shape_stride_dtype,
)
from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    BatchedTritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
    BatchedPrepareAndFinalize,
)
from vllm.model_executor.warmup import deep_gemm_warmup
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    calc_diff,
    fp8_m_grouped_gemm_nt_masked,
    is_deep_gemm_supported,
)

from .test_deepgemm import make_block_quant_fp8_weights
from .utils import make_dummy_moe_config

BLOCK_SIZE = [128, 128]


@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_masked_deepgemm_warmup_covers_runtime_expected_m(
    monkeypatch, tmp_path, workspace_init
) -> None:
    """Startup warmup must populate every masked-GEMM config reachable by the
    runtime's 16-token expected-M buckets."""
    monkeypatch.setenv("DG_JIT_CACHE_DIR", str(tmp_path))
    e, t, k, n = 4, 64, 128, 256
    w1, w2, w1_scale, w2_scale = make_block_quant_fp8_weights(e, n, k, BLOCK_SIZE)
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        per_act_token_quant=False,
        block_shape=BLOCK_SIZE,
    )
    experts = BatchedDeepGemmExperts(
        max_num_tokens=t,
        num_dispatchers=1,
        quant_config=quant_config,
        moe_config=make_dummy_moe_config(
            num_experts=e,
            num_local_experts=e,
            hidden_dim=k,
            intermediate_size=n // 2,
        ),
    )
    deep_gemm_warmup.GROUPED_FP8_GEMM_NT_MASKED_WARMUP_CACHE.clear()

    deep_gemm_warmup._deepgemm_grouped_fp8_gemm_nt_masked_warmup(
        experts, w1, w2, w1_scale, w2_scale
    )
    torch.cuda.synchronize()
    warmed_files = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    scale_fmt = DeepGemmQuantScaleFMT.from_oracle()
    scale_shape, scale_stride, scale_dtype = scales_shape_stride_dtype(
        e, t, k // 128, scale_fmt
    )
    a = torch.zeros((e, t, k), dtype=torch.float8_e4m3fn, device="cuda")
    a_scale = torch.zeros(scale_shape, dtype=scale_dtype, device="cuda").as_strided(
        scale_shape, scale_stride
    )
    # w1 packs the gate and up projections, so its output dimension is 2 * n.
    out = torch.empty((e, t, w1.size(1)), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.full((e,), t, dtype=torch.int32, device="cuda")
    for expected_m in range(16, t + 1, 16):
        fp8_m_grouped_gemm_nt_masked(
            (a, a_scale),
            (w1, w1_scale),
            out,
            masked_m,
            expected_m,
        )
    torch.cuda.synchronize()

    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == warmed_files


@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
@pytest.mark.parametrize("E", [16, 32])  # number of experts
@pytest.mark.parametrize("T", [256, 512])  # tokens per expert
@pytest.mark.parametrize("K", [128, 256])  # hidden dim
@pytest.mark.parametrize("N", [512, 1024])  # intermediate dim per expert
@pytest.mark.parametrize("topk", [2, 4])
@pytest.mark.parametrize("clamp_source", [None, "quant", "model"])
def test_batched_deepgemm_vs_triton(
    E: int, T: int, K: int, N: int, topk: int, clamp_source, monkeypatch, workspace_init
):
    """Compare BatchedDeepGemmExperts to BatchedTritonExperts."""
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    torch.manual_seed(0)

    device = "cuda"
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(E, N, K, BLOCK_SIZE)

    M = E * T  # total tokens
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16) / 10.0
    if clamp_source is not None:
        a *= 1000
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    a.clamp_(fp8_info.min, fp8_info.max)

    # random router outputs → top-k indices / weights
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    # token number for each expert
    cnt = torch.bincount(topk_ids.flatten(), minlength=E)
    max_cnt = int(cnt.max().item())
    # next power of 2 for max token number
    max_num_tokens = 1 << (max_cnt - 1).bit_length()

    prep_finalize = BatchedPrepareAndFinalize(
        max_num_tokens=max_num_tokens,
        num_local_experts=E,
        num_dispatchers=1,
        rank=0,
    )

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        per_act_token_quant=False,
        block_shape=BLOCK_SIZE,
        gemm1_clamp_limit=10.0 if clamp_source == "quant" else None,
    )
    moe_config = make_dummy_moe_config()
    moe_config.swiglu_limit = 10.0 if clamp_source == "model" else None

    # triton (reference)
    triton_experts = BatchedTritonExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
        moe_config=moe_config,
    )
    mk_triton = FusedMoEKernel(
        prep_finalize,
        triton_experts,
    )

    out_triton = mk_triton.apply(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    # deepgemm
    deepgemm_experts = BatchedDeepGemmExperts(
        max_num_tokens=max_num_tokens,
        num_dispatchers=1,
        quant_config=quant_config,
        moe_config=moe_config,
    )
    mk_deepgemm = FusedMoEKernel(
        prep_finalize,
        deepgemm_experts,
    )

    out_deepgemm = mk_deepgemm.apply(
        hidden_states=a,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=None,
        apply_router_weight_on_input=False,
    )

    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 1e-3, f"Output diff too large: {diff}"
