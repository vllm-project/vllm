# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import (
    make_dummy_moe_config,
    make_test_weights,
    modular_triton_fused_moe,
)
from tests.kernels.quant_utils import native_w8a8_block_matmul
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts import triton_moe
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform

if current_platform.get_device_capability() < (8, 9):
    pytest.skip("FP8 Triton requires CUDA 8.9 or higher", allow_module_level=True)
if current_platform.is_fp8_fnuz():
    pytest.skip(
        "Tests in this file require float8_e4m3fn and platform does not support",
        allow_module_level=True,
    )

pytest.importorskip("torch.cuda")

vllm_config = VllmConfig()

E = 8
K = 512
N = 256
TOPK = 2
BLOCK_SHAPE = [128, 128]
DTYPE = torch.bfloat16
MS = [16, 64]
SEEDS = [0]
CLAMP_IGNORED_REASON = (
    "TritonExperts fp8 block fast path ignores the swiglu clamp; remove when fixed"
)


@pytest.fixture(autouse=True)
def setup_cuda():
    torch.set_default_device("cuda")


def swiglu_with_clamp(gate_up: torch.Tensor, clamp: float | None) -> torch.Tensor:
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d]
    up = gate_up[..., d:]
    if clamp is not None:
        gate = torch.clamp(gate, max=clamp)
        up = torch.clamp(up, min=-clamp, max=clamp)
    return F.silu(gate) * up


def torch_w8a8_block_fp8_moe_with_clamp(
    a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, block_shape, clamp
):
    """Blockwise fp8 MoE reference whose SwiGLU honors the gemm1 clamp limit."""
    B, D = a.shape
    topk = topk_ids.size(1)
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    block_k = block_shape[1]
    a_q, a_s = per_token_group_quant_fp8(a, block_k, dtype=current_platform.fp8_dtype())
    a_q = a_q.to(torch.float32)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
            )
            act_out = swiglu_with_clamp(inter_out, clamp)
            act_out_q, act_out_s = per_token_group_quant_fp8(
                act_out, block_k, dtype=current_platform.fp8_dtype()
            )
            out[mask] = native_w8a8_block_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def make_inputs_exceeding_clamp(M: int, seed: int):
    torch.manual_seed(seed)
    a = torch.randn((M, K), dtype=DTYPE) / 10
    num_amplified = max(1, M // 4)
    amplification = torch.logspace(2, 6, num_amplified, base=2, dtype=torch.float32)
    a[:num_amplified] *= amplification.to(DTYPE).unsqueeze(1)
    score = torch.randn((M, E), dtype=DTYPE)
    return a, score


def make_block_fp8_experts(clamp: float | None):
    (_, w1, w1_s, _), (_, w2, w2_s, _) = make_test_weights(
        E,
        N,
        K,
        DTYPE,
        torch.float8_e4m3fn,
        per_out_ch_quant=False,
        block_shape=BLOCK_SHAPE,
    )
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        block_shape=BLOCK_SHAPE,
        gemm1_clamp_limit=clamp,
    )
    kernel = modular_triton_fused_moe(make_dummy_moe_config(), quant_config)
    return w1, w2, w1_s, w2_s, kernel


def per_token_relative_l2(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    diff = (out.float() - ref.float()).norm(dim=-1)
    return diff / ref.float().norm(dim=-1).clamp_min(1e-6)


@pytest.mark.parametrize(
    "clamp",
    [
        pytest.param(
            7.0,
            marks=pytest.mark.xfail(strict=True, reason=CLAMP_IGNORED_REASON),
        ),
        pytest.param(
            10.0,
            marks=pytest.mark.xfail(strict=True, reason=CLAMP_IGNORED_REASON),
        ),
    ],
)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_triton_experts_fp8_block_honors_swiglu_clamp(
    clamp, M, seed, workspace_init, disable_deepgemm_ue8m0
):
    """Blockwise fp8 TritonExperts must clamp the SwiGLU between both GEMMs."""
    a, score = make_inputs_exceeding_clamp(M, seed)
    w1, w2, w1_s, w2_s, kernel = make_block_fp8_experts(clamp)
    topk_weights, topk_ids, _ = fused_topk(a, score.float(), TOPK, False)

    with set_current_vllm_config(vllm_config):
        clamped_ref = torch_w8a8_block_fp8_moe_with_clamp(
            a, w1, w2, w1_s, w2_s, topk_weights, topk_ids, BLOCK_SHAPE, clamp
        )
        unclamped_ref = torch_w8a8_block_fp8_moe_with_clamp(
            a, w1, w2, w1_s, w2_s, topk_weights, topk_ids, BLOCK_SHAPE, None
        )
        out = kernel.apply(
            a,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
            global_num_experts=E,
        )

    clamp_effect = per_token_relative_l2(unclamped_ref, clamped_ref).max().item()
    assert clamp_effect > 0.5, (
        f"inputs never exceeded clamp={clamp}, so this test cannot detect a "
        f"missing clamp: clamped and unclamped references differ by only "
        f"{clamp_effect:.4f} (max per-token relative L2)"
    )

    error = per_token_relative_l2(out, clamped_ref).max().item()
    assert error < 0.15, (
        f"max per-token relative L2 error against the clamped reference is "
        f"{error:.4f} for clamp={clamp}, M={M}"
    )


@pytest.mark.parametrize(
    "clamp",
    [
        None,
        pytest.param(
            10.0,
            marks=pytest.mark.xfail(strict=True, reason=CLAMP_IGNORED_REASON),
        ),
    ],
)
@torch.inference_mode()
def test_triton_experts_fp8_block_fast_path_selection(
    clamp, monkeypatch, workspace_init, disable_deepgemm_ue8m0
):
    """The fused SiLU+quant fast path runs only when no SwiGLU clamp is set."""
    a, score = make_inputs_exceeding_clamp(MS[0], SEEDS[0])
    w1, w2, _, _, kernel = make_block_fp8_experts(clamp)
    topk_weights, topk_ids, _ = fused_topk(a, score.float(), TOPK, False)

    fused_call_count = 0
    fused_silu_and_mul_quant = triton_moe.ops.silu_and_mul_per_block_quant

    def counting_silu_and_mul_per_block_quant(*args, **kwargs):
        nonlocal fused_call_count
        fused_call_count += 1
        return fused_silu_and_mul_quant(*args, **kwargs)

    monkeypatch.setattr(
        triton_moe.ops,
        "silu_and_mul_per_block_quant",
        counting_silu_and_mul_per_block_quant,
    )

    with set_current_vllm_config(vllm_config):
        kernel.apply(
            a,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
            global_num_experts=E,
        )

    if clamp is None:
        assert fused_call_count > 0, (
            "unclamped blockwise fp8 SiLU should take the fused fast path"
        )
    else:
        assert fused_call_count == 0, (
            f"clamp={clamp} must disable the fused fast path, which cannot "
            f"apply the clamp, but it ran {fused_call_count} times"
        )
