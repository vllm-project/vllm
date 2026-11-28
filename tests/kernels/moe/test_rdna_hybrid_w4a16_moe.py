# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for RDNAHybridW4A16Experts (HIP decode + Triton prefill).

Compares both paths against torch_experts on dequantized symmetric int4
weights, loaded through the same repack as compressed-tensors checkpoints.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.rdna_hybrid_w4a16_moe import (
    RDNAHybridW4A16Experts,
    _process_weights_rdna_hybrid,
    rdna_hybrid_w4a16_moe_kernel_available,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl


def _wmma_gemm1_available() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "fused_moe_wmma_int4_gemm"
    )


def _make_weights(
    E: int, K: int, N: int, group_size: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns compressed-tensors packed [E, N, K//8] int32 weights,
    [E, N, K//G] scales and the dequantized [E, N, K] reference."""
    w = torch.randn(E, N, K // group_size, group_size, device="cuda") / 10
    scales = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 7
    w_q = torch.round(w / scales).clamp(-7, 7).to(torch.int32) + 8
    w_ref = ((w_q - 8) * scales).reshape(E, N, K).to(dtype)
    w_packed = torch.stack(
        [
            pack_quantized_values_into_int32(
                w_q[e].reshape(N, K), scalar_types.uint4, packed_dim=1
            )
            for e in range(E)
        ]
    )
    return w_packed, scales.squeeze(-1).to(dtype), w_ref


def _make_experts(
    e: int,
    topk: int,
    k: int,
    n: int,
    group_size: int,
    dtype: torch.dtype,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> RDNAHybridW4A16Experts:
    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, group_size],
    )
    moe_config = make_dummy_moe_config(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size=n,
        in_dtype=dtype,
    )
    return RDNAHybridW4A16Experts(moe_config=moe_config, quant_config=quant_config)


def _run(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    dtype: torch.dtype = torch.float16,
    apply_router_weight_on_input: bool = False,
):
    torch.manual_seed(1)
    w1, w1_scale, w1_ref = _make_weights(e, k, 2 * n, group_size, dtype)
    w2, w2_scale, w2_ref = _make_weights(e, n, k, group_size, dtype)
    w1, w2, w1_scale, w2_scale, *_ = _process_weights_rdna_hybrid(
        w1, w2, w1_scale, w2_scale
    )

    hidden = torch.randn(m, k, device="cuda", dtype=dtype) / 10
    scores = torch.randn(m, e, device="cuda", dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(hidden, scores, topk, False)

    kernel = mk.FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        _make_experts(e, topk, k, n, group_size, dtype, w1_scale, w2_scale),
    )

    with set_current_vllm_config(VllmConfig()):
        ref = torch_experts(
            hidden,
            w1_ref,
            w2_ref,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            apply_router_weights_on_input=apply_router_weight_on_input,
        )
        out = kernel.apply(
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
    atol = 2e-2 if dtype == torch.float16 else 4e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=0)


@pytest.mark.skipif(
    not rdna_hybrid_w4a16_moe_kernel_available(),
    reason="RDNA hybrid W4A16 MoE kernel is not available",
)
@pytest.mark.parametrize("m", [1, 4, 16, 64])
@pytest.mark.parametrize(
    "n,k",
    [
        (256, 256),
        (512, 256),
        # Qwen3.5-A3B gate_up (K=2048) and down (K=512): tuned decode paths.
        (1024, 2048),
        (2048, 512),
    ],
)
@pytest.mark.parametrize("e,topk", [(8, 1), (8, 2), (16, 4), (16, 8)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rdna_hybrid_w4a16_moe(m, n, k, e, topk, dtype, workspace_init):
    """HIP for decode (m <= 5); WMMA GEMM1 (bf16, top_k=8) or Triton for
    prefill."""
    _run(m, n, k, e, topk, 128, dtype)


@pytest.mark.skipif(
    not rdna_hybrid_w4a16_moe_kernel_available(),
    reason="RDNA hybrid W4A16 MoE kernel is not available",
)
@pytest.mark.parametrize("m", [1, 4, 16, 64])
def test_rdna_hybrid_w4a16_moe_router_weight_on_input(m, workspace_init):
    """Router weight is applied once (topk == 1 only): HIP and Triton."""
    _run(m, 256, 256, 8, 1, 128, apply_router_weight_on_input=True)


@pytest.mark.skipif(
    not rdna_hybrid_w4a16_moe_kernel_available(),
    reason="RDNA hybrid W4A16 MoE kernel is not available",
)
@pytest.mark.parametrize(
    "m,k",
    [
        (1, 256),
        # block_size_m 2 and 4.
        (8, 256),
        (16, 256),
        (64, 256),
        # K * block_size_m exceeds the MoE LDS budget: the non-sml kernel.
        (16, 4096),
    ],
)
def test_rdna_hybrid_w4a16_moe_force_hip(m, k, monkeypatch, workspace_init):
    monkeypatch.setattr(RDNAHybridW4A16Experts, "MAX_SKINNY_BATCH_SIZE", 10000)
    _run(m, 256, k, 8, 2, 128)


@pytest.mark.skipif(
    not rdna_hybrid_w4a16_moe_kernel_available() or not _wmma_gemm1_available(),
    reason="WMMA GEMM1 op is not available",
)
@pytest.mark.parametrize("padded_rows", [False, True])
@pytest.mark.parametrize("K,N", [(2048, 1024), (1024, 512)])
@pytest.mark.parametrize("num_tokens", [6, 33, 256, 2048])
def test_fused_moe_wmma_int4_gemm_matches_triton(num_tokens, K, N, padded_rows):
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (  # noqa: E501
        pack_int4_exllama_shuffle,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_kernel_hybrid_triton,
    )

    E, top_k, group_size, block_m = 64, 8, 128, 32
    torch.manual_seed(0)
    device = "cuda"

    w_q = torch.randint(0, 16, (E, N, K), dtype=torch.int32, device=device)
    packed = torch.stack([pack_int4_exllama_shuffle(w_q[e]) for e in range(E)])
    if padded_rows:
        buf = torch.zeros((E, N, K // 8 + 32), dtype=torch.int32, device=device)
        buf[:, :, : K // 8].copy_(packed)
        w = buf[:, :, : K // 8]
        assert w.stride(1) > K // 8
    else:
        w = packed
    scales = (
        torch.rand(E, N, K // group_size, device=device, dtype=torch.bfloat16) * 0.02
        + 0.005
    )
    a = torch.randn(num_tokens, K, device=device, dtype=torch.bfloat16) / 10

    topk_ids = torch.topk(
        torch.randn(num_tokens, E, device=device), top_k, dim=-1
    ).indices.to(torch.int32)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, E, ignore_invalid_experts=True
    )
    P = num_tokens * top_k
    num_slots = sorted_token_ids.size(0)

    ref = torch.zeros(num_slots, N, device=device, dtype=torch.bfloat16)
    out = torch.zeros_like(ref)

    unit_scale = torch.ones(1, device=device, dtype=torch.bfloat16)
    experts = _make_experts(
        E, top_k, K, N // 2, group_size, torch.bfloat16, unit_scale, unit_scale
    )
    config = experts._triton_config(K, N, P)
    invoke_fused_moe_kernel_hybrid_triton(
        A=a,
        B=w,
        C=ref,
        B_scale=scales,
        topk_weights=None,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        compute_type=tl.bfloat16,
        group_size=group_size,
    )
    ops.fused_moe_wmma_int4_gemm(
        a,
        w,
        scales,
        out,
        sorted_token_ids,
        expert_ids,
        P,
        top_k,
        block_m,
        expert_ids.numel(),
    )

    ref_f, out_f = ref[:P].float(), out[:P].float()
    rel = (out_f - ref_f).norm() / ref_f.norm()
    assert rel < 1e-2, f"relative error {rel.item():.4e}"
