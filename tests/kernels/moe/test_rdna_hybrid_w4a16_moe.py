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
    rdna_hybrid_w4a16_moe_kernel_available,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _process_weights_rdna_hybrid,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types


def _make_weights(
    E: int, K: int, N: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns compressed-tensors packed [E, N, K//8] int32 weights,
    [E, N, K//G] scales and the dequantized [E, N, K] reference."""
    w = torch.randn(E, N, K // group_size, group_size, device="cuda") / 10
    scales = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 7
    w_q = torch.round(w / scales).clamp(-7, 7).to(torch.int32) + 8
    w_ref = ((w_q - 8) * scales).reshape(E, N, K).half()
    w_packed = torch.stack(
        [
            pack_quantized_values_into_int32(
                w_q[e].reshape(N, K), scalar_types.uint4, packed_dim=1
            )
            for e in range(E)
        ]
    )
    return w_packed, scales.squeeze(-1).half(), w_ref


def _run(m: int, n: int, k: int, e: int, topk: int, group_size: int):
    torch.manual_seed(1)
    w1, w1_scale, w1_ref = _make_weights(e, k, 2 * n, group_size)
    w2, w2_scale, w2_ref = _make_weights(e, n, k, group_size)
    w1, w2, w1_scale, w2_scale, *_ = _process_weights_rdna_hybrid(
        w1, w2, w1_scale, w2_scale
    )

    hidden = torch.randn(m, k, device="cuda", dtype=torch.float16) / 10
    scores = torch.randn(m, e, device="cuda", dtype=torch.float16)
    topk_weights, topk_ids, _ = fused_topk(hidden, scores, topk, False)

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
        in_dtype=torch.float16,
    )
    kernel = mk.FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        RDNAHybridW4A16Experts(moe_config=moe_config, quant_config=quant_config),
    )

    with set_current_vllm_config(VllmConfig()):
        ref = torch_experts(
            hidden,
            w1_ref,
            w2_ref,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
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
            apply_router_weight_on_input=False,
        )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


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
@pytest.mark.parametrize("e,topk", [(8, 2), (16, 4)])
def test_rdna_hybrid_w4a16_moe(m, n, k, e, topk, workspace_init):
    """HIP for decode (m <= 5), Triton for prefill."""
    _run(m, n, k, e, topk, 128)


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
