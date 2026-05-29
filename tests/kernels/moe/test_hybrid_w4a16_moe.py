# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HybridW4A16MoEExperts (Triton prefill + HIP decode).

Validates the hybrid MoE kernel by:
1. Creating random fp16 MoE weights
2. Quantizing them to symmetric 4-bit with group_size=32 or 128
3. Packing into ExLlama shuffle format [E, N, K//8] int32
4. Running HybridW4A16MoEExperts via FusedMoEModularKernel
5. Comparing against torch_experts reference using dequantized weights

Tests exercise both paths:
- Decode (M<=5): HIP wvSplitK_int4 kernel
- Prefill (M>5): Triton fused_moe kernel with use_shuffle_w4a16
"""

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (
    pack_int4_exllama_shuffle,
)
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.hybrid_w4a16_moe import (
    HybridW4A16MoEExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

NUM_BITS = 4
PACK_FACTOR = 32 // NUM_BITS  # 8 nibbles per int32


def _symmetric_quantize_4bit_skinny(
    w: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric 4-bit quantization → skinny ExLlama format.

    Input:  w [K, N] fp16
    Returns:
      q_skinny: [N, K//8] int32 (ExLlama shuffle packed)
      scales:   [N, K//G] fp16 (skinny layout)
      w_ref:    [K, N] fp16 (dequantized reference)
    """
    K, N = w.shape
    assert K % group_size == 0
    num_groups = K // group_size

    w_grouped = w.reshape(num_groups, group_size, N)
    abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scales = abs_max / 7.0

    # Quantize to unsigned [0, 15] with zero_point = 8
    w_q = torch.round(w_grouped / scales).clamp(-7, 7).int() + 8
    w_q = w_q.reshape(K, N)

    # Dequantized reference
    w_ref = (
        ((w_q.float() - 8.0).reshape(num_groups, group_size, N) * scales)
        .reshape(K, N)
        .half()
    )

    # Pack into ExLlama shuffle: transpose to [N, K], pack to [N, K//8]
    w_q_uint4 = w_q.to(torch.uint8)  # values in [0, 15]
    w_q_t = w_q_uint4.t().contiguous()  # [N, K]
    q_skinny = pack_int4_exllama_shuffle(w_q_t)  # [N, K//8] int32

    # Scales: [num_groups, N] → [N, num_groups] (skinny layout)
    scales_skinny = scales.squeeze(1).t().contiguous()  # [N, K//G]

    return q_skinny, scales_skinny, w_ref


def _make_hybrid_moe_weights(
    E: int,
    K: int,
    N: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fake skinny-packed MoE weights for E experts.

    Returns (w_skinny, scales, w_ref) where:
    - w_skinny: [E, N, K//8] int32 (ExLlama shuffle packed)
    - scales:   [E, N, K//G] fp16 (skinny layout)
    - w_ref:    [E, N, K] fp16 (torch_experts convention)
    """
    all_skinny = []
    all_scales = []
    all_ref = []

    for _ in range(E):
        w_fp = torch.randn(K, N, device=device, dtype=torch.float16) / 10.0
        q_skinny, scales, w_ref = _symmetric_quantize_4bit_skinny(w_fp, group_size)
        all_skinny.append(q_skinny)
        all_scales.append(scales)
        all_ref.append(w_ref.t())  # transpose to [N, K] for torch_experts

    w_skinny = torch.stack(all_skinny)  # [E, N, K//8]
    w_scales = torch.stack(all_scales)  # [E, N, K//G]
    w_ref = torch.stack(all_ref)  # [E, N, K]

    return w_skinny, w_scales, w_ref


def _run_hybrid_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    group_size: int,
    force_triton: bool = False,
    force_hip: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build weights, run HybridW4A16MoEExperts and torch_experts reference.

    Args:
        force_triton: Force the Triton prefill path for all batch sizes.
        force_hip: Force the HIP wvSplitK path for all batch sizes.

    Returns (hybrid_output, reference_output).
    """
    set_random_seed(1)
    device = torch.device("cuda")

    assert k % group_size == 0

    # w1: gate+up projection [E, 2*N, K//8], ref [E, 2*N, K]
    w1_skinny, w1_scales, w1_ref = _make_hybrid_moe_weights(
        e, k, 2 * n, group_size, device
    )
    # w2: down projection [E, K, N//8], ref [E, K, N]
    w2_skinny, w2_scales, w2_ref = _make_hybrid_moe_weights(e, n, k, group_size, device)

    hidden = torch.randn(m, k, device=device, dtype=torch.float16) / 10
    scores = torch.randn(m, e, device=device, dtype=torch.float16)

    topk_weights, topk_ids, _ = fused_topk(hidden, scores, topk, False)

    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=None,
        w2_zp=None,
        block_shape=[0, group_size],
    )

    moe_config = make_dummy_moe_config(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size_per_partition=n,
        in_dtype=torch.float16,
    )

    experts = HybridW4A16MoEExperts(
        moe_config=moe_config,
        quant_config=quant_config,
    )

    orig_threshold = HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE
    if force_triton:
        HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE = 0
    elif force_hip:
        HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE = 10000

    try:
        mk = FusedMoEKernelModularImpl(
            fused_experts=experts,
            prepare_finalize=MoEPrepareAndFinalizeNoDPEPModular(),
            shared_experts=None,
        )

        init_workspace_manager(device)
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):
            torch_output = torch_experts(
                hidden,
                w1_ref,
                w2_ref,
                topk_weight=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=e,
            )

            hybrid_out = mk.apply(
                hidden_states=hidden,
                w1=w1_skinny,
                w2=w2_skinny,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=e,
                expert_map=None,
                activation=MoEActivation.SILU,
                apply_router_weight_on_input=False,
            )
    finally:
        HybridW4A16MoEExperts.MAX_SKINNY_BATCH_SIZE = orig_threshold

    return hybrid_out, torch_output


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
@pytest.mark.parametrize("m", [1, 4, 16, 64])
@pytest.mark.parametrize(
    "n,k",
    [
        (256, 256),
        (512, 256),
        # Qwen3.5-A3B MoE shapes -- exercise the gfx11 K=2048 (gate_up) and
        # K=512 (down) dispatch branches added by this PR.
        (1024, 2048),
        (2048, 512),
    ],
)
@pytest.mark.parametrize("e,topk", [(8, 2), (16, 4)])
@pytest.mark.parametrize("group_size", [32, 128])
def test_hybrid_w4a16_moe(m: int, n: int, k: int, e: int, topk: int, group_size: int):
    """Test natural dispatch: HIP for decode (m<=5), Triton for prefill (m>5)."""
    hybrid_out, torch_output = _run_hybrid_moe(m, n, k, e, topk, group_size)
    torch.testing.assert_close(hybrid_out, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
@pytest.mark.parametrize("m", [1, 4, 16])
@pytest.mark.parametrize("n,k", [(256, 256)])
@pytest.mark.parametrize("e,topk", [(8, 2)])
@pytest.mark.parametrize("group_size", [32])
def test_hybrid_w4a16_moe_force_triton(
    m: int, n: int, k: int, e: int, topk: int, group_size: int
):
    """Force the Triton path for all batch sizes (including m=1)."""
    hybrid_out, torch_output = _run_hybrid_moe(
        m, n, k, e, topk, group_size, force_triton=True
    )
    torch.testing.assert_close(hybrid_out, torch_output, atol=2e-2, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="HybridW4A16MoEExperts requires ROCm",
)
@pytest.mark.parametrize("m", [1, 16, 64])
@pytest.mark.parametrize("n,k", [(256, 256)])
@pytest.mark.parametrize("e,topk", [(8, 2)])
@pytest.mark.parametrize("group_size", [32])
def test_hybrid_w4a16_moe_force_hip(
    m: int, n: int, k: int, e: int, topk: int, group_size: int
):
    """Force the HIP wvSplitK path for all batch sizes (including m=64)."""
    hybrid_out, torch_output = _run_hybrid_moe(
        m, n, k, e, topk, group_size, force_hip=True
    )
    torch.testing.assert_close(hybrid_out, torch_output, atol=2e-2, rtol=0)
