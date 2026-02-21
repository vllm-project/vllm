# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests for ExllamaExperts (fused MoE exllama 4-bit kernel).

This test validates the exllama-based MoE GPTQ kernel by:
1. Creating random fp16 MoE weights
2. Quantizing them to symmetric 4-bit with group_size=128
3. Packing into GPTQ int32 format and applying gptq_shuffle
4. Running ExllamaExperts via FusedMoEModularKernel
5. Comparing against torch_experts reference using dequantized weights
"""

import numpy as np
import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import torch_experts
from vllm._custom_ops import gptq_shuffle
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.exllama_moe import ExllamaExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager

NUM_BITS = 4
GROUP_SIZE = 128
PACK_FACTOR = 32 // NUM_BITS  # 8 nibbles per int32


def _symmetric_quantize_4bit(
    w: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric 4-bit quantization → (q_packed [K/8, N] int32, scales, w_ref).

    Returns packed weights ready for gptq_shuffle, per-group scales,
    and the dequantized reference tensor.
    """
    K, N = w.shape
    assert K % group_size == 0
    num_groups = K // group_size

    w_grouped = w.reshape(num_groups, group_size, N)
    abs_max = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scales = abs_max / 7.0  # symmetric 4-bit: range [-7, 7] mapped to [1, 15]

    # Quantize to unsigned [0, 15] with zero_point = 8
    w_q = torch.round(w_grouped / scales).clamp(-7, 7).int() + 8
    w_q = w_q.reshape(K, N)

    # Dequantized reference
    w_ref = (
        ((w_q.float() - 8.0).reshape(num_groups, group_size, N) * scales)
        .reshape(K, N)
        .half()
    )

    # Pack 8 rows of 4-bit values into one int32 row → shape [K/8, N]
    w_q_np = w_q.cpu().numpy().astype(np.uint32)
    packed = np.zeros((K // PACK_FACTOR, N), dtype=np.uint32)
    for i in range(PACK_FACTOR):
        packed |= w_q_np[i::PACK_FACTOR, :] << (NUM_BITS * i)
    q_packed = torch.from_numpy(packed.astype(np.int32)).to(w.device)

    scales_out = scales.squeeze(1)  # [num_groups, N]
    return q_packed, scales_out, w_ref


def _make_exllama_moe_weights(
    E: int,
    K: int,
    N: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fake GPTQ-packed MoE weights for E experts.

    Returns (w_packed, scales, qzeros, w_ref) where:
    - w_packed: [E, K/8, N] int32 (exllama format)
    - scales:   [E, num_groups, N] fp16
    - qzeros:   [E, num_groups, N/8] int32
    - w_ref:    [E, N, K] fp16 (transposed, torch_experts convention)
    """
    num_groups = K // group_size
    dummy_perm = torch.empty(0, dtype=torch.int32, device=device)

    all_packed = []
    all_scales = []
    all_ref = []

    for _ in range(E):
        w_fp = torch.randn(K, N, device=device, dtype=torch.float16) / 10.0
        q_packed, scales, w_ref = _symmetric_quantize_4bit(w_fp, group_size)
        gptq_shuffle(q_packed, dummy_perm, NUM_BITS)
        all_packed.append(q_packed)
        all_scales.append(scales)
        all_ref.append(w_ref.t())  # transpose to [N, K] for torch_experts

    w_packed = torch.stack(all_packed)  # [E, K/8, N]
    w_scales = torch.stack(all_scales)  # [E, num_groups, N]
    w_ref = torch.stack(all_ref)  # [E, N, K]

    # GPTQv1 adds +1 to qzeros, so store 7 per nibble → 0x77777777
    qzeros = torch.full(
        (E, num_groups, N // PACK_FACTOR),
        0x77777777,
        dtype=torch.int32,
        device=device,
    )

    return w_packed, w_scales, qzeros, w_ref


@pytest.mark.skipif(
    not (current_platform.is_rocm() or current_platform.is_cuda()),
    reason="Requires ROCm or CUDA",
)
@pytest.mark.parametrize("m", [1, 4, 16])
@pytest.mark.parametrize("n,k", [(256, 256), (512, 256)])
@pytest.mark.parametrize("e,topk", [(8, 2), (16, 4)])
def test_exllama_moe(m: int, n: int, k: int, e: int, topk: int):
    torch.cuda.manual_seed(1)
    device = torch.device("cuda")
    group_size = GROUP_SIZE

    assert k % group_size == 0
    assert n % PACK_FACTOR == 0

    # w1: gate+up projection, exllama [E, K/8, 2*N], ref [E, 2*N, K]
    w1_packed, w1_scales, w1_qzeros, w1_ref = _make_exllama_moe_weights(
        e, k, 2 * n, group_size, device
    )
    # w2: down projection, exllama [E, N/8, K], ref [E, K, N]
    w2_packed, w2_scales, w2_qzeros, w2_ref = _make_exllama_moe_weights(
        e, n, k, group_size, device
    )

    hidden = torch.randn(m, k, device=device, dtype=torch.float16) / 10
    scores = torch.randn(m, e, device=device, dtype=torch.float16)

    topk_weights, topk_ids, _ = fused_topk(hidden, scores, topk, False)

    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        w1_zp=w1_qzeros,
        w2_zp=w2_qzeros,
        block_shape=[0, group_size],
    )

    moe_config = make_dummy_moe_config(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size_per_partition=n,
        in_dtype=torch.float16,
    )

    mk = FusedMoEModularKernel(
        fused_experts=ExllamaExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
        prepare_finalize=MoEPrepareAndFinalizeNoEP(),
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

        exllama_out = mk(
            hidden_states=hidden,
            w1=w1_packed,
            w2=w2_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            expert_map=None,
            activation=MoEActivation.SILU,
        )

    torch.testing.assert_close(exllama_out, torch_output, atol=2e-2, rtol=0)
