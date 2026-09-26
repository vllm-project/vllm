# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fine-grained MoE (E >= 128) token alignment bypass.

Validates that:
1. Small-M token alignment bypass produces bit-for-bit identical outputs
   to the aligned path (atol=0.0).
2. Adaptive BLOCK_SIZE_M correctly scales for E >= 128 to prevent excessive
   padding overhead.
3. Backward compatibility is fully maintained for coarse-grained MoE
   (e.g. E=8 Mixtral).
"""

import pytest
import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    fused_experts,
    get_default_config,
    moe_align_block_size,
)


def test_fine_grained_moe_default_config():
    """Verify adaptive BLOCK_SIZE_M and tokens_per_expert for E >= 128."""
    # E = 128 fine-grained experts (e.g. Gemma 4)
    cfg_m1 = get_default_config(M=1, E=128, N=704, K=2816, topk=8, dtype=None)
    assert cfg_m1["BLOCK_SIZE_M"] == 16

    cfg_m64 = get_default_config(M=64, E=128, N=704, K=2816, topk=8, dtype=None)
    assert cfg_m64["BLOCK_SIZE_M"] == 16

    cfg_m128 = get_default_config(M=128, E=128, N=704, K=2816, topk=8, dtype=None)
    assert cfg_m128["BLOCK_SIZE_M"] == 32

    cfg_m512 = get_default_config(M=512, E=128, N=704, K=2816, topk=8, dtype=None)
    assert cfg_m512["BLOCK_SIZE_M"] == 64

    # E = 8 coarse-grained experts (e.g. Mixtral)
    cfg_e8_m64 = get_default_config(M=64, E=8, N=14336, K=4096, topk=2, dtype=None)
    assert cfg_e8_m64["BLOCK_SIZE_M"] == 32


@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
def test_prepare_expert_assignment_crossover(M: int):
    """Verify crossover between naive bypass and aligned mode."""
    device = torch.device("cuda:0")
    E = 128
    top_k = 8
    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)
    config = {"BLOCK_SIZE_M": 64}

    sorted_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens=M,
        top_k_num=top_k,
        global_num_experts=E,
        expert_map=None,
    )

    if M * top_k * 4 <= E:
        # Naive bypass active (M <= 4)
        assert sorted_ids is None
        assert expert_ids.shape == (M * top_k,)
        assert config["BLOCK_SIZE_M"] == 16
        assert num_tokens_post_padded.item() == M * top_k * 16
    else:
        # Aligned mode active (M >= 5)
        assert sorted_ids is not None
        assert expert_ids.ndim == 1


@pytest.mark.parametrize("M", [1, 2, 4, 8, 16])
def test_fused_experts_fine_grained_parity(M: int):
    """Verify numerical accuracy and bit-for-bit parity of naive bypass."""
    device = torch.device("cuda:0")
    torch.manual_seed(42 + M)

    E = 128
    top_k = 8
    hidden_size = 2816
    intermediate_size = 704
    dtype = torch.bfloat16

    w1 = (
        torch.randn((E, 2 * intermediate_size, hidden_size), dtype=dtype, device=device)
        * 0.02
    )
    w2 = (
        torch.randn((E, intermediate_size, hidden_size), dtype=dtype, device=device)
        * 0.02
    )

    x = torch.randn((M, hidden_size), dtype=dtype, device=device)
    logits = torch.randn((M, E), dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)
    topk_ids = topk_ids.to(torch.int32)

    with set_current_vllm_config(VllmConfig()):
        out_fused = fused_experts(
            x,
            w1,
            w2.transpose(1, 2).contiguous(),
            topk_weights,
            topk_ids,
            global_num_experts=E,
        )

        out_ref = torch_experts(
            x,
            w1,
            w2.transpose(1, 2).contiguous(),
            topk_weights,
            topk_ids,
            global_num_experts=E,
        )
        torch.testing.assert_close(out_fused, out_ref, atol=2e-2, rtol=2e-2)

        if M * top_k * 4 <= E:
            # When naive bypass is active, verify bit-for-bit equivalence
            # with forced aligned mode
            orig_fn = fused_moe_mod._prepare_expert_assignment

            def forced_aligned(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map=None,
                **kwargs,
            ):
                """Force the aligned (sorting) path for parity comparison."""
                return moe_align_block_size(
                    topk_ids,
                    config["BLOCK_SIZE_M"],
                    global_num_experts,
                    expert_map,
                    ignore_invalid_experts=kwargs.get("ignore_invalid_experts", False),
                )

            fused_moe_mod._prepare_expert_assignment = forced_aligned
            try:
                out_aligned = fused_experts(
                    x,
                    w1,
                    w2.transpose(1, 2).contiguous(),
                    topk_weights,
                    topk_ids,
                    global_num_experts=E,
                )
            finally:
                fused_moe_mod._prepare_expert_assignment = orig_fn

            torch.testing.assert_close(out_fused, out_aligned, atol=0.0, rtol=0.0)


def test_fused_experts_expert_map():
    """Verify naive bypass works correctly when expert_map is provided."""
    device = torch.device("cuda:0")
    torch.manual_seed(123)

    E = 128
    top_k = 8
    M = 1
    hidden_size = 2816
    intermediate_size = 704
    dtype = torch.bfloat16

    w1 = (
        torch.randn((E, 2 * intermediate_size, hidden_size), dtype=dtype, device=device)
        * 0.02
    )
    w2 = (
        torch.randn((E, intermediate_size, hidden_size), dtype=dtype, device=device)
        * 0.02
    )

    x = torch.randn((M, hidden_size), dtype=dtype, device=device)
    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.full((M, top_k), 1.0 / top_k, dtype=dtype, device=device)

    # Identity expert_map
    expert_map = torch.arange(E, dtype=torch.int32, device=device)

    with set_current_vllm_config(VllmConfig()):
        out_with_map = fused_experts(
            x,
            w1,
            w2.transpose(1, 2).contiguous(),
            topk_weights,
            topk_ids,
            global_num_experts=E,
            expert_map=expert_map,
        )
        out_no_map = fused_experts(
            x,
            w1,
            w2.transpose(1, 2).contiguous(),
            topk_weights,
            topk_ids,
            global_num_experts=E,
            expert_map=None,
        )
        torch.testing.assert_close(out_with_map, out_no_map, atol=0.0, rtol=0.0)


def test_prepare_expert_assignment_with_expert_map():
    """Verify naive bypass remaps expert IDs via non-identity expert_map."""
    device = torch.device("cuda:0")
    E = 128
    top_k = 8
    M = 1  # M * top_k * 4 = 32 <= 128, so naive path is active

    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)

    # Non-identity expert_map: reverse mapping
    expert_map = torch.arange(E - 1, -1, -1, dtype=torch.int32, device=device)
    config = {"BLOCK_SIZE_M": 64}

    sorted_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens=M,
        top_k_num=top_k,
        global_num_experts=E,
        expert_map=expert_map,
        ignore_invalid_experts=False,
    )

    # Naive path should be active
    assert sorted_ids is None
    # Expert IDs should be remapped through expert_map
    expected_ids = expert_map[topk_ids.view(-1)]
    torch.testing.assert_close(expert_ids, expected_ids, atol=0, rtol=0)


def test_moe_sum_dispatch():
    """Verify moe_sum dispatches correctly for 2-arg and 4-arg paths."""
    device = torch.device("cuda:0")
    M, top_k, hidden = 2, 4, 64
    dtype = torch.float32

    input_tensor = torch.randn(M, top_k, hidden, dtype=dtype, device=device)

    # 2-arg path: topk_ids=None, expert_map=None
    output_2arg = torch.zeros(M, 1, hidden, dtype=dtype, device=device)
    ops.moe_sum(input_tensor, output_2arg, topk_ids=None, expert_map=None)
    expected = input_tensor.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output_2arg, expected, atol=1e-5, rtol=1e-5)

    # 4-arg path: with topk_ids and identity expert_map
    E = 16
    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)
    expert_map = torch.arange(E, dtype=torch.int32, device=device)
    output_4arg = torch.zeros(M, 1, hidden, dtype=dtype, device=device)
    ops.moe_sum(input_tensor, output_4arg, topk_ids=topk_ids, expert_map=expert_map)

    # Both paths should produce the same reduction result with identity map
    torch.testing.assert_close(output_4arg, expected, atol=1e-5, rtol=1e-5)
