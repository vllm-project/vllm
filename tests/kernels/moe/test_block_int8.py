# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quant_utils import (
    native_per_token_group_quant_int8,
    native_w8a8_block_matmul,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
from vllm.platforms import current_platform

if current_platform.get_device_capability() < (7, 0):
    pytest.skip("INT8 Triton requires CUDA 7.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192

DTYPES = [torch.bfloat16]

MNK_FACTORS = [
    (1, 128, 128),
    (1, 128, 7168),
    (1, 1024, 7168),
    (1, 4096, 512),
    (1, 4096, 7168),
    (33, 512, 512),
    (33, 128, 7168),
    (33, 1024, 7168),
    (33, 4096, 128),
    (33, 4096, 7168),
    (128, 128, 128),
    (128, 1024, 7168),
    (128, 4096, 512),
    (128, 4096, 7168),
    (222, 512, 512),
    (222, 1024, 7168),
    (222, 4096, 7168),
    (2048, 128, 128),
    (2048, 1024, 7168),
    (2048, 4096, 4096),
]

E = [8, 24]
TOP_KS = [2, 6]
# BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
BLOCK_SIZE = [[128, 128]]
SEEDS = [0]


# For test
def torch_w8a8_block_int8_moe(a, w1, w2, w1_s, w2_s, score, topk, block_shape):
    """This function performs fused moe with block-wise quantization using
    native torch."""
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_int8(a, block_k)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
            )
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_int8(act_out, block_k)
            act_out = act_out.to(torch.float32)
            out[mask] = native_w8a8_block_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


@pytest.fixture(autouse=True, scope="module")
def setup_cuda():
    """Sets the default CUDA device for all tests in this module."""
    torch.set_default_device("cuda")


@pytest.mark.parametrize(("M", "N", "K"), MNK_FACTORS)
@pytest.mark.parametrize("E", E)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("block_size", BLOCK_SIZE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_w8a8_block_int8_fused_moe(M, N, K, E, topk, block_size, dtype, seed):
    """Tests the fused_moe kernel with W8A8 INT8 block quantization against a
    native torch reference."""
    torch.manual_seed(seed)

    a = torch.randn((M, K), dtype=dtype) / 10
    score = torch.randn((M, E), dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score.float(), topk, False)

    w1, w2, quant_config = make_test_quant_config(
        E,
        N,
        K,
        dtype,
        quant_dtype=torch.int8,
        per_act_token_quant=False,
        block_shape=block_size,
    )

    # Set the context to avoid lots of warning spam.
    with set_current_vllm_config(vllm_config):
        out = fused_experts(
            a, w1, w2, topk_weights, topk_ids, quant_config=quant_config
        )
        ref_out = torch_w8a8_block_int8_moe(
            a,
            w1,
            w2,
            quant_config.w1_scale,
            quant_config.w2_scale,
            score,
            topk,
            block_size,
        )

    # Check results
    torch.testing.assert_close(out, ref_out, atol=0.065, rtol=0.065)
