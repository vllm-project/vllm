# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.moe.utils import make_test_quant_config, make_test_weights
from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    _valid_deep_gemm_shape,
    deep_gemm_moe_fp8,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk,
    modular_triton_fused_moe,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used,
)
from vllm.utils.import_utils import has_deep_gemm

dg_available = has_deep_gemm()

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()

# Test configurations
DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
# Deepseek-V3's intermediate size 18432, so N is 18432*2/8=4608 at TP8
# and its hidden size is 7168.
MNK_FACTORS = [
    (1, 128, 128),
    (1, 128, 7168),
    (1, 1024, 7168),
    (1, 4608, 128),
    (1, 4608, 7168),
    (83, 128, 128),
    (83, 512, 512),
    (83, 4608, 512),
    (83, 4608, 7168),
    (128, 512, 512),
    (128, 1024, 7168),
    (128, 4608, 7168),
    (2048, 128, 128),
    (2048, 1024, 7168),
    (2048, 4608, 512),
    (2048, 4608, 7168),
    (8192, 128, 128),
    (8192, 128, 7168),
    (8192, 1024, 7168),
    (8192, 4608, 7168),
]

MNK_FACTORS_DG = [
    (128, 128, 128),
    (128, 128, 7168),
    (128, 1024, 7168),
    (128, 4608, 128),
    (128, 4608, 7168),
    (192, 512, 512),
    (192, 1024, 7168),
    (192, 4608, 7168),
    (1335, 128, 128),
    (1335, 1024, 7168),
    (1335, 4608, 512),
    (1335, 4608, 7168),
    (2048, 128, 128),
    (2048, 128, 7168),
    (2048, 1024, 7168),
    (2048, 4608, 7168),
]

BLOCK_SIZE = [[128, 128]]
E = [2, 8, 16]  # [128, 256]
TOP_KS = [1, 2, 6]
SEEDS = [0]


def torch_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, block_shape):
    """Fused moe with block-wise quantization using native torch."""
    B, D = a.shape
    topk = topk_ids.size(1)
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_fp8(a, block_k)
    a_q = a_q.to(torch.float32)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
            )
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_fp8(act_out, block_k)
            out[mask] = native_w8a8_block_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


# Skip all tests if CUDA is not available
pytest.importorskip("torch.cuda")


@pytest.fixture(autouse=True)
def setup_cuda():
    torch.set_default_device("cuda")


@pytest.mark.parametrize(("M", "N", "K"), MNK_FACTORS)
@pytest.mark.parametrize("E", E)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("block_size", BLOCK_SIZE)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_w8a8_block_fp8_fused_moe(
    M, N, K, E, topk, block_size, dtype, seed, monkeypatch
):
    if topk > E:
        pytest.skip(f"Skipping test; topk={topk} > E={E}")

    torch.manual_seed(seed)

    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "2048")

    a = torch.randn((M, K), dtype=dtype) / 10
    score = torch.randn((M, E), dtype=dtype)

    w1, w2, quant_config = make_test_quant_config(
        E,
        N,
        K,
        dtype,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=block_size,
    )

    m_fused_moe = modular_triton_fused_moe(quant_config)

    topk_weights, topk_ids, _ = fused_topk(a, score.float(), topk, False)

    # Set the context to avoid lots of warning spam.
    with set_current_vllm_config(vllm_config):
        ref_out = torch_w8a8_block_fp8_moe(
            a,
            w1,
            w2,
            quant_config.w1_scale,
            quant_config.w2_scale,
            topk_weights,
            topk_ids,
            block_size,
        )

        out = fused_experts(
            a, w1, w2, topk_weights, topk_ids, quant_config=quant_config
        )

        m_out = m_fused_moe(a, w1, w2, topk_weights, topk_ids)

    # 0.039 only needed for M >= 8192
    tol = 0.035 if M < 8192 else 0.039
    torch.testing.assert_close(out, ref_out, atol=tol, rtol=tol)
    torch.testing.assert_close(m_out, ref_out, atol=tol, rtol=tol)


@pytest.mark.parametrize(("M", "N", "K"), MNK_FACTORS_DG)
@pytest.mark.parametrize("E", E)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not dg_available, reason="DeepGemm kernels not available.")
@pytest.mark.skipif(is_deep_gemm_e8m0_used(), reason="Not E8M0 scale MOE")
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_fused_moe(M, N, K, E, topk, seed, monkeypatch):
    if topk > E:
        pytest.skip(f"Skipping test: topk={topk} > E={E}")

    if not _valid_deep_gemm_shape(M, N, K):
        pytest.skip(f"Skipping test: invalid size m={M}, n={N}, k={K}")

    chunk_size = 1024

    torch.manual_seed(seed)

    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", str(chunk_size))
    block_size = get_mk_alignment_for_contiguous_layout()
    dtype = torch.bfloat16

    a = torch.randn((M, K), dtype=dtype) / 10
    score = torch.randn((M, E), dtype=dtype)

    (_, w1, w1_s, _), (_, w2, w2_s, _) = make_test_weights(
        E,
        N,
        K,
        dtype,
        torch.float8_e4m3fn,
        per_out_ch_quant=False,
        block_shape=block_size,
    )

    # Note: for now use_compile will error out if the problem size is
    # large enough to trigger chunking. I'm leaving the flag and
    # setup code in case we are able to revisit this later.
    use_compile = False

    use_cudagraph = (
        chunk_size < M and N >= 1024 and K >= 1024 and current_platform.is_cuda_alike()
    )

    topk_weights, topk_ids, _ = fused_topk(a, score.float(), topk, False)

    # Set the context to avoid lots of warning spam.
    with set_current_vllm_config(vllm_config):
        ref_out = torch_w8a8_block_fp8_moe(
            a, w1, w2, w1_s, w2_s, topk_weights, topk_ids, block_size
        )

        if use_compile:
            deep_gemm_moe_fp8_fn = torch.compile(
                deep_gemm_moe_fp8, backend="inductor", fullgraph=True
            )
            torch._dynamo.mark_dynamic(a, 0)
            torch._dynamo.mark_dynamic(topk_weights, 0)
            torch._dynamo.mark_dynamic(topk_ids, 0)
        else:
            deep_gemm_moe_fp8_fn = deep_gemm_moe_fp8

        out = deep_gemm_moe_fp8_fn(a, w1, w2, w1_s, w2_s, topk_weights, topk_ids)

        if use_cudagraph:
            out.fill_(0)
            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = deep_gemm_moe_fp8_fn(
                    a, w1, w2, w1_s, w2_s, topk_weights, topk_ids
                )
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()

    torch.testing.assert_close(out, ref_out, atol=0.035, rtol=0.035)
