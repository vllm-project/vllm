# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm._custom_ops import cpu_gemm_wna16  # noqa: E402
from vllm.model_executor.kernels.linear.mixed_precision.cpu import (  # noqa: E402
    _get_isa_hint,
)


@pytest.mark.cpu_test
def test_cpu_gemm_wna16_3d_input():
    """cpu_gemm_wna16 flattens 3D [B, T, K] to 2D and reshapes the output back."""
    if not hasattr(torch.ops._C, "cpu_gemm_wna16"):
        pytest.skip("cpu_gemm_wna16 op is not available")

    B, T, K, N = 2, 5, 64, 64
    pack_factor = 8
    group_size = 32
    dtype = torch.bfloat16
    q_weight = torch.zeros(N // 16, K * 16 // pack_factor, dtype=torch.int32)
    scales = torch.ones(K // group_size, N, dtype=dtype)

    x_3d = torch.randn(B, T, K, dtype=dtype)
    x_2d = x_3d.reshape(-1, K)
    isa_hint = _get_isa_hint(dtype)
    kwargs = dict(
        q_weight=q_weight,
        scales=scales,
        zeros=None,
        g_idx=None,
        bias=None,
        pack_factor=pack_factor,
        isa_hint=isa_hint,
    )

    out_3d = cpu_gemm_wna16(input=x_3d, **kwargs)
    out_2d = cpu_gemm_wna16(input=x_2d, **kwargs)

    assert out_3d.shape == (B, T, N)
    assert out_2d.shape == (B * T, N)
    torch.testing.assert_close(out_3d.reshape(-1, N), out_2d)


MODELS = [
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
    "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",
    "RedHatAI/Qwen3-1.7B-quantized.w4a16",  # with zp
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",
    "Qwen/Qwen3-0.6B-FP8",  # FP8 W8A16 block-quantized linear
    "Qwen/Qwen3-30B-A3B-FP8",  # FP8 W8A16 block-quantized MoE
    "openai/gpt-oss-20b",  # MXFP4 W4A16
    "QuixiAI/Qwen3-30B-A3B-AWQ",  # AWQ W4A16 MoE
    "Qwen/Qwen3-30B-A3B-GPTQ-Int4",  # GPTQ W4A16 MoE
    "RedHatAI/Qwen3-30B-A3B-quantized.w4a16",  # compressed-tensors W4A16 MoE
]
DTYPE = ["bfloat16"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_cpu_quant(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)
    assert output
    print(output)
