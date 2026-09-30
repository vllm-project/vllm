# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.scalar_type import scalar_types

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

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


def _pack_block_weight(w_q: torch.Tensor) -> torch.Tensor:
    """Repack [input_size, output_size] int4 weights into the blocked layout
    consumed by ``ops.cpu_gemm_wna16`` (16 output channels per block)."""
    input_size = w_q.shape[0]
    packed = pack_quantized_values_into_int32(w_q, scalar_types.uint4b8, 1)
    return (
        packed.view(input_size, -1, 2)
        .permute(1, 0, 2)
        .reshape(-1, input_size * 2)
        .contiguous()
    )


# The WNA16 int4 GEMM is the kernel behind AWQ/GPTQ on the CPU backend. It
# must stay available (and correct) on x86 CPUs without AVX-512, where only
# the portable vector/fallback path is compiled.
@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.X86,
    reason="cpu_gemm_wna16 is only implemented for x86",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("zero_points", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_cpu_wna16_matches_dequant_reference(dtype, zero_points, num_tokens):
    torch.manual_seed(0)
    input_size, output_size, group_size = 128, 64, 32
    group_num = input_size // group_size

    w_q = torch.randint(0, 16, (input_size, output_size), dtype=torch.int32)
    scales = (torch.rand(group_num, output_size) * 0.1 + 0.01).to(dtype)
    q_weight = _pack_block_weight(w_q)

    # GPTQ stores a symmetric 4-bit weight offset by 8; AWQ subtracts a
    # per-channel zero point instead.
    dequant = w_q.to(torch.float32) - 8.0
    zeros = None
    if zero_points:
        zp = torch.randint(0, 16, (group_num, output_size), dtype=torch.int32)
        dequant = w_q.to(torch.float32) - zp.repeat_interleave(group_size, dim=0).to(
            torch.float32
        )
        zeros = pack_quantized_values_into_int32(
            zp, scalar_types.uint4b8, 1
        ).contiguous()

    weight = (dequant * scales.float().repeat_interleave(group_size, dim=0)).to(dtype)
    x = torch.randn(num_tokens, input_size, dtype=dtype)
    expected = x.float() @ weight.float()

    out = ops.cpu_gemm_wna16(
        input=x,
        q_weight=q_weight,
        scales=scales,
        zeros=zeros,
        bias=None,
        pack_factor=8,
        isa_hint="vec",
    )
    torch.testing.assert_close(out.float(), expected, atol=0.15, rtol=0.02)
