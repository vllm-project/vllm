# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.scalar_type import scalar_types

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

from vllm._cpu_ops import cpu_gemm_wna16  # noqa: E402
from vllm.model_executor.kernels.linear.mixed_precision.cpu import (  # noqa: E402
    _get_isa_hint,
)

# Isolated runs of cpu_gemm_wna16 can hang unless AMX tiles are initialized.
# Guard so non-AMX hosts still take the vec/rvv path.
if torch.cpu._is_amx_tile_supported():
    torch.cpu._init_amx()


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
    "vllm-project/TinyLlama-1.1B-Chat-v1.0-AWQ",
    "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4",  # without g_idx
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


@pytest.mark.skipif(
    current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC,
    reason="PowerPC-only test",
)
def test_cpu_wna16_power_bf16_uses_vsx():
    """Use the POWER10 BF16 MMA microkernel instead of generic vectors."""
    assert _get_isa_hint(torch.bfloat16) == "vsx"
    assert _get_isa_hint(torch.float16) == "vec"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_cpu_quant(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)
    assert output
    print(output)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("wtype", [scalar_types.uint4, scalar_types.uint8])
@pytest.mark.parametrize("packed_dim", [0, 1])
@pytest.mark.parametrize("transposed", [False, True])
def test_quantized_packing_preserves_integer_lanes(
    dtype, wtype, packed_dim, transposed
):
    """Packing must preserve every lane before widening to the output dtype."""
    bits = wtype.size_bits
    mask = (1 << bits) - 1
    values = torch.arange(32 * 64).reshape(32, 64) % (mask + 1)
    if dtype != torch.uint8:
        values -= 1 << (bits - 1)
    values = values.to(dtype)
    if transposed:
        values = values.t()
    original = values.clone()

    lanes = np.moveaxis(values.numpy().astype(np.int64), packed_dim, -1)
    lanes = lanes.reshape(*lanes.shape[:-1], -1, 32 // bits)
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    packed = np.bitwise_or.reduce((lanes.astype(np.uint32) & mask) << shifts, axis=-1)
    expected = torch.from_numpy(
        np.moveaxis(packed.view(np.int32), -1, packed_dim).copy()
    )

    actual = pack_quantized_values_into_int32(values, wtype, packed_dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    unpacked = unpack_quantized_values_into_int32(actual, wtype, packed_dim)
    torch.testing.assert_close(unpacked, values.to(torch.int32) & mask, rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
