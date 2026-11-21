# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401
from vllm.model_executor.determinism.batch_invariant import (
    dequantize_gptq_weight,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    gptq_quantize_weights,
)
from vllm.scalar_type import scalar_types


def test_gptq_shuffle_opcheck():
    weight = torch.randint(
        -2000000, 2000000, (1792, 4096), device="cuda", dtype=torch.int32
    )
    bit = 4
    opcheck(torch.ops._C.gptq_shuffle, (weight, bit))


def test_gptq_gemm_opcheck():
    a = torch.rand((240, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        -2000000, 2000000, (512, 6144), device="cuda", dtype=torch.int32
    )
    zeros = torch.zeros((32, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 6144), device="cuda", dtype=torch.float16)
    use_exllama = True
    bit = 4
    # Test both GPTQv1 and GPTQv2 format
    opcheck(torch.ops._C.gptq_gemm, (a, weight, zeros, scales, use_exllama, True, bit))
    opcheck(torch.ops._C.gptq_gemm, (a, weight, zeros, scales, use_exllama, False, bit))


def gptq_gemm_peak_bytes(size_m, use_exllama, bit):
    a = torch.rand((size_m, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        -2000000,
        2000000,
        (4096 * bit // 32, 4096),
        device="cuda",
        dtype=torch.int32,
    )
    zeros = torch.zeros((32, 4096 * bit // 32), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 4096), device="cuda", dtype=torch.float16)
    args = (a, weight, zeros, scales, use_exllama, False, bit)

    torch.ops._C.gptq_gemm(*args)
    torch.accelerator.synchronize()
    before = torch.accelerator.memory_allocated()
    torch.accelerator.reset_peak_memory_stats()
    output = torch.ops._C.gptq_gemm(*args)
    torch.accelerator.synchronize()
    assert output.shape == (size_m, 4096)
    return torch.accelerator.max_memory_allocated() - before


@pytest.mark.parametrize(
    "use_exllama,bit,fused_m,reconstruct_m",
    [(True, 4, 1, 64), (True, 8, 24, 25), (False, 4, 8, 9)],
)
def test_gptq_gemm_allocates_temp_dq_only_for_reconstruct(
    use_exllama, bit, fused_m, reconstruct_m
):
    temp_dq_bytes = 4096 * 4096 * 2
    fused_peak = gptq_gemm_peak_bytes(fused_m, use_exllama, bit)
    reconstruct_peak = gptq_gemm_peak_bytes(reconstruct_m, use_exllama, bit)

    assert fused_peak < temp_dq_bytes
    assert reconstruct_peak >= temp_dq_bytes


@pytest.mark.parametrize("quant_type", [scalar_types.uint4b8, scalar_types.uint8b128])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dequantize_gptq_weight_matches_reference(quant_type, group_size, dtype):
    # The batch-invariant AutoGPTQ path unpacks the checkpoint layout in
    # Python instead of repacking for Marlin; it must reproduce the reference
    # dequantized weight exactly.
    size_k, size_n = 512, 256
    torch.manual_seed(0)
    w = torch.randn(size_k, size_n, dtype=dtype, device="cuda")
    w_ref, q_w, scales = gptq_quantize_weights(w, quant_type, group_size)
    qweight = gptq_pack(q_w, quant_type.size_bits, size_k, size_n)

    w_deq = dequantize_gptq_weight(
        qweight,
        scales,
        num_bits=quant_type.size_bits,
        group_size=group_size,
        zero_bias=quant_type.bias,
    )

    assert w_deq.shape == (size_k, size_n)
    assert w_deq.dtype == dtype
    torch.testing.assert_close(w_deq, w_ref, rtol=0, atol=0)
