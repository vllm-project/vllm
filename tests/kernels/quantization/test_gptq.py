# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401
from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
    ExllamaLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
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


@pytest.mark.parametrize("quant_type", [scalar_types.uint4b8, scalar_types.uint8b128])
@pytest.mark.parametrize("group_size", [16, 32, 96])
@pytest.mark.parametrize("size_n", [256, 200])
@pytest.mark.parametrize("size_m", [0, 4, 64])
@pytest.mark.parametrize("zero_points", [False, True])
def test_exllama_linear_kernel(
    quant_type, group_size, size_n, size_m, zero_points, dist_init
):
    """The exllama kernels only switch quantization groups at 32-row steps from
    the start of each 128-row block of K, so group sizes such as 16 or 96 must
    still give the right output, also when N is not a multiple of 128."""
    torch.manual_seed(0)
    size_k = 768
    q = torch.randint(
        0, 1 << quant_type.size_bits, (size_k, size_n), dtype=torch.int32, device="cuda"
    )
    scales = (
        torch.rand(size_k // group_size, size_n, device="cuda") * 0.02 + 0.001
    ).half()
    x = torch.randn(size_m, size_k, device="cuda").half()
    zeros = torch.full_like(scales, quant_type.bias, dtype=torch.int32)
    if zero_points:
        zeros = torch.randint_like(zeros, 1, 1 << quant_type.size_bits)
    w_ref = (q - zeros.repeat_interleave(group_size, dim=0)).float() * (
        scales.float().repeat_interleave(group_size, dim=0)
    )
    ref = x.float() @ w_ref

    no_loader = lambda *args, **kwargs: None  # noqa: E731
    layer = torch.nn.Module()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(
            data=pack_quantized_values_into_int32(q, quant_type, packed_dim=0),
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=32 // quant_type.size_bits,
        ),
    )
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(
            data=scales, weight_loader=no_loader, input_dim=0, output_dim=1
        ),
    )
    if zero_points:
        # GPTQ v1 checkpoints store the zero points minus one.
        layer.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                pack_quantized_values_into_int32(zeros - 1, quant_type, packed_dim=1),
                requires_grad=False,
            ),
        )
    config = MPLinearLayerConfig(
        full_weight_shape=(size_k, size_n),
        partition_weight_shape=(size_k, size_n),
        weight_type=quant_type,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=zero_points,
    )
    kernel = ExllamaLinearKernel(
        config,
        w_q_param_name="qweight",
        w_s_param_name="scales",
        w_zp_param_name="qzeros" if zero_points else None,
    )
    kernel.process_weights_after_loading(layer)
    out = kernel.apply_weights(layer, x)

    assert out.shape == (size_m, size_n)
    assert (out.float() - ref).norm() <= 1e-2 * ref.norm()
