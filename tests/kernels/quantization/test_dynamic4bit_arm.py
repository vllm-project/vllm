# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
    Dynamic4bitLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.scalar_type import scalar_types


pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or current_platform.get_cpu_architecture() != CpuArchEnum.ARM,
    reason="Requires Arm CPU",
)


def make_config(weight_type, group_size, k=1024, n=1024, zero_points=False):
    return MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=weight_type,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=zero_points,
    )


@pytest.mark.parametrize("group_size", [64, 128])
def test_w4a16_kernel_selection(group_size):
    config = make_config(scalar_types.uint4b8, group_size)

    supported, reason = Dynamic4bitLinearKernel.can_implement(config)

    assert supported, reason

    selected = choose_mp_linear_kernel(config)

    assert selected is Dynamic4bitLinearKernel


def test_w4a8_backward_compatibility():
    config = make_config(scalar_types.int4, 64)

    supported, reason = Dynamic4bitLinearKernel.can_implement(config)

    assert supported, reason

    selected = choose_mp_linear_kernel(config)

    assert selected is Dynamic4bitLinearKernel


@pytest.mark.parametrize("group_size", [64, 128])
def test_w4a16_int4_unpack_repack(group_size):
    torch.manual_seed(0)

    k = 1024
    n = 128

    values = torch.randint(
        0, 16, (n, k), dtype=torch.int32
    )

    packed_int32 = pack_quantized_values_into_int32(
        values, scalar_types.uint4b8, packed_dim=1
    )

    unpacked = unpack_quantized_values_into_int32(
        packed_int32, scalar_types.uint4b8, packed_dim=1
    )

    assert torch.equal(unpacked, values)

    packed_uint8 = (
        (unpacked[:, 1::2] << 4) | unpacked[:, ::2]
    ).to(torch.uint8)

    expected_uint8 = (
        (values[:, 1::2] << 4) | values[:, ::2]
    ).to(torch.uint8)

    torch.testing.assert_close(
        packed_uint8,
        expected_uint8,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("group_size", [64, 128])
def test_w4a16_reject_zero_points(group_size):
    config = make_config(
        scalar_types.uint4b8,
        group_size,
        zero_points=True,
    )

    supported, reason = Dynamic4bitLinearKernel.can_implement(config)

    assert not supported
    assert "Zero points" in reason


@pytest.mark.parametrize("group_size", [0, 48, 96])
def test_invalid_group_size(group_size):
    config = make_config(scalar_types.uint4b8, group_size)

    supported, _ = Dynamic4bitLinearKernel.can_implement(config)

    assert not supported


def test_partitioned_weight_shape():
    config = MPLinearLayerConfig(
        full_weight_shape=(1024, 1024),
        partition_weight_shape=(512, 1024),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
    )

    supported, reason = Dynamic4bitLinearKernel.can_implement(config)

    assert supported, reason


@pytest.mark.parametrize("group_size", [64, 128])
def test_w4a16_kleidiai_matmul_correctness(group_size):
    """Verify W4A16 INT4 conversion and KleidiAI matmul numerically."""
    torch.manual_seed(0)

    m, k, n = 4, 256, 128

    # Symmetric INT4 values stored as offset-binary [0, 15].
    values = torch.randint(0, 16, (n, k), dtype=torch.int32)

    # Simulate compressed-tensors packed INT32 checkpoint weights.
    packed_int32 = pack_quantized_values_into_int32(
        values, scalar_types.uint4b8, packed_dim=1
    )

    # Apply the same conversion as Dynamic4bitLinearKernel.
    unpacked = unpack_quantized_values_into_int32(
        packed_int32, scalar_types.uint4b8, packed_dim=1
    )

    packed_uint8 = (
        (unpacked[:, 1::2] << 4) | unpacked[:, ::2]
    ).to(torch.uint8)

    # Groupwise BF16 scales.
    scales = torch.full(
        (n, k // group_size),
        0.02,
        dtype=torch.bfloat16,
    )

    # Use FP32 activations for groupwise KleidiAI matmul.
    x = torch.randn(m, k, dtype=torch.float32)

    packed_kleidiai = torch.ops.aten._dyn_quant_pack_4bit_weight(
        packed_uint8,
        scales,
        None,
        group_size,
        k,
        n,
    )

    output = torch.ops.aten._dyn_quant_matmul_4bit(
        x,
        packed_kleidiai,
        group_size,
        k,
        n,
    )

    # Reference: dequantize offset-binary INT4 values.
    weights_fp32 = (values.float() - 8.0) * scales.float().repeat_interleave(
        group_size, dim=1
    )

    reference = x @ weights_fp32.T

    # Dynamic activation quantization introduces numerical error.
    torch.testing.assert_close(
        output.float(),
        reference,
        rtol=0.08,
        atol=0.15,
    )
