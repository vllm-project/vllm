# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed-tensors FP4 layers refuse TP splits that cut block-scale groups.

A row-parallel shard narrows the packed codes at element offsets but the block
scales at whole groups, so a rank whose input slice starts inside a group would
pair its codes with other groups' scales. The sizes are those of
RedHatAI/gemma-4-26B-A4B-it-NVFP4 (dense down_proj K = 2112, expert down_proj
K = 704: 16.5 and 5.5 NVFP4 groups per rank at TP=8, whole groups at TP=4) and
an MXFP4 expert down_proj with K = 640 (2.5 MXFP4 groups per rank at TP=8).
"""

from dataclasses import replace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A4Mxfp4,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="FP4 linear kernels need CUDA."
)

# name: (scheme or method factory, unsharded input size, group size)
LINEAR_SCHEMES = {
    "nvfp4": (lambda: CompressedTensorsW4A4Fp4(), 2112, 16),
    "nvfp4a16": (lambda: CompressedTensorsW4A4Fp4(use_a16=True), 2112, 16),
    "mxfp4": (lambda: CompressedTensorsW4A4Mxfp4(), 640, 32),
}
MOE_METHODS = {
    "nvfp4": (lambda moe: CompressedTensorsW4A4Nvfp4MoEMethod(moe), 704, 16),
    "nvfp4a16": (
        lambda moe: CompressedTensorsW4A4Nvfp4MoEMethod(moe, use_a16=True),
        704,
        16,
    ),
    "mxfp4": (lambda moe: CompressedTensorsW4A4Mxfp4MoEMethod(moe), 640, 32),
}


@pytest.mark.parametrize("scheme_name", LINEAR_SCHEMES)
def test_row_parallel_split_inside_scale_group_is_refused(dist_init, scheme_name):
    make_scheme, input_size, group_size = LINEAR_SCHEMES[scheme_name]
    scheme = make_scheme()

    def create_weights(tp_size: int) -> torch.nn.Module:
        layer = torch.nn.Module()
        scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size // tp_size,
            output_partition_sizes=[64],
            output_size=64,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        return layer

    with pytest.raises(ValueError, match="not divisible by group_size"):
        create_weights(tp_size=8)
    layer = create_weights(tp_size=4)
    assert layer.weight_scale.shape[1] * group_size == input_size // 4


@pytest.mark.parametrize("method_name", MOE_METHODS)
def test_moe_w2_split_inside_scale_group_is_refused(dist_init, method_name):
    make_method, intermediate_size, group_size = MOE_METHODS[method_name]

    def create_weights(tp_size: int) -> torch.nn.Module:
        moe = FusedMoEConfig(
            num_experts=2,
            experts_per_token=1,
            hidden_dim=256,
            intermediate_size=intermediate_size,
            num_local_experts=2,
            num_logical_experts=2,
            activation=MoEActivation.SILU,
            device="cuda",
            routing_method=RoutingMethodType.TopK,
            moe_parallel_config=replace(
                FusedMoEParallelConfig.make_no_parallel(), tp_size=tp_size
            ),
            in_dtype=torch.bfloat16,
        )
        method = make_method(moe)
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            num_experts=2,
            hidden_size=256,
            intermediate_size_per_partition=moe.intermediate_size_per_partition,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        return layer

    with pytest.raises(ValueError, match="not divisible by group_size"):
        create_weights(tp_size=8)
    layer = create_weights(tp_size=4)
    assert layer.w2_weight_scale.shape[2] * group_size == intermediate_size // 4
