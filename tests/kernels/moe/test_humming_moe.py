# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingExpertsBase,
)
from vllm.model_executor.layers.quantization.utils import humming_utils


@pytest.mark.parametrize(
    ("activation", "expected_shapes"),
    [
        (MoEActivation.SILU, {"w13": (3712, 2688), "w2": (2688, 1856)}),
        (
            MoEActivation.RELU2_NO_MUL,
            {"w13": (1856, 2688), "w2": (2688, 1856)},
        ),
    ],
)
def test_default_sublayer_shapes_follow_activation(activation, expected_shapes):
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            activation=activation,
            has_bias=False,
            hidden_dim=2688,
            intermediate_size_per_partition=1856,
            num_local_experts=128,
        ),
        params_dtype=torch.bfloat16,
        layer_name="model.layers.0.mixer.experts",
        locks=object(),
    )

    with patch.object(
        humming_utils,
        "_process_single_sublayer",
        return_value=(object(), object()),
    ) as process_sublayer:
        humming_utils.convert_to_humming_moe_kernel_format(
            layer,
            weight_schema=object(),
            input_schema=object(),
        )

    actual_shapes = {
        call.kwargs["sublayer_name"]: (
            call.kwargs["shape_n"],
            call.kwargs["shape_k"],
        )
        for call in process_sublayer.call_args_list
    }
    assert actual_shapes == expected_shapes


class RecordingSchema:
    def __init__(self):
        self.calls = []

    def convert_humming(self, **kwargs):
        self.calls.append(kwargs)
        return object(), kwargs["tensors"]


@pytest.mark.parametrize(
    ("activation", "shape_n", "expected_stacks"),
    [
        (MoEActivation.SILU, 3712, [1856, 1856]),
        (MoEActivation.RELU2_NO_MUL, 1856, [1856]),
    ],
)
def test_w13_conversion_stacks_follow_activation(activation, shape_n, expected_stacks):
    pytest.importorskip("humming")
    layer = torch.nn.Module()
    layer.moe_config = SimpleNamespace(activation=activation)
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(torch.ones(1), requires_grad=False),
    )
    weight_schema = RecordingSchema()
    input_schema = RecordingSchema()

    humming_utils._convert_sublayer_to_humming(
        layer=layer,
        sublayer_name="w13",
        shape_n=shape_n,
        shape_k=2688,
        weight_schema=weight_schema,
        input_schema=input_schema,
        num_experts=128,
        param_dtype=torch.bfloat16,
    )

    assert weight_schema.calls[0]["shape_n_stacks"] == expected_stacks
    assert input_schema.calls[0]["shape_n_stacks"] == expected_stacks


@pytest.mark.parametrize(
    ("activation", "gate_up_width"),
    [
        (MoEActivation.SILU, 3712),
        (MoEActivation.RELU2_NO_MUL, 1856),
    ],
)
def test_buffer_shapes_follow_activation(activation, gate_up_width):
    pytest.importorskip("humming")
    from vllm.utils.humming import GemmType, dtypes

    layer = SimpleNamespace(
        hidden_size=2688,
        intermediate_size_per_partition=1856,
        humming_metas={
            "w13": SimpleNamespace(
                a_dtype=dtypes.bfloat16,
                c_dtype=dtypes.bfloat16,
            )
        },
    )
    experts = SimpleNamespace(
        layer=layer,
        num_experts=128,
        is_batched=lambda: False,
        humming_gemm_type=lambda: GemmType.INDEXED,
    )

    buffer_metas, _ = HummingExpertsBase.get_buffer_metas(
        experts, M=7, topk=6, activation=activation
    )

    assert buffer_metas["gate_up_output"]["shape"] == (42, gate_up_width)
    assert buffer_metas["activation_output"]["shape"] == (42, 1856)
    assert buffer_metas["quanted_down_input"]["shape"] == (42, 1856)


def test_problem_size_uses_logical_intermediate_width():
    layer = SimpleNamespace(
        intermediate_size_per_partition=1856,
        humming_metas={
            "w13": SimpleNamespace(num_experts=128, shape_n=1856, shape_k=2688),
            "w2": SimpleNamespace(num_experts=128),
        },
    )
    experts = SimpleNamespace(layer=layer, is_batched=lambda: False)

    problem_size = HummingExpertsBase.moe_problem_size(
        experts,
        a1=torch.empty(7, 2688),
        w1=torch.empty(128, 1),
        w2=torch.empty(128, 1),
        topk_ids=torch.empty(7, 6, dtype=torch.long),
    )

    assert problem_size == (128, 7, 1856, 2688, 6)
