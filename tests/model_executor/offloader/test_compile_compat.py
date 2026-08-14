# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    get_layer_from_name,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.utils.torch_utils import _USE_LAYERNAME


def test_fused_moe_resolves_layer_from_no_compile_context():
    layer = MagicMock(spec=MoERunnerInterface)
    context = ForwardContext(
        no_compile_layers={"model.layers.0.mlp.experts": layer},
        attn_metadata={},
        slot_mapping={},
    )

    with override_forward_context(context):
        assert get_layer_from_name("model.layers.0.mlp.experts") is layer


@pytest.mark.skipif(
    _USE_LAYERNAME,
    reason=(
        "Fast cold-start 'from_forward_context' resolution path is only "
        "active when _USE_LAYERNAME is False (torch < 2.11). On torch >= 2.11 "
        "the LayerName-based path is used instead and this protocol no longer "
        "applies."
    ),
)
def test_fused_moe_fast_cold_start_resolves_ordered_no_compile_layers():
    first_layer = object()
    second_layer = object()
    context = ForwardContext(
        no_compile_layers={
            "model.layers.0.mlp.experts": first_layer,
            "model.layers.1.mlp.experts": second_layer,
        },
        attn_metadata={},
        slot_mapping={},
        all_moe_layers=[
            "model.layers.0.mlp.experts",
            "model.layers.1.mlp.experts",
        ],
    )

    with override_forward_context(context):
        assert get_layer_from_name("from_forward_context") is first_layer
        assert get_layer_from_name("from_forward_context") is second_layer

    assert context.moe_layer_index == 2
