# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
)


def test_short_conv_state_shape_widens_for_spec_decode():
    # Spec decode rolls the conv window back to the last accepted token, so
    # the cached state must hold conv_kernel - 1 + num_spec entries
    # (mirrors mamba2_state_shape).
    (base_shape,) = MambaStateShapeCalculator.short_conv_state_shape(
        tp_world_size=1, intermediate_size=64, conv_kernel=3
    )
    (spec_shape,) = MambaStateShapeCalculator.short_conv_state_shape(
        tp_world_size=1, intermediate_size=64, conv_kernel=3, num_spec=4
    )
    assert sorted(base_shape) == [2, 64]
    assert sorted(spec_shape) == [6, 64]
