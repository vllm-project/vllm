# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen2_moe import (
    _ceil_to_multiple,
    _dense_mlp_padded_intermediate_multiple,
)


def test_dense_mlp_tp3_padding_aligns_awq_group_size():
    intermediate_size = 17408
    tp_size = 3

    padded = _ceil_to_multiple(
        intermediate_size,
        _dense_mlp_padded_intermediate_multiple(tp_size),
    )

    assert padded == 17472
    assert padded % tp_size == 0
    assert (padded // tp_size) % 32 == 0
