# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm.distributed.layer_parallel import (
    LayerParallelPlan,
    LayerType,
    ParallelAxis,
    ParallelGroupType,
    clear_layer_parallel_config,
    get_layer_parallel_config,
    init_layer_parallel_config,
)


@pytest.fixture(autouse=True)
def _reset_layer_parallel_config():
    clear_layer_parallel_config()
    yield
    clear_layer_parallel_config()


def _axis(
    world_size: int,
    rank: int,
    group: ParallelGroupType = ParallelGroupType.TENSOR,
) -> ParallelAxis:
    return ParallelAxis(world_size=world_size, rank=rank, group=group)


def test_default_layer_uses_default_plan():
    tensor_axis = _axis(8, 5)
    default_plan = LayerParallelPlan(input=tensor_axis, output=tensor_axis)
    init_layer_parallel_config(default_plan)

    assert get_layer_parallel_config() == default_plan
    assert get_layer_parallel_config(LayerType.ATTENTION) == default_plan


def test_layer_override_can_reshard_between_input_and_output():
    tensor_axis = _axis(8, 5)
    attention_axis = _axis(2, 1, ParallelGroupType.ATTENTION_TENSOR)
    default_plan = LayerParallelPlan(input=tensor_axis, output=tensor_axis)
    attention_plan = LayerParallelPlan(
        input=attention_axis,
        output=tensor_axis,
    )
    init_layer_parallel_config(
        default_plan,
        {LayerType.ATTENTION: attention_plan},
    )

    assert get_layer_parallel_config() == default_plan
    assert get_layer_parallel_config(LayerType.ATTENTION) == attention_plan
    assert attention_plan.reshards_output
    assert attention_plan.get_output_size(input_size=16) == 4


def test_plan_rejects_non_divisible_output_size():
    plan = LayerParallelPlan(
        input=_axis(2, 0, ParallelGroupType.ATTENTION_TENSOR),
        output=_axis(8, 0),
    )

    with pytest.raises(ValueError, match="Global input size"):
        plan.get_output_size(input_size=3)


@pytest.mark.parametrize(
    "world_size, rank",
    [
        (0, 0),
        (2, -1),
        (2, 2),
    ],
)
def test_axis_rejects_invalid_size_or_rank(world_size: int, rank: int):
    with pytest.raises(ValueError):
        _axis(world_size, rank)


def test_plan_is_immutable():
    axis = _axis(4, 2)
    plan = LayerParallelPlan(input=axis, output=axis)

    with pytest.raises(FrozenInstanceError):
        plan.input = _axis(2, 0)
