# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.distributed import communication_op, parallel_state

pytestmark = pytest.mark.skip_global_cleanup


def test_fixed_order_all_reduce_dispatches_custom_op(monkeypatch):
    monkeypatch.setenv("VLLM_TP_FIXED_ORDER_ALLREDUCE", "1")
    group = SimpleNamespace(
        world_size=3,
        unique_name="tp:test",
        all_reduce=Mock(),
    )
    communication_op._fixed_order_notice_printed = True
    with (
        patch.object(communication_op, "get_tp_group", return_value=group),
        patch.object(
            torch.ops.vllm,
            "fixed_order_all_reduce_",
        ) as fixed_order,
    ):
        input_ = torch.tensor([0.0, 0.0])
        result = communication_op.tensor_model_parallel_all_reduce(input_)

    assert result is input_
    fixed_order.assert_called_once()
    assert fixed_order.call_args.kwargs == {"group_name": "tp:test"}
    group.all_reduce.assert_not_called()


def test_fixed_order_all_reduce_gathers_then_sums_by_rank():
    rank_values = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([10.0, 20.0]),
        torch.tensor([100.0, 200.0]),
    )
    gathered_inputs = []

    def gather(input_, dim):
        gathered_inputs.append((input_.clone(), dim))
        return torch.stack(rank_values)

    group = SimpleNamespace(
        world_size=3,
        _all_gather_out_place=Mock(side_effect=gather),
    )

    with patch.dict(parallel_state._groups, {"tp:test": lambda: group}):
        result = torch.tensor([0.0, 0.0])
        parallel_state.fixed_order_all_reduce_(result, "tp:test")

    torch.testing.assert_close(result, torch.tensor([111.0, 222.0]))
    (gathered_input, dim) = gathered_inputs[0]
    torch.testing.assert_close(gathered_input, torch.tensor([[0.0, 0.0]]))
    assert dim == 0


def test_fixed_order_all_reduce_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_TP_FIXED_ORDER_ALLREDUCE", raising=False)
    expected = torch.tensor([3.0])
    group = SimpleNamespace(
        world_size=2,
        unique_name="tp:test",
        all_reduce=Mock(return_value=expected),
    )

    with patch.object(communication_op, "get_tp_group", return_value=group):
        result = communication_op.tensor_model_parallel_all_reduce(torch.tensor([1.0]))

    assert result is expected
    group.all_reduce.assert_called_once()
