# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the XPU post-capture collective reset (no XPU needed).

The hardware regression test is test_xpu_graph_capture_all_reduce.py; these
tests keep the hook from being dropped silently in a refactor or rebase.
"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.distributed.device_communicators import xpu_communicator
from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
from vllm.distributed.parallel_state import GraphCaptureContext, GroupCoordinator


def _communicator(world_size: int) -> XpuCommunicator:
    comm = XpuCommunicator.__new__(XpuCommunicator)
    comm.world_size = world_size
    comm.device = torch.device("cpu")
    comm.device_group = MagicMock(name="device_group")
    return comm


@pytest.mark.parametrize(
    ("world_size", "capturing", "expect_call"),
    [(2, False, True), (1, False, False), (2, True, False)],
)
def test_reset_after_graph_capture(world_size, capturing, expect_call):
    comm = _communicator(world_size)
    with (
        patch.object(xpu_communicator.dist, "all_reduce") as all_reduce,
        patch.object(torch.xpu, "is_current_stream_capturing", return_value=capturing),
    ):
        comm.reset_after_graph_capture()

    if not expect_call:
        all_reduce.assert_not_called()
        return
    all_reduce.assert_called_once()
    (tensor,), kwargs = all_reduce.call_args
    assert tensor.numel() == 1
    assert kwargs["group"] is comm.device_group


def _coordinator(device_communicator) -> GroupCoordinator:
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.device_communicator = device_communicator
    return group


def _run_graph_capture(group: GroupCoordinator, body=lambda: None) -> None:
    stream = MagicMock(name="capture_stream")
    with (
        patch.object(parallel_state.torch.cuda, "current_stream", return_value=stream),
        patch.object(parallel_state.torch.cuda, "stream", return_value=nullcontext()),
        group.graph_capture(GraphCaptureContext(stream)),
    ):
        body()


def test_graph_capture_resets_xpu_collectives_on_exit():
    comm = MagicMock(spec=XpuCommunicator)
    comm.ca_comm = None
    comm.aiter_ar_comm = None

    _run_graph_capture(_coordinator(comm))

    comm.reset_after_graph_capture.assert_called_once_with()


def test_graph_capture_skips_reset_when_capture_raises():
    comm = MagicMock(spec=XpuCommunicator)
    comm.ca_comm = None
    comm.aiter_ar_comm = None

    def fail():
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        _run_graph_capture(_coordinator(comm), fail)

    comm.reset_after_graph_capture.assert_not_called()
