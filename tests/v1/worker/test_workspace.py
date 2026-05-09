# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.worker import workspace
from vllm.v1.worker.workspace import WorkspaceManager


def _forward_context(mode: CUDAGraphMode) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=mode,
    )


def _workspace_data_ptr(manager: WorkspaceManager) -> int:
    (buffer,) = manager.get_simultaneous(((4,), torch.float32))
    return buffer.data_ptr()


def test_workspace_buffers_are_isolated_by_cudagraph_runtime_mode():
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=1)

    ptrs = []
    for mode in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
    ):
        with override_forward_context(_forward_context(mode)):
            ptrs.append(_workspace_data_ptr(manager))

    assert len(ptrs) == len(set(ptrs))


def test_workspace_buffer_is_reused_for_same_cudagraph_runtime_mode():
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=1)

    with override_forward_context(_forward_context(CUDAGraphMode.FULL)):
        first_ptr = _workspace_data_ptr(manager)
        second_ptr = _workspace_data_ptr(manager)

    assert first_ptr == second_ptr


def test_workspace_buffers_remain_isolated_by_ubatch(monkeypatch):
    manager = WorkspaceManager(torch.device("cpu"), num_ubatches=2)

    with override_forward_context(_forward_context(CUDAGraphMode.FULL)):
        monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
        ubatch0_ptr = _workspace_data_ptr(manager)
        monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 1)
        ubatch1_ptr = _workspace_data_ptr(manager)

    assert ubatch0_ptr != ubatch1_ptr
