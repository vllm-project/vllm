# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for CUDA checkpoint/restore functionality."""

import pytest
import torch

from vllm.device_allocator.cuda_checkpoint import (
    CudaCheckpointer,
    cuda_checkpoint_available,
)

from ..utils import create_new_process_for_each_test

pytestmark = pytest.mark.skipif(
    not cuda_checkpoint_available,
    reason="CUDA checkpoint APIs not available (requires NVIDIA driver >= 570)",
)


@create_new_process_for_each_test()
def test_basic_suspend_resume():
    """Test that tensors are preserved across a suspend/resume cycle."""
    checkpointer = CudaCheckpointer.get_instance()

    # Create a tensor with known values
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    expected = x.clone()

    # Suspend
    handle = checkpointer.suspend()
    assert checkpointer.is_suspended
    assert handle is not None

    # Resume
    checkpointer.resume(handle)
    assert not checkpointer.is_suspended

    # Verify tensor is preserved
    assert torch.equal(x, expected), "Tensor data changed after suspend/resume"


@create_new_process_for_each_test()
def test_cuda_graph_preservation():
    """Test that a CUDA graph survives a checkpoint cycle."""
    checkpointer = CudaCheckpointer.get_instance()

    # Create and capture a simple CUDA graph
    x = torch.randn(256, device="cuda")
    y = torch.empty_like(x)

    # Warmup
    y.copy_(x * 2.0 + 1.0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y.copy_(x * 2.0 + 1.0)

    # Run the graph to verify it works before suspend
    x.fill_(3.0)
    graph.replay()
    expected_val = 3.0 * 2.0 + 1.0
    assert torch.allclose(y, torch.full_like(y, expected_val))

    # Suspend and resume
    handle = checkpointer.suspend()
    checkpointer.resume(handle)

    # Replay the graph again after resume
    x.fill_(5.0)
    graph.replay()
    expected_val = 5.0 * 2.0 + 1.0
    assert torch.allclose(y, torch.full_like(y, expected_val)), (
        "CUDA graph output incorrect after suspend/resume"
    )


@create_new_process_for_each_test()
def test_error_double_suspend():
    """Test that suspending while already suspended raises an error."""
    checkpointer = CudaCheckpointer.get_instance()

    checkpointer.suspend()
    assert checkpointer.is_suspended

    with pytest.raises(RuntimeError, match="already suspended"):
        checkpointer.suspend()

    # Clean up
    checkpointer.resume()


@create_new_process_for_each_test()
def test_resume_without_suspend():
    """Test that resuming without a prior suspend raises an error."""
    checkpointer = CudaCheckpointer.get_instance()

    with pytest.raises(RuntimeError, match="not suspended"):
        checkpointer.resume()


@create_new_process_for_each_test()
def test_suspend_resume_implicit_handle():
    """Test suspend/resume using the implicitly stored handle."""
    checkpointer = CudaCheckpointer.get_instance()

    x = torch.ones(64, device="cuda")

    checkpointer.suspend()
    assert checkpointer.is_suspended

    # Resume without passing handle explicitly
    checkpointer.resume()
    assert not checkpointer.is_suspended

    assert torch.equal(x, torch.ones(64, device="cuda"))
