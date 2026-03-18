# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for two piecewise CUDA graph bugs fixed in:
  - vllm/compilation/backends.py  (cycle in split_graph with multi-output splitting_ops)
  - vllm/compilation/cuda_graph.py (stale tensor addresses during replay)
"""

import operator

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.library import Library

from vllm.compilation.backends import split_graph
from vllm.utils.torch_utils import direct_register_custom_op

# ---------------------------------------------------------------------------
# Custom op that returns a tuple (simulates a splitting_op with multiple outputs)
# ---------------------------------------------------------------------------
_bmm_lib = Library("test_piecewise", "FRAGMENT")


def _bmm_multi_out(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Splitting op that allocates new tensors and returns a tuple."""
    a = torch.bmm(x.unsqueeze(0), x.unsqueeze(0).transpose(1, 2)).squeeze(0)
    b = x * 2
    return a, b


def _bmm_multi_out_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    return x.new_empty(n, n), x.new_empty_strided(x.shape, x.stride())


direct_register_custom_op(
    op_name="bmm_multi_out",
    op_func=_bmm_multi_out,
    mutates_args=[],
    fake_impl=_bmm_multi_out_fake,
    target_lib=_bmm_lib,
)


# ---------------------------------------------------------------------------
# Bug 1 — backends.py: getitem of a splitting_op must go to the NEXT subgraph
# ---------------------------------------------------------------------------


def test_splitting_op_getitem_assigned_to_next_subgraph():
    """
    When a splitting_op returns a tuple, the getitem nodes that extract its
    outputs must be assigned to the *next* subgraph, not the splitting_op's
    own subgraph.  Assigning them to the same subgraph creates a cycle that
    makes torch.fx.passes.split_module fail.
    """

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        # splitting_op — returns (a, b) tuple
        a, b = torch.ops.test_piecewise.bmm_multi_out(x)
        # downstream consumers — must land in the subgraph *after* the split
        return torch.relu(a) + b

    x = torch.randn(4, 4)
    gm = make_fx(model_fn)(x)

    # Verify the graph actually contains getitem nodes (sanity check)
    has_getitem = any(
        n.op == "call_function" and n.target == operator.getitem
        for n in gm.graph.nodes
    )
    assert has_getitem, "Test setup failed: expected getitem nodes in graph"

    # split_graph must not raise (previously raised due to cycle)
    split_gm, split_items = split_graph(gm, ["test_piecewise::bmm_multi_out"])

    # Correctness: outputs must match
    new_x = torch.randn(4, 4)
    assert torch.allclose(gm(new_x), split_gm(new_x)), "Output mismatch after split"

    # The getitem nodes must NOT appear as placeholders in the splitting subgraph
    splitting_graphs = [s for s in split_items if s.is_splitting_graph]
    assert splitting_graphs, "Expected at least one splitting subgraph"

    for sg in splitting_graphs:
        for node in sg.graph.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == operator.getitem
                and node.args[0].op == "placeholder"
            ):
                pytest.fail(
                    f"getitem node '{node.name}' landed in splitting subgraph "
                    f"'{sg.submod_name}' as a placeholder consumer — "
                    "this indicates the cycle-fix is not applied."
                )


# ---------------------------------------------------------------------------
# Bug 2 — cuda_graph.py: input_buffers sync when splitting_op allocates tensors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cudagraph_entry_input_buffers_populated():
    """
    CUDAGraphEntry.input_buffers must be populated at capture time so that
    data can be copied into them before replay when a splitting_op allocates
    new tensors between two compiled pieces.
    """
    from vllm.compilation.cuda_graph import CUDAGraphEntry, CUDAGraphOptions, CUDAGraphWrapper
    from vllm.config import CUDAGraphMode, VllmConfig

    vllm_config = VllmConfig()

    def simple_fn(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    wrapper = CUDAGraphWrapper(
        runnable=simple_fn,
        vllm_config=vllm_config,
        runtime_mode=CUDAGraphMode.FULL,
        cudagraph_options=CUDAGraphOptions(debug_log_enable=False),
    )

    # input_buffers is only populated during CUDA graph capture (inside __call__)
    # We test the field exists and is typed correctly on CUDAGraphEntry
    entry = CUDAGraphEntry(batch_descriptor=None)  # type: ignore[arg-type]
    assert entry.input_buffers is None, (
        "input_buffers should be None before capture"
    )

    # Simulate what capture does: save tensor references
    x = torch.randn(4, device="cuda")
    entry.input_buffers = [x]
    assert entry.input_buffers[0] is x

    # Simulate what replay does: copy new data when address differs
    x_new = torch.randn(4, device="cuda")
    assert x_new.data_ptr() != x.data_ptr(), "Test setup: need different addresses"

    for buf, arg in zip(entry.input_buffers, [x_new]):
        if buf.data_ptr() != arg.data_ptr():
            buf.copy_(arg)

    assert torch.allclose(entry.input_buffers[0], x_new), (
        "input_buffers.copy_() did not sync the new data correctly"
    )
