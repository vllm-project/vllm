# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

# Distinctive size so a freed block is reused cleanly by an equally-sized alloc.
N = 1_000_003


def test_good_inplace_mutation(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    buf = torch.zeros(1024, device="cuda")
    src = torch.ones(1024, device="cuda")

    def clean_break():
        # In-place into a pre-existing buffer; the intermediate is freed.
        buf.copy_(src * 2.0)

    cap = BreakableCUDAGraphCapture(debug=True)
    with cap:
        x = torch.zeros(4, device="cuda")
        x.add_(1.0)
        cap.add_eager(clean_break)
        x.add_(1.0)
    assert cap.num_eager_breaks == 1


def test_good_new_alloc(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    src = torch.ones(1024, device="cuda")

    def returns_break():
        # A fresh tensor that IS returned is the explicit output, not a leak.
        return src * 2.0

    cap = BreakableCUDAGraphCapture(debug=True)
    with cap:
        x = torch.zeros(4, device="cuda")
        x.add_(1.0)
        cap.add_eager(returns_break)
        x.add_(1.0)
    assert cap.num_eager_breaks == 1


def test_non_explicit_output_raises(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    sink = []
    src = torch.ones(1024, device="cuda")

    def leaky_break():
        # Fresh alloc survives via the side channel but isn't returned.
        sink.append(src * 3.0)

    cap = BreakableCUDAGraphCapture(debug=True)
    with pytest.raises(RuntimeError, match="eager break"), cap:
        x = torch.zeros(4, device="cuda")
        x.add_(1.0)
        cap.add_eager(leaky_break)
        x.add_(1.0)


def test_non_explicit_output_reused_address_raises(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    sink = []
    holder = {"a": torch.empty(N, device="cuda")}
    torch.accelerator.synchronize()

    def reuse_break():
        # Free a pre-existing block, then reuse its address for a survivor.
        # The live-segment diff is blind to this; the event log is not.
        del holder["a"]
        sink.append(torch.empty(N, device="cuda"))

    cap = BreakableCUDAGraphCapture(debug=True)
    with pytest.raises(RuntimeError, match="eager break"), cap:
        x = torch.zeros(4, device="cuda")
        x.add_(1.0)
        cap.add_eager(reuse_break)
        x.add_(1.0)


def test_non_explicit_output_custom_op_raises(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    sink = []
    lib = torch.library.Library("bcg_dbg_test", "FRAGMENT")
    lib.define("leak(Tensor x) -> Tensor")

    def _leak_impl(x):
        # Allocation inside an opaque op is still seen at the allocator level.
        sink.append(x + 1.0)
        return x.clone()

    lib.impl("leak", _leak_impl, "CUDA")
    leak_op = torch.ops.bcg_dbg_test.leak.default
    inp = torch.ones(1024, device="cuda")

    def opaque_break():
        return leak_op(inp)

    cap = BreakableCUDAGraphCapture(debug=True)
    with pytest.raises(RuntimeError, match="eager break"), cap:
        x = torch.zeros(4, device="cuda")
        x.add_(1.0)
        cap.add_eager(opaque_break)
        x.add_(1.0)


def test_good_marked_non_explicit_output(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.compilation.breakable_cudagraph_debug import mark_bcg_output

    sink = []
    src = torch.ones(1024, device="cuda")

    def marked_break():
        out = src * 3.0
        sink.append(out)
        mark_bcg_output(out)

    cap = BreakableCUDAGraphCapture(debug=True)
    with cap:
        x = torch.zeros(4, device="cuda")
        x.add_(1.0)
        cap.add_eager(marked_break)
        x.add_(1.0)
    assert cap.num_eager_breaks == 1


def test_non_explicit_output_not_flagged_when_debug_off(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    sink = []
    src = torch.ones(1024, device="cuda")

    def leaky_break():
        sink.append(src * 3.0)

    cap = BreakableCUDAGraphCapture(debug=False)
    with cap:
        x = torch.zeros(4, device="cuda")
        x.add_(1.0)
        cap.add_eager(leaky_break)
        x.add_(1.0)
    assert cap.num_eager_breaks == 1


def test_check_non_explicit_outputs_rejects_user_memory_history(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph_debug import check_non_explicit_outputs

    def device_trace_actions():
        trace = torch._C._cuda_memorySnapshot(None).get("device_traces", [])
        return [e["action"] for e in trace[torch.cuda.current_device()]]

    torch.cuda.memory._record_memory_history(None, clear_history=True)
    try:
        torch.cuda.memory._record_memory_history(
            "all", context="alloc", stacks="python", clear_history=True
        )
        torch.empty(17, device="cuda")
        assert device_trace_actions()

        def inner():
            return torch.empty(19, device="cuda")

        with pytest.raises(RuntimeError, match="PyTorch CUDA memory history"):
            check_non_explicit_outputs(inner, (), {})

        torch.empty(23, device="cuda")
        assert "alloc" in device_trace_actions()
        assert torch._C._cuda_isHistoryEnabled()
    finally:
        torch.cuda.memory._record_memory_history(None, clear_history=True)
