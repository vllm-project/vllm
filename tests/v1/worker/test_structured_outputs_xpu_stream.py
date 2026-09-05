# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream selection in StructuredOutputsWorker.apply_grammar_bitmask.

SYCL command graphs reject cross-queue event dependencies, so under XPU
graphs the grammar-bitmask H2D copies must stay on the current stream
instead of joining a private copy stream via ``wait_stream`` (see
``vllm/v1/worker/gpu/pp_utils.py`` for the same XPU limitation). These tests
run on CPU tensors only, with ``torch.cuda.{Stream,stream,current_stream}``
monkeypatched to lightweight fakes, so no accelerator is required or used.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import structured_outputs as so_module
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker


class _FakeStream:
    """Stand-in for ``torch.cuda.Stream``: records ``wait_stream`` calls."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.wait_stream_calls: list[_FakeStream] = []

    def wait_stream(self, other: "_FakeStream") -> None:
        self.wait_stream_calls.append(other)

    def __repr__(self) -> str:
        return f"_FakeStream({self.name!r})"


class _FakeStreamContext:
    """Stand-in for the context manager ``torch.cuda.stream(...)`` returns."""

    def __init__(self, stream: _FakeStream) -> None:
        self.stream = stream
        self.enter_count = 0

    def __enter__(self) -> _FakeStream:
        self.enter_count += 1
        return self.stream

    def __exit__(self, *exc_info: Any) -> None:
        return None


class _FakeKernel:
    """Stand-in for the ``@triton.jit`` kernel: records ``kernel[grid](...)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple, dict]] = []

    def __getitem__(self, grid: Any) -> Any:
        def _launch(*args: Any, **kwargs: Any) -> None:
            self.calls.append((grid, args, kwargs))

        return _launch


def _install_fakes(monkeypatch: pytest.MonkeyPatch, is_xpu: bool) -> dict[str, Any]:
    """Patch everything apply_grammar_bitmask touches with device-free fakes."""
    monkeypatch.setattr(
        so_module, "current_platform", SimpleNamespace(is_xpu=lambda: is_xpu)
    )
    # pin_memory() on a real tensor requires an initialized accelerator; keep
    # this test's copies plain CPU-to-CPU regardless of the host's PIN_MEMORY.
    monkeypatch.setattr(so_module, "PIN_MEMORY", False)

    current_stream = _FakeStream("current")
    stream_ctx_calls: list[_FakeStreamContext] = []

    def fake_stream_ctor(*args: Any, **kwargs: Any) -> _FakeStream:
        return _FakeStream("copy")

    def fake_stream_ctx(stream: _FakeStream) -> _FakeStreamContext:
        ctx = _FakeStreamContext(stream)
        stream_ctx_calls.append(ctx)
        return ctx

    def fake_current_stream(*args: Any, **kwargs: Any) -> _FakeStream:
        return current_stream

    monkeypatch.setattr(torch.cuda, "Stream", fake_stream_ctor)
    monkeypatch.setattr(torch.cuda, "stream", fake_stream_ctx)
    monkeypatch.setattr(torch.cuda, "current_stream", fake_current_stream)

    copy_calls: list[Any] = []

    def fake_async_copy_to_gpu(
        x: Any, out: torch.Tensor | None = None, device: Any = None
    ) -> torch.Tensor:
        copy_calls.append(x)
        assert out is not None
        return out

    monkeypatch.setattr(so_module, "async_copy_to_gpu", fake_async_copy_to_gpu)

    kernel = _FakeKernel()
    monkeypatch.setattr(so_module, "_apply_grammar_bitmask_kernel", kernel)

    return {
        "current_stream": current_stream,
        "stream_ctx_calls": stream_ctx_calls,
        "copy_calls": copy_calls,
        "kernel": kernel,
    }


def _run_apply_grammar_bitmask(
    monkeypatch: pytest.MonkeyPatch, is_xpu: bool
) -> tuple[StructuredOutputsWorker, dict[str, Any]]:
    fakes = _install_fakes(monkeypatch, is_xpu)

    worker = StructuredOutputsWorker(
        max_num_logits=4,
        vocab_size=32,
        device=torch.device("cpu"),
        mask_stride=4,
        num_bonus_tokens=0,
    )

    # Two requests, one logit position each, both selected for grammar masking.
    input_batch = SimpleNamespace(
        req_ids=["a", "b"],
        cu_num_logits_np=np.array([0, 1, 2], dtype=np.int64),
        num_draft_tokens_per_req=None,
        cu_num_logits=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    logits = torch.zeros(2, 32)
    grammar_bitmask = np.zeros((2, 1), dtype=np.int32)

    worker.apply_grammar_bitmask(logits, input_batch, ["a", "b"], grammar_bitmask)
    return worker, fakes


def test_non_xpu_uses_copy_stream_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, fakes = _run_apply_grammar_bitmask(monkeypatch, is_xpu=False)

    assert worker.use_copy_stream is True

    # torch.cuda.stream(self.copy_stream) is constructed once and entered by
    # both `with copy_ctx:` blocks (bitmask copy, then mapping copy).
    assert len(fakes["stream_ctx_calls"]) == 1
    ctx = fakes["stream_ctx_calls"][0]
    assert ctx.stream is worker.copy_stream
    assert ctx.enter_count == 2

    # Both directions of the cross-stream join happen.
    current_stream = fakes["current_stream"]
    assert current_stream.wait_stream_calls == [worker.copy_stream]
    assert worker.copy_stream.wait_stream_calls == [current_stream]

    assert len(fakes["copy_calls"]) == 1
    assert len(fakes["kernel"].calls) == 1


def test_xpu_skips_copy_stream_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, fakes = _run_apply_grammar_bitmask(monkeypatch, is_xpu=True)

    assert worker.use_copy_stream is False

    # No cross-queue context is ever opened on XPU.
    assert fakes["stream_ctx_calls"] == []

    # No cross-stream event dependency is created either.
    current_stream = fakes["current_stream"]
    assert current_stream.wait_stream_calls == []
    assert worker.copy_stream.wait_stream_calls == []

    # The copies and the kernel launch still happen, just on the current stream.
    assert len(fakes["copy_calls"]) == 1
    assert len(fakes["kernel"].calls) == 1
