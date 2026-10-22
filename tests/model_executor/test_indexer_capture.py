# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU oracles for the selections consumed by sparse attention adapters."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.indexer_capture import (
    IndexerCapturer,
    IndexerCaptureSink,
)

pytestmark = pytest.mark.cpu_test


def test_raw_capture_preserves_selection_order_and_padding():
    consumed = torch.tensor([[8, 2, -1, 4], [0, -1, -1, -1]], dtype=torch.int32)
    sink = IndexerCaptureSink(torch.empty_like(consumed))
    sink.capture(consumed)
    assert torch.equal(sink.rows, consumed)


def test_capture_snapshots_survive_shared_buffer_reuse(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.indexer_capture.current_platform",
        SimpleNamespace(device_type="cpu"),
    )

    class Source(torch.nn.Module):
        indexer_capture: IndexerCaptureSink | None = None

        def __init__(self, layer):
            super().__init__()
            self.indexer_capture_layer_id = layer
            self.indexer_capture_topk = 2

    first, second = Source(0), Source(1)
    model = torch.nn.ModuleList([first, second])
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2),
        max_concurrent_batches=2,
        model_config=SimpleNamespace(
            model="synthetic",
            revision="r1",
            code_revision=None,
            hf_config=SimpleNamespace(_commit_hash="sha"),
            hf_text_config=SimpleNamespace(to_dict=lambda: {"num_hidden_layers": 2}),
            dtype=torch.bfloat16,
            quantization=None,
        ),
    )
    capture = IndexerCapturer(model, config)
    shared = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
    assert first.indexer_capture is not None
    first.indexer_capture.capture(shared)
    shared.add_(10)
    assert second.indexer_capture is not None
    second.indexer_capture.capture(shared)
    snapshot = capture.snapshot(2)
    shared.zero_()
    first.indexer_capture.capture(shared)
    shared.add_(20)
    second.indexer_capture.capture(shared)
    next_snapshot = capture.snapshot(2)
    assert torch.equal(
        next_snapshot.rows, torch.tensor([[[0, 0], [20, 20]], [[0, 0], [20, 20]]])
    )
    with pytest.raises(AssertionError, match="not consumed"):
        capture.snapshot(2)
    capture.release_snapshot(next_snapshot.slot)
    assert capture.snapshot(2).slot == next_snapshot.slot
    assert torch.equal(
        snapshot.rows, torch.tensor([[[3, 1], [13, 11]], [[4, 2], [14, 12]]])
    )
    capture.close()
    assert first.indexer_capture is None and second.indexer_capture is None
    capture.close()
