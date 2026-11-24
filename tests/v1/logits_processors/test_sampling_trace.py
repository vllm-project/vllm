# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU behavioral checks for the example's public LogitsProcessor interface."""

import numpy as np
import pytest
import torch

from examples.features.logits_processor.sampling_trace import SamplingTraceProcessor
from vllm.config import DeviceConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext


def make_processor(**options):
    config = VllmConfig(
        additional_config={"sampling_trace": options},
        device_config=DeviceConfig(device="cpu"),
    )
    return SamplingTraceProcessor(config, None)


def make_context(slots=(0, 1), row_slots=(0, 1), positions=(7, 8)):
    return LogitsContext(
        expanded_idx_mapping=torch.tensor(row_slots),
        idx_mapping=torch.tensor(slots),
        idx_mapping_np=np.array(slots),
        expanded_local_pos=torch.arange(len(row_slots)),
        input_ids=torch.arange(len(row_slots)),
        pos=torch.tensor(positions),
        seq_lens_upper_bound_np=np.array([9] * len(slots)),
    )


def add_request(processor, index, label=None):
    return processor.add_request(
        index, SamplingParams(extra_args={"sampling_trace_label": label})
    )


def test_returns_original_logits_and_preserves_gradients():
    processor = make_processor()
    value = torch.randn(2, 5, requires_grad=True)
    before = value.detach().clone()
    result = processor.apply(value, make_context())
    assert result is value
    torch.testing.assert_close(value, before, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))
    (snapshot,) = processor.take_snapshots()
    assert not snapshot.logits.requires_grad
    assert snapshot.logits.device == value.device


def test_snapshot_owns_logits_and_context_storage():
    processor = make_processor()
    value = torch.randn(2, 5)
    context = make_context()
    expected = value.clone()
    processor.apply(value, context)
    value.zero_()
    context.pos.zero_()
    context.idx_mapping_np[:] = 9
    (snapshot,) = processor.take_snapshots()
    torch.testing.assert_close(snapshot.logits, expected, rtol=0, atol=0)
    assert snapshot.positions.tolist() == [7, 8]
    assert snapshot.batch_slots.tolist() == [0, 1]


def test_tracks_actual_reordered_and_speculative_rows():
    processor = make_processor()
    add_request(processor, 0, "first")
    add_request(processor, 1, "second")
    context = make_context(slots=(1, 0), row_slots=(1, 1, 0), positions=(9, 10, 7))
    value = torch.randn(3, 5)
    processor.apply(value, context)
    (snapshot,) = processor.take_snapshots()
    assert snapshot.batch_slots.tolist() == [1, 0]
    assert snapshot.request_slots.tolist() == [1, 1, 0]
    assert snapshot.positions.tolist() == [9, 10, 7]
    assert snapshot.labels == {1: "second", 0: "first"}


def test_request_slot_reuse_clears_old_label():
    processor = make_processor()
    add_request(processor, 0, "old")
    add_request(processor, 0)
    processor.apply(torch.randn(2, 5), make_context())
    (snapshot,) = processor.take_snapshots()
    assert snapshot.labels[0] is None


def test_step_budget_and_drain_reset():
    processor = make_processor(max_steps=1)
    value = torch.randn(2, 5)
    for _ in range(3):
        assert processor.apply(value, make_context()) is value
    assert len(processor.take_snapshots()) == 1
    assert processor.take_snapshots() == ()
    processor.apply(value, make_context())
    assert len(processor.take_snapshots()) == 1


def test_oversized_frame_is_skipped_before_clone(monkeypatch):
    processor = make_processor(max_bytes=1)
    value = torch.randn(2, 5)

    def forbidden(*args, **kwargs):
        raise AssertionError("storage budget must be checked before cloning")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    assert processor.apply(value, make_context()) is value
    assert processor.take_snapshots() == ()
    assert processor.dropped_steps == 1


@pytest.mark.parametrize("options", [{"max_steps": 0}, {"max_bytes": 0}])
def test_disabled_capture_does_not_clone(options, monkeypatch):
    processor = make_processor(**options)
    assert not add_request(processor, 0)
    value = torch.randn(2, 5)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled capture must not clone")

    monkeypatch.setattr(torch.Tensor, "clone", forbidden)
    assert processor.apply(value, make_context()) is value
    assert processor.take_snapshots() == ()


@pytest.mark.parametrize(
    "options",
    [
        {"max_steps": -1},
        {"max_steps": True},
        {"max_steps": 1.5},
        {"max_bytes": -1},
        {"max_bytes": "64"},
    ],
)
def test_rejects_invalid_budgets(options):
    with pytest.raises(ValueError):
        make_processor(**options)


@pytest.mark.parametrize("label", [129 * "x", 1, [], {}])
def test_rejects_invalid_labels(label):
    processor = make_processor()
    with pytest.raises(ValueError):
        add_request(processor, 0, label)


def test_preserves_masked_and_nonfinite_values():
    processor = make_processor()
    value = torch.tensor([[float("-inf"), float("nan"), 1.0]])
    before = value.view(torch.int32).clone()
    context = make_context(slots=(0,), row_slots=(0,), positions=(4,))
    assert processor.apply(value, context) is value
    (snapshot,) = processor.take_snapshots()
    assert torch.equal(snapshot.logits.view(torch.int32), before)


def test_capture_never_reads_tensor_values_on_host(monkeypatch):
    processor = make_processor()
    value = torch.randn(2, 5)
    context = make_context()

    def forbidden(*args, **kwargs):
        raise AssertionError("capture must not read a tensor on the host")

    for name in ("cpu", "item", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    assert processor.apply(value, context) is value
    assert len(processor.take_snapshots()) == 1


def test_byte_budget_accumulates_across_steps_and_resets_on_drain():
    context = make_context()
    value = torch.randn(2, 5)
    tensors = [
        value,
        context.expanded_idx_mapping,
        context.expanded_local_pos,
        context.input_ids,
        context.pos,
    ]
    frame_bytes = sum(t.numel() * t.element_size() for t in tensors)
    frame_bytes += (
        context.idx_mapping_np.nbytes + context.seq_lens_upper_bound_np.nbytes
    )
    processor = make_processor(max_bytes=frame_bytes * 2)
    for _ in range(3):
        processor.apply(value, context)
    assert len(processor.take_snapshots()) == 2
    assert processor.dropped_steps == 1
    processor.apply(value, context)
    assert len(processor.take_snapshots()) == 1
