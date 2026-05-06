# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for driver-side capture consumers via ``LLM``.

This test requires CUDA and a full vLLM install with a small model
(e.g. ``facebook/opt-125m``).  It is unconditionally skipped in CI
and unit-test runs; use it locally to verify the full pipeline.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pytest
import torch

from vllm.v1.capture.consumer import CaptureConsumer
from vllm.v1.capture.types import CaptureKey


class _E2ERecordingConsumer(CaptureConsumer):
    """A trivial driver consumer that records calls for assertion."""

    location: ClassVar[Literal["worker", "driver"]] = "driver"

    def __init__(self) -> None:
        self.captures: list[tuple[CaptureKey, torch.Tensor, dict[str, Any]]] = []

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        self.captures.append((key, tensor.clone(), dict(sidecar)))


@pytest.mark.skip(reason="requires CUDA and full vLLM install")
def test_llm_with_driver_capture_consumer():
    """Verify that ``LLM(capture_consumers=[instance])`` properly
    wires a driver-side consumer so that ``on_capture`` fires for
    each request.

    To run locally::

        .venv/bin/python -m pytest tests/v1/capture/test_driver_consumer_e2e.py \
            -v -k test_llm_with_driver_capture_consumer --no-header
    """
    from vllm import LLM, SamplingParams

    consumer = _E2ERecordingConsumer()
    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        capture_consumers=[consumer],
    )

    outputs = llm.generate(
        ["Hello world"],
        SamplingParams(max_tokens=8),
    )

    assert len(outputs) == 1
    # The consumer should have received at least one on_capture call.
    assert len(consumer.captures) > 0

    for captured_key, tensor, _sidecar in consumer.captures:
        assert tensor.ndim == 2
        assert tensor.shape[1] > 0  # hidden_size
