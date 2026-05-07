# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase D multi-consumer runner tests.

Verify that when two consumers are configured side-by-side (filesystem +
logging in the primary case, plus an in-test fake consumer to cover
``_BatchedAdapter``) the runner-side flow drives both through the
manager and surfaces a nested
``{consumer_name: CaptureResult}`` dict on finalize — the shape the
scheduler threads into ``ModelRunnerOutput.capture_results``.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, ClassVar

import pytest
import torch

from vllm.v1.capture.consumer import CaptureConsumer, _BatchedAdapter
from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.manager import CaptureManager
from vllm.v1.capture.plan import CaptureBatchView
from vllm.v1.capture.types import (
    CaptureKey,
    CaptureSpec,
    VllmInternalRequestId,
)


class _RecordingConsumer(CaptureConsumer):
    """Minimal CaptureConsumer that records every ``on_capture`` call."""

    location: ClassVar[str] = "worker"

    def __init__(self, vllm_config: Any, params: dict[str, Any]) -> None:
        super().__init__(vllm_config, params)
        self._hooks = params["hooks"]
        self._positions = params.get("positions", "last_prompt")
        self.captured: list[tuple[CaptureKey, tuple[int, ...]]] = []

    def global_capture_spec(self) -> CaptureSpec:
        return CaptureSpec(hooks=self._hooks, positions=self._positions)

    def on_capture(
        self,
        key: CaptureKey,
        tensor: torch.Tensor,
        sidecar: dict[str, Any],
    ) -> None:
        self.captured.append((key, tuple(tensor.shape)))


class _FakeVllmConfig:
    def __init__(self) -> None:
        self.capture_consumers_config = None


def _wait_for_filesystem_result(
    consumer: FilesystemConsumer,
    key: tuple[str, int, str],
    *,
    timeout: float = 5.0,
) -> None:
    capture_key = (VllmInternalRequestId(key[0]), key[1], key[2])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = consumer.get_result(capture_key)
        if result is not None and result.status in ("ok", "error", "partial_error"):
            return
        time.sleep(0.005)
    pytest.fail(f"timeout waiting for filesystem key {key}")


def test_two_consumers_both_see_captures(tmp_path: pathlib.Path) -> None:
    """Drive one request through a filesystem + a recording consumer.
    Finalize must yield a terminal ``CaptureResult`` for both.
    """
    # Recording consumer wrapped in the standard batched adapter — this
    # is what ``build_consumers`` does for ``CaptureConsumer`` subclasses.
    recording = _RecordingConsumer(
        _FakeVllmConfig(),
        params={"hooks": {"post_mlp": [1]}, "positions": "last_prompt"},
    )
    recording_sink = _BatchedAdapter(recording)

    # Filesystem consumer is a direct CaptureSink.
    fs_consumer = FilesystemConsumer(
        _FakeVllmConfig(),
        params={"root": str(tmp_path), "writer_threads": 1},
    )

    mgr = CaptureManager(
        consumers=(fs_consumer, recording_sink),
        # Filesystem has no global spec (per-request only); recording
        # has a global spec.
        consumer_specs=(None, recording.global_capture_spec()),
        num_hidden_layers=4,
        hidden_size=8,
        model_dtype=torch.float32,
        device="cpu",
    )

    # Name-to-index table the runner maintains.  Two consumer entries.
    name_to_index = {"filesystem": 0, "recording": 1}
    index_to_name = {v: k for k, v in name_to_index.items()}

    req_id = "req-multi-1"
    fs_client_spec = CaptureSpec(
        hooks={"post_mlp": [1]},
        positions="last_prompt",
    )

    # Register: filesystem gets a client spec; recording's global spec
    # auto-applies.
    mgr.register_request(
        req_id,
        client_specs={0: fs_client_spec},
        num_prompt_tokens=4,
        sidecar_fields={
            "tag_slug": "multi",
            "request_id_slug": req_id,
            "vllm_internal_request_id": req_id,
        },
    )

    # Fake forward step.
    batch_view = CaptureBatchView(
        req_ids=[req_id],
        num_prompt_tokens=[4],
        num_computed_tokens=[0],
        num_scheduled_tokens=[4],
        token_offsets=[0],
    )
    plan = mgr.build_step_plan(batch_view)
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    mgr.on_hook(1, "post_mlp", hidden)
    mgr.dispatch_step_captures(plan)

    # Finalize — indexed by consumer index.
    indexed = mgr.finalize_request(req_id)
    # Both consumers must have produced a terminal result.
    assert set(indexed.keys()) == {0, 1}

    # Translate to the nested name-keyed shape the runner stashes on
    # ``ModelRunnerOutput.capture_results``.
    named_results = {index_to_name[idx]: result for idx, result in indexed.items()}
    assert set(named_results.keys()) == {"filesystem", "recording"}
    assert named_results["recording"].status in ("ok", "error", "partial_error")

    # Give filesystem writer time to flush before asserting the on-disk
    # result status.
    _wait_for_filesystem_result(fs_consumer, (req_id, 1, "post_mlp"))

    # Recording consumer received the capture via ``on_capture``.
    assert len(recording.captured) == 1
    rec_key, rec_shape = recording.captured[0]
    assert rec_key == (VllmInternalRequestId(req_id), 1, "post_mlp")
    # "last_prompt" at num_prompt_tokens=4 → one row, hidden_size=8.
    assert rec_shape == (1, 8)

    fs_consumer.shutdown()


def test_one_consumer_errors_other_still_finalizes(tmp_path: pathlib.Path) -> None:
    """If one consumer's ``on_capture`` raises, the other still finalizes
    with an ``ok`` status — invariant 9 (consumer isolation).
    """

    class _FailingConsumer(CaptureConsumer):
        location: ClassVar[str] = "worker"

        def global_capture_spec(self) -> CaptureSpec:
            return CaptureSpec(
                hooks={"post_mlp": [0]},
                positions="last_prompt",
            )

        def on_capture(self, key, tensor, sidecar):
            raise RuntimeError("boom")

    failing = _FailingConsumer(_FakeVllmConfig(), params={})
    recording = _RecordingConsumer(
        _FakeVllmConfig(),
        params={"hooks": {"post_mlp": [0]}, "positions": "last_prompt"},
    )
    failing_sink = _BatchedAdapter(failing)
    recording_sink = _BatchedAdapter(recording)

    mgr = CaptureManager(
        consumers=(failing_sink, recording_sink),
        consumer_specs=(
            failing.global_capture_spec(),
            recording.global_capture_spec(),
        ),
        num_hidden_layers=2,
        hidden_size=4,
        model_dtype=torch.float32,
        device="cpu",
    )

    mgr.register_request(
        "req-isolated",
        client_specs=None,
        num_prompt_tokens=2,
    )

    batch_view = CaptureBatchView(
        req_ids=["req-isolated"],
        num_prompt_tokens=[2],
        num_computed_tokens=[0],
        num_scheduled_tokens=[2],
        token_offsets=[0],
    )
    plan = mgr.build_step_plan(batch_view)
    hidden = torch.zeros((2, 4), dtype=torch.float32)
    mgr.on_hook(0, "post_mlp", hidden)
    mgr.dispatch_step_captures(plan)

    indexed = mgr.finalize_request("req-isolated")
    assert set(indexed.keys()) == {0, 1}

    # Consumer 0 raised → status == "error".
    # Consumer 1 succeeded → status == "ok".
    assert indexed[0].status == "error"
    assert indexed[1].status == "ok"
