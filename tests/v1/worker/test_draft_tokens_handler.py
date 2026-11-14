# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


def _input_batch(
    req_ids: list[str], structured: bool, device: torch.device | None = None
) -> InputBatch:
    num_reqs = len(req_ids)
    buffers = InputBuffers(num_reqs, num_reqs, device or torch.device("cuda"))
    batch = InputBatch.make_dummy(num_reqs, num_reqs, buffers, is_padding=False)
    return replace(batch, req_ids=req_ids, has_structured_output_reqs=structured)


def _set_draft_toks(
    handler: DraftTokensHandler,
    req_ids: list[str],
    tokens: list[list[int]],
    structured: bool,
) -> None:
    batch = _input_batch(req_ids=req_ids, structured=structured)
    handler.set_draft_tokens(batch, torch.tensor(tokens, device="cuda"))


def _get_draft_toks(handler: DraftTokensHandler) -> dict[str, list[int]]:
    drafts = handler.get_draft_tokens()
    assert drafts is not None
    return dict(zip(drafts.req_ids, drafts.draft_token_ids, strict=True))


@pytest.fixture
def cuda_copies(monkeypatch):
    """Expose host copy buffers only when their CUDA event completes."""
    from vllm.v1.worker.gpu.spec_decode import utils

    class Copies:
        def __init__(self):
            self.pending: list[tuple[np.ndarray, np.ndarray]] = []
            self.ready_through: int | None = None
            self.waited: list[int] = []

        def to_numpy(self, tensor):
            source = tensor.numpy().copy()
            destination = np.full_like(source, -99)
            self.pending.append((destination, source))
            return destination

        def event(self, **kwargs):
            event = Mock()

            def record():
                index = len(self.pending) - 1

                def synchronize():
                    assert self.ready_through is None or index <= self.ready_through, (
                        "retrieving a request waited for an unrelated pending copy"
                    )
                    self.waited.append(index)
                    for destination, source in self.pending[: index + 1]:
                        np.copyto(destination, source)

                event.synchronize.side_effect = synchronize

            event.record.side_effect = record
            return event

    copies = Copies()
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Event", copies.event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    monkeypatch.setattr(utils, "async_copy_to_np", copies.to_numpy)
    return copies


def test_variable_length_drafts_wait_for_copy_before_trimming(cuda_copies):
    """Remote rows retain their exact lengths after the asynchronous copy."""
    device = torch.device("cpu")
    handler = DraftTokensHandler(device, variable_length=True)
    batch = _input_batch(["a", "empty"], structured=False, device=device)
    handler.set_draft_tokens(batch, torch.tensor([[11, 12, -1], [-1, -1, -1]]))

    assert _get_draft_toks(handler) == {"a": [11, 12], "empty": []}
    assert cuda_copies.waited == [0]


def test_retained_drafts_wait_only_for_their_own_copy(cuda_copies):
    """An older request remains readable while an unrelated copy is pending."""
    device = torch.device("cpu")
    handler = DraftTokensHandler(device, variable_length=True)
    handler.set_draft_tokens(
        _input_batch(["a", "b"], structured=False, device=device),
        torch.tensor([[11, 12, -1], [21, 22, -1]]),
    )
    handler.set_draft_tokens(
        _input_batch(["b"], structured=False, device=device),
        torch.tensor([[31, -1, -1]]),
    )
    cuda_copies.ready_through = 0
    selected = SchedulerOutput.make_empty()
    selected.has_structured_output_spec_decode = True
    selected.scheduled_spec_decode_tokens = {"a": [-1, -1]}
    handler.begin_step(selected)
    assert _get_draft_toks(handler) == {"a": [11, 12]}
    assert cuda_copies.waited == [0]

    cuda_copies.ready_through = 1
    handler.begin_step(SchedulerOutput.make_empty())
    assert _get_draft_toks(handler) == {"a": [11, 12], "b": [31]}
    assert cuda_copies.waited == [0, 1]
    assert _get_draft_toks(handler) == {"a": [11, 12], "b": [31]}
    assert cuda_copies.waited == [0, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_draft_tokens_follow_request_lifecycle() -> None:
    handler = DraftTokensHandler(torch.device("cuda"))

    # The worker samples two structured-output requests
    _set_draft_toks(
        handler,
        req_ids=["a", "b"],
        tokens=[[11, 12], [21, 22]],
        structured=True,
    )

    # "b" finishes before this result is handed back to the scheduler
    handler.discard("b")
    assert _get_draft_toks(handler) == {"a": [11, 12]}

    # "a" continues and "b" is re-added with a new draft row
    _set_draft_toks(
        handler,
        req_ids=["a", "b"],
        tokens=[[13, 14], [23, 24]],
        structured=True,
    )
    assert _get_draft_toks(handler) == {
        "a": [13, 14],
        "b": [23, 24],
    }

    # A batch without structured-output must not erase drafts from live requests
    _set_draft_toks(
        handler,
        req_ids=["plain"],
        tokens=[[31, 32]],
        structured=False,
    )
    assert _get_draft_toks(handler) == {
        "a": [13, 14],
        "b": [23, 24],
        "plain": [-1, -1],
    }

    # Grammar validation only fetches the requests in the latest scheduler
    # output.
    deferred_output = SchedulerOutput.make_empty()
    deferred_output.has_structured_output_spec_decode = True
    deferred_output.scheduled_spec_decode_tokens = {"a": [-1, -1]}
    handler.begin_step(deferred_output)
    assert _get_draft_toks(handler) == {"a": [13, 14]}

    # Without structured-output spec decode, all retained drafts are fetched.
    handler.begin_step(SchedulerOutput.make_empty())

    # Requests finish independently, and the handler is empty once all are gone
    handler.discard("a")
    assert _get_draft_toks(handler) == {
        "b": [23, 24],
        "plain": [-1, -1],
    }
    handler.discard("b")
    handler.discard("plain")
    assert _get_draft_toks(handler) == {}
