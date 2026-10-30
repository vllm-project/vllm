# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


def _input_batch(req_ids: list[str], structured: bool) -> InputBatch:
    num_reqs = len(req_ids)
    buffers = InputBuffers(num_reqs, num_reqs, torch.device("cuda"))
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

    # Requests finish independently, and the handler is empty once all are gone
    handler.discard("a")
    assert _get_draft_toks(handler) == {
        "b": [23, 24],
        "plain": [-1, -1],
    }
    handler.discard("b")
    handler.discard("plain")
    assert _get_draft_toks(handler) == {}
