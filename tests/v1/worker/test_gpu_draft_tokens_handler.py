# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Comment: from tomasruizt#20 (consolidated); discard() API per
# the converged handler.
from dataclasses import replace

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_overlapping_batches_preserve_drafts_until_request_removal() -> None:
    handler = DraftTokensHandler(max_num_reqs=8, device=torch.device("cuda"))
    _add_draft_toks(handler, req_id="first", tokens=[11, 12], structured=True)
    _add_draft_toks(handler, req_id="second", tokens=[21, 22], structured=True)
    assert _get_draft_toks(handler) == {"first": [11, 12], "second": [21, 22]}

    # A newer plain batch must not erase a pending structured batch's drafts.
    _add_draft_toks(handler, req_id="plain", tokens=[31, 32], structured=False)
    assert _get_draft_toks(handler) == {
        "first": [11, 12],
        "second": [21, 22],
        "plain": [-1, -1],
    }

    _add_draft_toks(handler, req_id="first", tokens=[13, 14], structured=True)
    # "plain" is still live, so its width placeholder persists until discard.
    assert _get_draft_toks(handler) == {
        "first": [13, 14],
        "second": [21, 22],
        "plain": [-1, -1],
    }
    handler.discard("first")
    handler.discard("second")
    handler.discard("plain")
    assert _get_draft_toks(handler) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_request_removed_before_retrieval_can_reuse_id() -> None:
    handler = DraftTokensHandler(max_num_reqs=8, device=torch.device("cuda"))
    _add_draft_toks(handler, req_id="reused", tokens=[11, 12], structured=True)
    handler.discard("reused")
    assert _get_draft_toks(handler) == {}

    _add_draft_toks(handler, req_id="reused", tokens=[21, 22], structured=False)
    assert _get_draft_toks(handler) == {"reused": [-1, -1]}
    _add_draft_toks(handler, req_id="reused", tokens=[31, 32], structured=True)
    assert _get_draft_toks(handler) == {"reused": [31, 32]}
    handler.discard("reused")
    assert _get_draft_toks(handler) == {}


def _add_draft_toks(
    handler: DraftTokensHandler, req_id: str, tokens: list[int], structured: bool
) -> None:
    batch = _input_batch(req_id=req_id, structured=structured)
    handler.set_draft_tokens(batch, torch.tensor([tokens], device="cuda"))


def _get_draft_toks(handler: DraftTokensHandler) -> dict[str, list[int]]:
    drafts = handler.get_draft_tokens()
    assert drafts is not None
    return dict(zip(drafts.req_ids, drafts.draft_token_ids, strict=True))


def _input_batch(req_id: str, structured: bool) -> InputBatch:
    buffers = InputBuffers(1, 1, torch.device("cuda"))
    batch = InputBatch.make_dummy(1, 1, buffers, is_padding=False)
    return replace(batch, req_ids=[req_id], has_structured_output_reqs=structured)
