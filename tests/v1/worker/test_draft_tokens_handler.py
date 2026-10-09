# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_draft_tokens_transferred_without_structured_output():
    """Drafts must be transferred even for batches without structured-output
    requests.

    The previous fast path cleared draft_tokens_np for such batches, so
    get_draft_tokens returned -1 placeholders that the scheduler treated
    as real draft tokens: with async scheduling disabled this killed the
    PP worker (1 query row vs 2 logits rows -> prepare_inputs assert);
    with async scheduling it silently corrupts structured-output
    constraints (#54437).
    """
    device = torch.device("cuda")
    handler = DraftTokensHandler(device)

    input_batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        has_structured_output_reqs=False,
    )
    draft_tokens = torch.tensor([[10, 11], [12, 13]], device=device)

    handler.set_draft_tokens(input_batch, draft_tokens)
    result = handler.get_draft_tokens()

    assert isinstance(result, DraftTokenIds)
    assert result.req_ids == ["r0", "r1"]
    assert result.draft_token_ids == [[10, 11], [12, 13]]
