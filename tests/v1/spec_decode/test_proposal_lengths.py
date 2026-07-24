# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-request effective proposal lengths (confidence truncation).

Covers the pure pieces that do not need a GPU model: the confidence-threshold
config validation, the first-below-threshold length computation used by the
DSpark speculator, and the DraftTokensHandler prefix clipping semantics.
"""

import numpy as np
import pytest
import torch

from vllm.config.speculative import SpeculativeConfig


def _lengths_from_conf_logits(
    conf_logits: torch.Tensor, logit_threshold: float
) -> torch.Tensor:
    # Mirrors DSparkSpeculator._sample_sequential's post-loop computation.
    n_spec = conf_logits.shape[1]
    below = conf_logits < logit_threshold
    first_below = below.to(torch.int32).argmax(dim=1)
    any_below = below.any(dim=1)
    return torch.where(any_below, first_below, torch.full_like(first_below, n_spec))


def test_confidence_threshold_rejects_out_of_range():
    with pytest.raises(ValueError, match="confidence_threshold"):
        SpeculativeConfig(method="dspark", confidence_threshold=1.0)
    with pytest.raises(ValueError, match="confidence_threshold"):
        SpeculativeConfig(method="dspark", confidence_threshold=-0.1)


def test_confidence_threshold_requires_dspark():
    with pytest.raises(ValueError, match="dspark"):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=3,
            prompt_lookup_max=4,
            confidence_threshold=0.5,
        )


def test_first_below_threshold_lengths():
    # threshold 0.5 -> logit threshold 0.0
    conf = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],  # never below -> full length
            [-1.0, 1.0, 1.0, 1.0],  # below at position 0 -> length 0
            [1.0, 1.0, -1.0, 1.0],  # below at position 2 -> length 2
            [1.0, -1.0, -1.0, -1.0],  # first below wins -> length 1
        ]
    )
    lengths = _lengths_from_conf_logits(conf, logit_threshold=0.0)
    assert lengths.tolist() == [4, 0, 2, 1]


class _FakeInputBatch:
    def __init__(self, req_ids, has_structured_output_reqs=False):
        self.req_ids = req_ids
        self.has_structured_output_reqs = has_structured_output_reqs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_handler_clips_proposals_to_lengths():
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

    device = torch.device("cuda")
    handler = DraftTokensHandler(device)
    batch = _FakeInputBatch(["a", "b", "c"])
    draft_tokens = torch.arange(12, device=device).reshape(3, 4)
    lengths = torch.tensor([4, 0, 2], dtype=torch.int32, device=device)

    handler.set_draft_tokens(batch, draft_tokens, proposal_lengths=lengths)
    out = handler.get_draft_tokens()
    assert out.req_ids == ["a", "b", "c"]
    assert [len(ids) for ids in out.draft_token_ids] == [4, 0, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_handler_without_lengths_preserves_uniform_behavior():
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

    device = torch.device("cuda")
    handler = DraftTokensHandler(device)
    batch = _FakeInputBatch(["a", "b"])
    draft_tokens = torch.zeros(2, 4, dtype=torch.int64, device=device)

    handler.set_draft_tokens(batch, draft_tokens)
    out = handler.get_draft_tokens()
    assert [len(ids) for ids in out.draft_token_ids] == [4, 4]
    assert all(t == -1 for ids in out.draft_token_ids for t in ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_handler_lengths_with_structured_output_tokens():
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

    device = torch.device("cuda")
    handler = DraftTokensHandler(device)
    batch = _FakeInputBatch(["a", "b"], has_structured_output_reqs=True)
    draft_tokens = torch.arange(8, device=device).reshape(2, 4)
    lengths = torch.tensor([1, 3], dtype=torch.int32, device=device)

    handler.set_draft_tokens(batch, draft_tokens, proposal_lengths=lengths)
    out = handler.get_draft_tokens()
    np.testing.assert_array_equal(out.draft_token_ids[0], [0])
    np.testing.assert_array_equal(out.draft_token_ids[1], [4, 5, 6])
