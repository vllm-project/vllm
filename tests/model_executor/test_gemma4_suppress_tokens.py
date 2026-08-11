# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from torch import nn

from vllm.model_executor.models.gemma4_mtp import Gemma4MTP, Gemma4MTPMaskedEmbedder
from vllm.model_executor.models.gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
)
from vllm.model_executor.models.utils import (
    make_suppress_token_ids,
    register_suppress_token_ids,
)


class _FakeLanguageModel(nn.Module):
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.clone()


class _FakeLogitsProcessor:
    def __call__(self, lm_head: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.clone()


def test_gemma4_unified_suppress_tokens_masking():
    model = Gemma4UnifiedForConditionalGeneration.__new__(
        Gemma4UnifiedForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.language_model = _FakeLanguageModel()
    model.register_buffer(
        "_suppress_token_ids",
        torch.tensor([1, 3], dtype=torch.long),
        persistent=False,
    )

    logits = model.compute_logits(torch.zeros(2, 5))

    assert torch.isneginf(logits[:, [1, 3]]).all()
    assert torch.equal(logits[:, [0, 2, 4]], torch.zeros(2, 3))


@pytest.mark.parametrize(
    ("suppress_tokens", "expected_suppressed"),
    [(0, [0]), (torch.empty(0, dtype=torch.long), [])],
)
def test_gemma4_mtp_suppress_tokens_masking(suppress_tokens, expected_suppressed):
    model = Gemma4MTP.__new__(Gemma4MTP)
    nn.Module.__init__(model)
    model.masked_embedding = None
    model.lm_head = nn.Identity()
    model.logits_processor = _FakeLogitsProcessor()
    register_suppress_token_ids(model, suppress_tokens, torch.empty(0), 5)

    logits = model.compute_logits(torch.zeros(2, 5))

    if expected_suppressed:
        assert torch.isneginf(logits[:, expected_suppressed]).all()
    else:
        assert torch.equal(logits, torch.zeros(2, 5))


def test_gemma4_mtp_sparse_argmax_excludes_suppressed_tokens(monkeypatch):
    masked_embedding = Gemma4MTPMaskedEmbedder(
        hidden_size=2,
        vocab_size=4,
        num_centroids=2,
        centroid_intermediate_top_k=1,
    )
    sparse_logits = torch.tensor([[10.0, 9.0]])
    sparse_indices = torch.tensor([[1, 2]])
    monkeypatch.setattr(
        masked_embedding,
        "_select_and_score",
        lambda hidden_states, lm_head_weight: (sparse_logits, sparse_indices),
    )

    top_tokens = masked_embedding.get_top_tokens(
        torch.zeros(1, 2),
        torch.zeros(4, 2),
        torch.tensor([1]),
        torch.tensor(0),
    )

    assert torch.equal(top_tokens, torch.tensor([2]))


@pytest.mark.parametrize(
    ("suppress_token_ids", "expected"),
    [
        (None, 1),
        ([1], 2),
        ([1, 2], 0),
    ],
)
def test_gemma4_mtp_sparse_argmax_matches_full_logits(
    monkeypatch,
    suppress_token_ids,
    expected,
):
    masked_embedding = Gemma4MTPMaskedEmbedder(
        hidden_size=2,
        vocab_size=4,
        num_centroids=2,
        centroid_intermediate_top_k=1,
    )
    sparse_logits = torch.tensor([[10.0, 9.0]])
    sparse_indices = torch.tensor([[1, 2]])
    monkeypatch.setattr(
        masked_embedding,
        "_select_and_score",
        lambda hidden_states, lm_head_weight: (sparse_logits, sparse_indices),
    )
    model = nn.Module()
    register_suppress_token_ids(model, suppress_token_ids, torch.empty(0), 4)

    sparse_top = masked_embedding.get_top_tokens(
        torch.zeros(1, 2),
        torch.zeros(4, 2),
        model._suppress_token_ids,
        model._suppress_fallback_token_id,
    )
    full_logits = masked_embedding(
        torch.zeros(1, 2),
        torch.zeros(4, 2),
    )
    if model._suppress_token_ids is not None:
        full_logits.index_fill_(1, model._suppress_token_ids, -float("inf"))

    assert torch.equal(sparse_top, full_logits.argmax(-1))
    assert sparse_top.item() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_gemma4_mtp_suppress_tokens_cuda_graph_capture_and_replay():
    model = Gemma4MTP.__new__(Gemma4MTP)
    nn.Module.__init__(model)
    model.masked_embedding = None
    model.lm_head = nn.Identity()
    model.logits_processor = _FakeLogitsProcessor()

    hidden_states = torch.zeros(2, 5, device="cuda")
    register_suppress_token_ids(model, [1, 3], hidden_states, 5)

    model.compute_logits(hidden_states)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model.compute_logits(hidden_states)

    for value in (1.0, 2.0):
        hidden_states.fill_(value)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.isneginf(logits[:, [1, 3]]).all()
        assert torch.equal(
            logits[:, [0, 2, 4]],
            torch.full((2, 3), value, device="cuda"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize(
    ("suppress_token_ids", "expected"),
    [([1], 2), ([1, 2], 0)],
    ids=["partial", "all-selected"],
)
def test_gemma4_mtp_sparse_suppress_tokens_cuda_graph_capture_and_replay(
    monkeypatch,
    suppress_token_ids,
    expected,
):
    model = Gemma4MTP.__new__(Gemma4MTP)
    nn.Module.__init__(model)
    model.masked_embedding = Gemma4MTPMaskedEmbedder(
        hidden_size=2,
        vocab_size=4,
        num_centroids=2,
        centroid_intermediate_top_k=1,
    ).cuda()
    indices = torch.tensor([[1, 2]], device="cuda")
    monkeypatch.setattr(
        model.masked_embedding,
        "_select_and_score",
        lambda hidden_states, lm_head_weight: (hidden_states, indices),
    )
    lm_head_weight = torch.zeros(4, 2, device="cuda")
    monkeypatch.setattr(model, "_get_full_lm_head_weight", lambda: lm_head_weight)
    register_suppress_token_ids(model, suppress_token_ids, lm_head_weight, 4)

    sparse_logits = torch.tensor([[10.0, 9.0]], device="cuda")
    model.get_top_tokens(sparse_logits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        top_tokens = model.get_top_tokens(sparse_logits)

    for values in ([10.0, 9.0], [1.0, 2.0]):
        sparse_logits.copy_(torch.tensor([values], device="cuda"))
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(top_tokens, torch.tensor([expected], device="cuda"))


def test_register_suppress_token_ids_follows_parameter_device_and_is_nonpersistent():
    model = nn.Module()
    reference = nn.Parameter(torch.empty(1, device="cpu"))

    register_suppress_token_ids(model, torch.tensor([1, 3]), reference, 5)

    assert model._suppress_token_ids.device == reference.device
    assert "_suppress_token_ids" not in model.state_dict()
    assert "_suppress_fallback_token_id" not in model.state_dict()
    model.to("meta")
    assert model._suppress_token_ids.device.type == "meta"
    assert model._suppress_fallback_token_id.device.type == "meta"


def test_register_suppress_token_ids_supports_meta_reference():
    model = nn.Module()
    reference = torch.empty(1, device="meta")

    register_suppress_token_ids(model, [1, 3], reference, 5)

    assert model._suppress_token_ids.device.type == "meta"
    assert model._suppress_fallback_token_id.device.type == "meta"


@pytest.mark.parametrize(
    ("suppress_tokens", "expected"),
    [
        (None, None),
        ([], None),
        ((), None),
        (np.array([], dtype=np.int64), None),
        (torch.empty(0, dtype=torch.long), None),
        (0, [0]),
        (7, [7]),
        ([1, 3], [1, 3]),
        ((2, 4), [2, 4]),
        (np.array([5, 6]), [5, 6]),
        (torch.tensor([8, 9]), [8, 9]),
        (torch.tensor(10), [10]),
    ],
)
def test_make_suppress_token_ids(suppress_tokens, expected):
    token_ids = make_suppress_token_ids(suppress_tokens, torch.device("cpu"), 16)

    if expected is None:
        assert token_ids is None
    else:
        assert token_ids is not None
        assert token_ids.ndim == 1
        assert token_ids.dtype == torch.long
        assert torch.equal(token_ids, torch.tensor(expected, dtype=torch.long))


@pytest.mark.parametrize("suppress_tokens", [True, 1.5, [1, 2.0], torch.tensor(3.0)])
def test_make_suppress_token_ids_rejects_non_integral_values(suppress_tokens):
    with pytest.raises(ValueError, match="integer token IDs"):
        make_suppress_token_ids(suppress_tokens, torch.device("cpu"), 4)


@pytest.mark.parametrize("suppress_tokens", [-1, 4, [0, 4]])
def test_make_suppress_token_ids_rejects_out_of_range_values(suppress_tokens):
    with pytest.raises(ValueError, match=r"must be in \[0, 4\)"):
        make_suppress_token_ids(suppress_tokens, torch.device("cpu"), 4)


def test_make_suppress_token_ids_rejects_entire_vocabulary():
    with pytest.raises(ValueError, match="entire vocabulary"):
        make_suppress_token_ids([0, 1, 2, 3], torch.device("cpu"), 4)
