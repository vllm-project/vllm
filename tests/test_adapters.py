# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sequence-classification weight-loading adapters.

Regression tests for https://github.com/vllm-project/vllm/issues/59784:
text configs such as transformers' ``Qwen3VLTextConfig`` do not define the
``tie_word_embeddings`` attribute. The loaders must treat a missing
attribute as ``False`` (matching the parent config's declared value)
instead of raising ``AttributeError``.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as vpe_module
import vllm.tokenizers as tokenizers_module
from vllm.model_executor.models import adapters


class _FakeLMHead:
    """Test double for ParallelLMHead recording tie_weights calls."""

    created: list["_FakeLMHead"] = []

    def __init__(self, vocab_size: int, hidden_size: int):
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size))
        self.tie_weights_calls: list[Any] = []
        _FakeLMHead.created.append(self)

    def tie_weights(self, embed_tokens):
        self.tie_weights_calls.append(embed_tokens)
        return self


class _FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return {"no": 0, "yes": 1}[token]


class _FakePoolingModel:
    """Minimal stand-in for a pooling model going through seq-cls loading."""

    def __init__(self, text_config: SimpleNamespace, num_labels: int):
        self.config = SimpleNamespace(
            get_text_config=lambda: text_config,
            classifier_from_token=["no", "yes"],
        )
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                tokenizer="fake",
                tokenizer_revision=None,
                tokenizer_mode="auto",
                trust_remote_code=False,
            )
        )
        # Inner backbone exposing input embeddings for the tie_weights path.
        self.model = SimpleNamespace(embed_tokens=object())
        self.score = nn.Linear(text_config.hidden_size, num_labels, bias=False)

    def named_children(self):
        return iter(())

    def _load_pooling_model_weights(self, weights):
        return set()


def _make_text_config(**overrides) -> SimpleNamespace:
    config = SimpleNamespace(
        vocab_size=8,
        hidden_size=4,
        classifier_from_token=["no", "yes"],
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.fixture()
def _patched_collaborators(monkeypatch):
    """Replace environment-dependent collaborators with fakes.

    ParallelLMHead needs distributed parallel state and get_tokenizer
    downloads from the Hub; neither is relevant to the tie_word_embeddings
    guard under test.
    """
    monkeypatch.setattr(vpe_module, "ParallelLMHead", _FakeLMHead)
    monkeypatch.setattr(
        tokenizers_module,
        "get_tokenizer",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    _FakeLMHead.created.clear()


@pytest.mark.parametrize(
    "loader,num_labels",
    [
        (adapters.load_weights_using_from_2_way_softmax, 1),
        (adapters.load_weights_no_post_processing, 2),
    ],
    ids=["from_2_way_softmax", "no_post_processing"],
)
def test_missing_tie_word_embeddings_falls_back_to_false(
    _patched_collaborators, loader, num_labels
):
    """A text config without tie_word_embeddings must not raise.

    Regression test for
    https://github.com/vllm-project/vllm/issues/59784: the missing
    attribute falls back to False, so the tied-weights path is skipped.
    """
    text_config = _make_text_config()
    assert not hasattr(text_config, "tie_word_embeddings")
    model = _FakePoolingModel(text_config, num_labels)

    loaded = loader(model, [])
    (lm_head,) = _FakeLMHead.created

    assert loaded == {"score.weight"}
    assert lm_head.tie_weights_calls == []
    assert not hasattr(model, "lm_head")


@pytest.mark.parametrize(
    "loader,num_labels",
    [
        (adapters.load_weights_using_from_2_way_softmax, 1),
        (adapters.load_weights_no_post_processing, 2),
    ],
    ids=["from_2_way_softmax", "no_post_processing"],
)
@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_tie_word_embeddings_present_behaves_as_before(
    _patched_collaborators, loader, num_labels, tie_word_embeddings
):
    """Configs defining tie_word_embeddings keep their exact behavior."""
    text_config = _make_text_config(tie_word_embeddings=tie_word_embeddings)
    model = _FakePoolingModel(text_config, num_labels)

    loaded = loader(model, [])
    (lm_head,) = _FakeLMHead.created

    assert loaded == {"score.weight"}
    if tie_word_embeddings:
        assert len(lm_head.tie_weights_calls) == 1
    else:
        assert lm_head.tie_weights_calls == []
