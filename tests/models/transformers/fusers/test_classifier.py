# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers modeling backend's classifier fuser."""

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from vllm.model_executor.models.transformers.fusers import ClassifierFuser
from vllm.model_executor.models.transformers.fusers.classifier import (
    trace_classification_model,
)


def build(auto_cls, model_type: str, device: str = "meta", **overrides):
    config = AutoConfig.for_model(model_type, **overrides)
    with torch.device(device):
        return auto_cls.from_config(config).eval()


def vllm_config(dtype=torch.float32, head_dtype=torch.float32):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=dtype, head_dtype=head_dtype)
    )


@pytest.mark.parametrize(
    ("auto_cls", "pooling"),
    [
        (AutoModelForSequenceClassification, "cls"),
        (AutoModelForSequenceClassification, "mean"),
        (AutoModelForTokenClassification, None),
    ],
)
def test_modernbert_head_matches_hf(auto_cls, pooling):
    """The fused head applied to vLLM-pooled hidden states reproduces HF's logits."""
    torch.manual_seed(0)
    model = build(
        auto_cls,
        "modernbert",
        device="cpu",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        classifier_pooling=pooling or "cls",
    )
    pooled = pooling is not None
    fuser = ClassifierFuser.match_model(model, pooled=pooled)
    assert fuser is not None
    head = fuser.fuse(model, "", vllm_config())

    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        hidden = model.model(input_ids).last_hidden_state[0]
        if pooling == "cls":
            hidden = hidden[:1]
        elif pooling == "mean":
            hidden = hidden.mean(0, keepdim=True)
        expected = model(input_ids).logits.reshape(hidden.shape[0], -1)
        torch.testing.assert_close(head(hidden), expected)


def test_head_matches_with_compiled_base_model():
    """`@support_torch_compile` overriding the base model's `__call__` is traced."""
    model = build(AutoModelForSequenceClassification, "modernbert")
    base_cls = type(model.model)
    model.model.__class__ = type(
        base_cls.__name__,
        (base_cls,),
        {"__call__": lambda self, *args, **kwargs: self.forward(*args, **kwargs)},
    )
    assert ClassifierFuser.match_model(model, pooled=True) is not None


def test_matching_does_not_change_config():
    """Tracing `labels` must not set `problem_type`, which picks the activation."""
    model = build(AutoModelForSequenceClassification, "modernbert", num_labels=1)
    ClassifierFuser.match_model(model, pooled=True)
    assert model.config.problem_type is None


@pytest.mark.parametrize(
    ("model_type", "classifier_only", "reason"),
    [
        ("distilbert", False, "applies an inline `nn.ReLU()`, which ends the trace"),
        ("deberta-v2", False, "pools inside its `pooler` module"),
        ("bert", False, "classifies the base model's `pooler_output`"),
        ("roberta", True, "pools inside its `classifier` module"),
        ("qwen2", True, "scores every token before pooling"),
    ],
    ids=["distilbert", "deberta-v2", "bert", "roberta", "qwen2"],
)
def test_unsupported_heads_are_not_matched(model_type, classifier_only, reason):
    """Unmatched heads may only fall back to a classifier that reads hidden states."""
    model = build(AutoModelForSequenceClassification, model_type, num_hidden_layers=1)
    graph = trace_classification_model(model)
    assert ClassifierFuser.match(graph, model, pooled=True) is None, reason
    assert ClassifierFuser.classifies_hidden_states(graph, model) is classifier_only


def test_only_classifier_runs_in_head_dtype():
    """Like native models, the head runs in the model dtype, the classifier doesn't."""
    model = build(
        AutoModelForTokenClassification,
        "modernbert",
        device="cpu",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
    ).to(torch.bfloat16)
    model.classifier.float()
    fuser = ClassifierFuser.match_model(model, pooled=False)
    head = fuser.fuse(model, "", vllm_config(torch.bfloat16, torch.float32))
    with torch.no_grad():
        logits = head(torch.randn(6, 32, dtype=torch.float32))
    assert logits.dtype == torch.float32
