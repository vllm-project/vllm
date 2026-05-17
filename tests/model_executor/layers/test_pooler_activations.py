# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.model_executor.layers.pooler.activations."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.pooler.activations import (
    LambdaPoolerActivation,
    PoolerClassify,
    PoolerIdentity,
    PoolerMultiLabelClassify,
    PoolerNormalize,
    get_act_fn,
    resolve_classifier_act_fn,
)


@pytest.fixture
def vllm_config():
    """VllmConfig with a minimal model_config stub for PoolerClassify."""
    config = VllmConfig()
    config.model_config = SimpleNamespace(hf_config=SimpleNamespace(num_labels=0))
    with set_current_vllm_config(config):
        yield config


# ---------------------------------------------------------------------------
# PoolerIdentity
# ---------------------------------------------------------------------------
class TestPoolerIdentity:
    def test_returns_input_unchanged(self):
        pooler = PoolerIdentity()
        x = torch.randn(4, 128)
        out = pooler(x)
        assert torch.equal(out, x)

    def test_forward_list(self):
        pooler = PoolerIdentity()
        tensors = [torch.randn(128), torch.randn(256)]
        out = pooler(tensors)
        assert len(out) == 2
        for orig, result in zip(tensors, out):
            assert torch.equal(orig, result)


# ---------------------------------------------------------------------------
# PoolerNormalize
# ---------------------------------------------------------------------------
class TestPoolerNormalize:
    def test_output_has_unit_norm(self):
        pooler = PoolerNormalize()
        x = torch.randn(4, 128)
        out = pooler(x)
        norms = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_single_vector(self):
        pooler = PoolerNormalize()
        x = torch.randn(1, 64)
        out = pooler(x)
        norm = torch.linalg.norm(out, dim=-1)
        assert torch.allclose(norm, torch.ones(1), atol=1e-5)

    def test_forward_list(self):
        pooler = PoolerNormalize()
        tensors = [torch.randn(1, 64), torch.randn(1, 128)]
        out = pooler(tensors)
        for t in out:
            norm = torch.linalg.norm(t, dim=-1)
            assert torch.allclose(norm, torch.ones(1), atol=1e-5)


# ---------------------------------------------------------------------------
# PoolerMultiLabelClassify
# ---------------------------------------------------------------------------
class TestPoolerMultiLabelClassify:
    def test_output_in_zero_one(self):
        pooler = PoolerMultiLabelClassify()
        x = torch.randn(4, 10)
        out = pooler(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_large_positive_maps_near_one(self):
        pooler = PoolerMultiLabelClassify()
        x = torch.full((1, 3), 100.0)
        out = pooler(x)
        assert torch.allclose(out, torch.ones(1, 3), atol=1e-4)

    def test_large_negative_maps_near_zero(self):
        pooler = PoolerMultiLabelClassify()
        x = torch.full((1, 3), -100.0)
        out = pooler(x)
        assert torch.allclose(out, torch.zeros(1, 3), atol=1e-4)


# ---------------------------------------------------------------------------
# PoolerClassify
# ---------------------------------------------------------------------------
class TestPoolerClassify:
    def test_dynamic_softmax_when_num_labels_ge_2(self):
        pooler = PoolerClassify(static_num_labels=False)
        assert pooler.num_labels is None
        x = torch.randn(2, 5)
        out = pooler(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_dynamic_sigmoid_when_num_labels_lt_2(self):
        pooler = PoolerClassify(static_num_labels=False)
        x = torch.zeros(1, 1)
        out = pooler(x)
        assert torch.allclose(out, torch.tensor([[0.5]]), atol=1e-5)

    def test_static_default_num_labels_zero_uses_sigmoid(self, vllm_config):
        pooler = PoolerClassify(static_num_labels=True)
        assert pooler.num_labels == 0
        x = torch.zeros(1, 3)
        out = pooler(x)
        assert torch.allclose(out, torch.full((1, 3), 0.5), atol=1e-5)

    def test_static_num_labels_ge_2_uses_softmax(self, vllm_config):
        vllm_config.model_config.hf_config.num_labels = 4
        pooler = PoolerClassify(static_num_labels=True)
        assert pooler.num_labels == 4
        x = torch.randn(2, 4)
        out = pooler(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# LambdaPoolerActivation
# ---------------------------------------------------------------------------
class TestLambdaPoolerActivation:
    def test_applies_custom_fn(self):
        pooler = LambdaPoolerActivation(nn.ReLU())
        x = torch.tensor([[-1.0, 2.0, -3.0]])
        out = pooler(x)
        expected = torch.tensor([[0.0, 2.0, 0.0]])
        assert torch.equal(out, expected)

    def test_forward_list(self):
        pooler = LambdaPoolerActivation(nn.ReLU())
        tensors = [torch.tensor([-1.0, 2.0]), torch.tensor([3.0, -4.0])]
        out = pooler(tensors)
        assert torch.equal(out[0], torch.tensor([0.0, 2.0]))
        assert torch.equal(out[1], torch.tensor([3.0, 0.0]))


# ---------------------------------------------------------------------------
# get_act_fn factory
# ---------------------------------------------------------------------------
class TestGetActFn:
    @staticmethod
    def _make_config(**kwargs):
        return SimpleNamespace(**kwargs)

    def test_regression(self, vllm_config):
        cfg = self._make_config(problem_type="regression")
        result = get_act_fn(cfg)
        assert isinstance(result, PoolerIdentity)

    def test_single_label_classification(self, vllm_config):
        cfg = self._make_config(
            problem_type="single_label_classification", num_labels=3
        )
        result = get_act_fn(cfg)
        assert isinstance(result, PoolerClassify)

    def test_multi_label_classification(self, vllm_config):
        cfg = self._make_config(problem_type="multi_label_classification")
        result = get_act_fn(cfg)
        assert isinstance(result, PoolerMultiLabelClassify)

    def test_sentence_transformers_activation(self, vllm_config):
        cfg = self._make_config(
            problem_type="",
            sentence_transformers={
                "activation_fn": "torch.nn.modules.activation.Sigmoid"
            },
        )
        result = get_act_fn(cfg)
        assert isinstance(result, PoolerClassify)

    def test_sbert_activation(self, vllm_config):
        cfg = self._make_config(
            problem_type="",
            sbert_ce_default_activation_function=(
                "torch.nn.modules.activation.Sigmoid"
            ),
        )
        result = get_act_fn(cfg)
        assert isinstance(result, PoolerClassify)

    def test_default_fallback(self, vllm_config):
        cfg = self._make_config(problem_type="")
        result = get_act_fn(cfg)
        assert isinstance(result, PoolerClassify)

    def test_sentence_transformers_takes_priority(self, vllm_config):
        cfg = self._make_config(
            problem_type="",
            sentence_transformers={"activation_fn": "torch.nn.modules.linear.Identity"},
            sbert_ce_default_activation_function=(
                "torch.nn.modules.activation.Sigmoid"
            ),
        )
        result = get_act_fn(cfg)
        assert isinstance(result, PoolerIdentity)

    def test_rejects_non_torch_activation(self, vllm_config):
        cfg = self._make_config(
            problem_type="",
            sentence_transformers={"activation_fn": "os.system"},
        )
        with pytest.raises(AssertionError, match="restricted"):
            get_act_fn(cfg)


# ---------------------------------------------------------------------------
# resolve_classifier_act_fn
# ---------------------------------------------------------------------------
class TestResolveClassifierActFn:
    def test_delegates_to_get_act_fn_when_none(self, vllm_config):
        model_config = vllm_config.model_config
        result = resolve_classifier_act_fn(model_config, act_fn=None)
        assert isinstance(result, PoolerClassify)

    def test_passes_through_provided_act_fn(self):
        custom = PoolerIdentity()
        result = resolve_classifier_act_fn(None, act_fn=custom)
        assert result is custom
