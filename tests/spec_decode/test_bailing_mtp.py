# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline unit tests for the Bailing MTP draft model.

These tests cover the parts of the BailingMTP implementation that do not
require a GPU or a real HuggingFace checkpoint:

* Class import and registration via the model registry.
* ``SpeculativeConfig.hf_config_override`` rewriting
  ``bailing_hybrid`` -> ``bailing_mtp``/``BailingMTPModel``.
* ``BailingMTP._rewrite_spec_layer_name`` handling the
  ``attention``/``self_attn`` and ``attention.dense``/``o_proj``
  checkpoint-name rewrites, and the shared-embed_tokens promotion.

End-to-end inference tests live in
``tests/models/language/generation/`` and are gated on a real
checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

# Skip the whole module if torch is unavailable so the test file is
# importable on the lint/CI side even on minimal containers.
torch = pytest.importorskip("torch")

from vllm.config.speculative import (  # noqa: E402
    MTPModelTypes,
    SpeculativeConfig,
)
from vllm.model_executor.models import registry as model_registry  # noqa: E402
from vllm.model_executor.models.bailing_mtp import (  # noqa: E402
    BailingMTP,
    BailingMultiTokenPredictor,
    BailingMultiTokenPredictorLayer,
)
from vllm.transformers_utils.model_arch_config_convertor import (  # noqa: E402
    MODEL_ARCH_CONFIG_CONVERTORS,
)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_bailing_mtp_registered_in_speculative_decoding_models() -> None:
    """The architecture must be discoverable by the speculative registry."""
    spec_models = (
        model_registry._SPECULATIVE_DECODING_MODELS  # type: ignore[attr-defined]
    )
    assert spec_models.get("BailingMTPModel") == ("bailing_mtp", "BailingMTP")


def test_bailing_mtp_in_mtp_literal() -> None:
    """``bailing_mtp`` must be a valid ``SpeculativeConfig.method`` value."""
    assert "bailing_mtp" in MTPModelTypes.__args__


def test_bailing_mtp_arch_config_convertor_registered() -> None:
    """The convertor must be registered under the ``bailing_mtp`` model_type."""
    assert "bailing_mtp" in MODEL_ARCH_CONFIG_CONVERTORS


# ---------------------------------------------------------------------------
# hf_config_override tests
# ---------------------------------------------------------------------------


def _make_bailing_hybrid_config(
    *, num_nextn_predict_layers: int | None = 1
) -> SimpleNamespace:
    """Build a minimal stand-in for a Bailing Hybrid HF config."""
    return SimpleNamespace(
        architectures=["BailingMoeV2_5ForCausalLM"],
        model_type="bailing_hybrid",
        num_nextn_predict_layers=num_nextn_predict_layers,
        update=lambda self, kwargs: None,  # placeholder; see below
    )


def test_hf_config_override_rewrites_bailing_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``bailing_hybrid`` is rewritten to ``bailing_mtp`` with the
    matching architecture and ``n_predict`` set from
    ``num_nextn_predict_layers``.
    """
    updates: list[dict] = []

    class _FakeConfig(SimpleNamespace):
        def update(self, kwargs: dict) -> None:  # type: ignore[override]
            updates.append(kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake = _FakeConfig(
        architectures=["BailingMoeV2_5ForCausalLM"],
        model_type="bailing_hybrid",
        num_nextn_predict_layers=3,
    )

    out = SpeculativeConfig.hf_config_override(fake)

    assert out is fake
    assert out.model_type == "bailing_mtp"
    assert out.architectures == ["BailingMTPModel"]
    assert out.n_predict == 3
    assert updates and updates[-1] == {
        "n_predict": 3,
        "architectures": ["BailingMTPModel"],
    }


# ---------------------------------------------------------------------------
# Weight-name rewrite tests
# ---------------------------------------------------------------------------


def _make_mtp_with_mock_decoder() -> BailingMultiTokenPredictorLayer:
    """Build a ``BailingMultiTokenPredictorLayer`` whose ``mtp_block`` and
    ``shared_head`` are ``MagicMock``s.  This is enough to exercise
    ``_rewrite_spec_layer_name`` (which only inspects string names).
    """
    layer = BailingMultiTokenPredictorLayer.__new__(BailingMultiTokenPredictorLayer)
    layer.mtp_block = mock.MagicMock()
    layer.shared_head = mock.MagicMock()
    layer.config = SimpleNamespace(num_nextn_predict_layers=1)
    return layer


def test_rewrite_spec_layer_name_renames_attention_to_self_attn() -> None:
    """Per-block attention weights get the ``mtp_block.`` prefix and
    ``attention`` is renamed to ``self_attn``.
    """
    layer = _make_mtp_with_mock_decoder()
    name = "model.layers.62.self_attn.q_a_proj.weight"
    new = BailingMTP._rewrite_spec_layer_name(layer, 62, name)
    assert new == "model.layers.62.mtp_block.self_attn.q_a_proj.weight"


def test_rewrite_spec_layer_name_renames_attention_dense_to_o_proj() -> None:
    """The MLA output projection is named ``attention.dense`` in the
    checkpoint but ``self_attn.o_proj`` in vLLM.
    """
    layer = _make_mtp_with_mock_decoder()
    name = "model.layers.62.self_attn.dense.weight"
    new = BailingMTP._rewrite_spec_layer_name(layer, 62, name)
    assert new == "model.layers.62.mtp_block.self_attn.o_proj.weight"


def test_rewrite_spec_layer_name_promotes_embed_tokens() -> None:
    """``embed_tokens`` is a shared weight: it should be lifted out of the
    per-layer prefix to the model-level prefix.
    """
    layer = _make_mtp_with_mock_decoder()
    name = "model.layers.62.embed_tokens.weight"
    new = BailingMTP._rewrite_spec_layer_name(layer, 62, name)
    assert new == "model.embed_tokens.weight"


def test_rewrite_spec_layer_name_keeps_mtp_heads() -> None:
    """MTP-specific head weights (enorm/hnorm/eh_proj/final_layernorm/
    shared_head) keep their original names and are not given the
    ``mtp_block.`` prefix.
    """
    layer = _make_mtp_with_mock_decoder()
    for head in ("enorm", "hnorm", "eh_proj", "final_layernorm", "shared_head"):
        name = f"model.layers.62.{head}.weight"
        new = BailingMTP._rewrite_spec_layer_name(layer, 62, name)
        assert new == name, f"unexpected rewrite for {head}: {new}"


# ---------------------------------------------------------------------------
# Sanity check: the predictor can be constructed with random tensors
# ---------------------------------------------------------------------------


def test_bailing_multi_token_predictor_layer_forward_shapes() -> None:
    """The forward method produces hidden states of the expected shape."""
    hidden_size = 16
    vocab_size = 32
    layer = BailingMultiTokenPredictorLayer.__new__(BailingMultiTokenPredictorLayer)
    layer.enorm = torch.nn.RMSNorm(hidden_size)
    layer.hnorm = torch.nn.RMSNorm(hidden_size)
    layer.eh_proj = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)
    layer.final_layernorm = torch.nn.RMSNorm(hidden_size)
    layer.mtp_block = mock.MagicMock()
    layer.mtp_block.return_value = (torch.zeros(2, hidden_size), None)
    layer.shared_head = mock.MagicMock()
    layer.config = SimpleNamespace(rms_norm_eps=1e-6)

    input_ids = torch.zeros(2, dtype=torch.long)
    positions = torch.zeros(2, dtype=torch.long)
    previous_hidden = torch.randn(2, hidden_size)
    inputs_embeds = torch.randn(2, hidden_size)

    out = layer(input_ids, positions, previous_hidden, inputs_embeds)
    assert out.shape == (2, hidden_size)
