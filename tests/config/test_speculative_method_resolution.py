# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for speculative method resolution and validation.

An explicitly configured `method` must never be silently overridden by
checkpoint auto-detection, and mismatches must fail with actionable errors.
See https://github.com/vllm-project/vllm/issues/47486.
"""

from unittest.mock import Mock

import pytest
from transformers import PretrainedConfig

from vllm.config import SpeculativeConfig
from vllm.transformers_utils.configs.eagle import EAGLEConfig


def _spec_cfg(
    method,
    model="org/some-draft",
    model_type="qwen2",
    architectures=("Qwen2ForCausalLM",),
    num_speculative_tokens=3,
):
    cfg = SpeculativeConfig.__new__(SpeculativeConfig)
    cfg.method = method
    cfg.num_speculative_tokens = num_speculative_tokens
    draft = Mock()
    draft.model = model
    draft.architectures = list(architectures)
    draft.hf_config = Mock(spec=["model_type", "architectures"])
    draft.hf_config.model_type = model_type
    draft.hf_config.architectures = list(architectures)
    cfg.draft_model_config = draft
    return cfg


class TestDetection:
    def test_name_hint_precedes_structural_matching_legacy_order(self):
        # Legacy detection order is name-hints-first and is load-bearing:
        # DeepSeek EAGLE heads are structurally MTP-typed and rely on the
        # "eagle-" name hint to route to the eagle path.
        cfg = _spec_cfg(
            None,
            model="eagle618/eagle-deepseek-v3-random",
            model_type="deepseek_mtp",
        )
        assert cfg._detect_draft_method() == "eagle"

    def test_mtp_from_model_type(self):
        cfg = _spec_cfg(None, model_type="gemma4_mtp")
        assert cfg._detect_draft_method() == "mtp"

    def test_eagle_from_name_hint(self):
        cfg = _spec_cfg(None, model="yuhuili/EAGLE-LLaMA3-Instruct-8B")
        assert cfg._detect_draft_method() == "eagle"

    def test_plain_causal_lm_is_undetected(self):
        assert _spec_cfg(None)._detect_draft_method() is None


class TestExplicitMethodIsNeverOverridden:
    def test_dir_name_does_not_hijack_draft_model(self):
        # GH #47486 bug 1: a path containing "eagle-" must not reroute an
        # explicit method="draft_model" onto the eagle path.
        cfg = _spec_cfg("draft_model", model="/models/my-eagle-draft")
        cfg._resolve_draft_method(method_was_explicit=True)
        assert cfg.method == "draft_model"

    def test_explicit_medusa_with_mtp_checkpoint_raises(self):
        # GH #47486 bug 2: previously silently converted to method="mtp".
        cfg = _spec_cfg("medusa", model_type="gemma4_mtp")
        with pytest.raises(ValueError, match="looks like a 'mtp' checkpoint"):
            cfg._resolve_draft_method(method_was_explicit=True)

    def test_explicit_draft_model_with_mtp_checkpoint_raises(self):
        cfg = _spec_cfg("draft_model", model_type="deepseek_mtp")
        with pytest.raises(ValueError, match="Use method='mtp'"):
            cfg._resolve_draft_method(method_was_explicit=True)

    def test_explicit_medusa_with_plain_checkpoint_names_the_checkpoint(self):
        # GH #47486 bug 3: previously "Unsupported speculative method:
        # 'medusa'", blaming the (supported) method instead of the checkpoint.
        cfg = _spec_cfg("medusa")
        with pytest.raises(ValueError, match="not recognized as any drafter"):
            cfg._resolve_draft_method(method_was_explicit=True)

    def test_explicit_eagle_with_mtp_typed_checkpoint_is_trusted(self):
        # DeepSeek EAGLE heads legitimately carry an MTP model_type; explicit
        # eagle must pass through (EAGLEConfig validates the config itself,
        # e.g. rejecting vocab_size-less non-eagle heads actionably).
        cfg = _spec_cfg("eagle", model="/models/ds-head", model_type="deepseek_mtp")
        cfg._resolve_draft_method(method_was_explicit=True)
        assert cfg.method == "eagle"

    def test_explicit_eagle_with_medusa_checkpoint_raises(self):
        cfg = _spec_cfg("eagle", model_type="medusa")
        with pytest.raises(ValueError, match="looks like a 'medusa'"):
            cfg._resolve_draft_method(method_was_explicit=True)

    def test_explicit_eagle_with_plain_checkpoint_is_trusted(self):
        # Eagle checkpoints are often structurally undetectable; an explicit
        # eagle method with a non-drafter-typed config must pass through
        # (downstream registry/EAGLEConfig checks handle invalid ones).
        cfg = _spec_cfg("eagle")
        cfg._resolve_draft_method(method_was_explicit=True)
        assert cfg.method == "eagle"


class TestAutoDetectionStillWorks:
    def test_defaulted_method_adopts_mtp(self):
        cfg = _spec_cfg("draft_model", model_type="gemma4_mtp")
        cfg._resolve_draft_method(method_was_explicit=False)
        assert cfg.method == "mtp"

    def test_defaulted_method_adopts_eagle_name_hint(self):
        cfg = _spec_cfg("draft_model", model="yuhuili/EAGLE-LLaMA3-Instruct-8B")
        cfg._resolve_draft_method(method_was_explicit=False)
        assert cfg.method == "eagle"

    def test_defaulted_method_keeps_draft_model_for_plain_lm(self):
        cfg = _spec_cfg("draft_model")
        cfg._resolve_draft_method(method_was_explicit=False)
        assert cfg.method == "draft_model"


class TestEagleConfigGuard:
    def test_missing_vocab_size_raises_actionable_error(self):
        # GH #47486 bug 4: previously a raw AttributeError.
        model = Mock(spec=["architectures", "model_type"])
        model.architectures = ["Gemma4AssistantForCausalLM"]
        model.model_type = "gemma4_assistant"
        with pytest.raises(ValueError, match="vocab_size"):
            EAGLEConfig(model=model, method="eagle")

    def test_vocab_size_present_is_accepted(self):
        model = PretrainedConfig(vocab_size=32000, architectures=["LlamaForCausalLM"])
        cfg = EAGLEConfig(model=model, method="eagle")
        assert cfg.truncated_vocab_size == 32000
