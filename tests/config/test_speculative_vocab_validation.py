# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import get_args
from unittest import mock

import pytest

from vllm.config.speculative import MTPModelTypes, SpeculativeConfig


def _create_spec_config_with_method(method, target_vocab, draft_vocab):
    """Create a SpeculativeConfig with mocked model configs for vocab testing."""
    spec_cfg = SpeculativeConfig.__new__(SpeculativeConfig)
    spec_cfg.method = method

    target_mock = mock.Mock()
    target_mock.get_vocab_size.return_value = target_vocab
    draft_mock = mock.Mock()
    draft_mock.get_vocab_size.return_value = draft_vocab

    spec_cfg.target_model_config = target_mock
    spec_cfg.draft_model_config = draft_mock
    return spec_cfg


class TestVerifyEqualVocabSize:
    """Tests for verify_equal_vocab_size covering all speculative methods."""

    def test_draft_model_matching_vocab_passes(self):
        cfg = _create_spec_config_with_method("draft_model", 32000, 32000)
        cfg.verify_equal_vocab_size()  # Should not raise

    def test_draft_model_mismatched_vocab_raises(self):
        cfg = _create_spec_config_with_method("draft_model", 32000, 64000)
        with pytest.raises(ValueError, match="same vocabulary size"):
            cfg.verify_equal_vocab_size()

    def test_eagle_matching_vocab_passes(self):
        cfg = _create_spec_config_with_method("eagle", 32000, 32000)
        cfg.verify_equal_vocab_size()  # Should not raise

    def test_eagle_mismatched_vocab_raises(self):
        cfg = _create_spec_config_with_method("eagle", 32000, 999999)
        with pytest.raises(ValueError, match="same vocabulary size"):
            cfg.verify_equal_vocab_size()

    def test_eagle3_mismatched_vocab_raises(self):
        cfg = _create_spec_config_with_method("eagle3", 32000, 16000)
        with pytest.raises(ValueError, match="same vocabulary size"):
            cfg.verify_equal_vocab_size()

    def test_mtp_matching_vocab_passes(self):
        cfg = _create_spec_config_with_method("mtp", 32000, 32000)
        cfg.verify_equal_vocab_size()  # Should not raise

    def test_mtp_mismatched_vocab_raises(self):
        cfg = _create_spec_config_with_method("mtp", 1000, 999999)
        with pytest.raises(ValueError, match="out-of-bounds"):
            cfg.verify_equal_vocab_size()

    @pytest.mark.parametrize("mtp_type", list(get_args(MTPModelTypes)))
    def test_all_mtp_variants_validated(self, mtp_type):
        cfg = _create_spec_config_with_method(mtp_type, 32000, 64000)
        with pytest.raises(ValueError, match="same vocabulary size"):
            cfg.verify_equal_vocab_size()

    def test_ngram_not_validated(self):
        """N-gram method uses target model config directly, no separate draft."""
        cfg = _create_spec_config_with_method("ngram", 32000, 64000)
        cfg.verify_equal_vocab_size()  # Should not raise

    def test_suffix_not_validated(self):
        cfg = _create_spec_config_with_method("suffix", 32000, 64000)
        cfg.verify_equal_vocab_size()  # Should not raise

    def test_none_configs_skip_validation(self):
        cfg = _create_spec_config_with_method("eagle", 32000, 64000)
        cfg.target_model_config = None
        cfg.verify_equal_vocab_size()  # Should not raise

        cfg2 = _create_spec_config_with_method("eagle", 32000, 64000)
        cfg2.draft_model_config = None
        cfg2.verify_equal_vocab_size()  # Should not raise

    def test_deprecated_method_delegates(self):
        """Old method delegates to verify_equal_vocab_size."""
        cfg = _create_spec_config_with_method("draft_model", 32000, 64000)
        with pytest.raises(ValueError, match="same vocabulary size"):
            cfg.verify_equal_vocab_size_if_draft_model()
