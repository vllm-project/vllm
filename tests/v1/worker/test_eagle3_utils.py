# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression tests for EAGLE3 auxiliary hidden-state flag handling in MRV2.

Issue: the MRV2 model runner unconditionally set use_aux_hidden_state_outputs=True
for every eagle3 config, even when the draft head's eagle_config contains
"use_aux_hidden_state": False (e.g. nvidia/gpt-oss-120b-Eagle3-v2 style heads).
This caused the wrong input shape to be passed to combine_hidden_states, degrading
acceptance rates.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_use_aux_hidden_state_from_config,
)


def _make_spec_config(eagle_config=None, has_draft=True):
    """Build a minimal mock SpeculativeConfig."""
    hf_config = SimpleNamespace()
    if eagle_config is not None:
        hf_config.eagle_config = eagle_config

    draft_model_config = MagicMock()
    draft_model_config.hf_config = hf_config

    spec_config = MagicMock()
    spec_config.draft_model_config = draft_model_config if has_draft else None
    return spec_config


class TestGetEagle3UseAuxHiddenStateFromConfig:
    """Tests for get_eagle3_use_aux_hidden_state_from_config."""

    def test_returns_true_when_no_eagle_config(self):
        """Without eagle_config key, default is True (use aux hidden states)."""
        spec_config = _make_spec_config(eagle_config=None)
        assert get_eagle3_use_aux_hidden_state_from_config(spec_config) is True

    def test_returns_true_when_use_aux_hidden_state_key_missing(self):
        """eagle_config dict present but key absent -> default True."""
        spec_config = _make_spec_config(eagle_config={"some_other_key": 42})
        assert get_eagle3_use_aux_hidden_state_from_config(spec_config) is True

    def test_returns_true_when_explicitly_true(self):
        """eagle_config["use_aux_hidden_state"] = True -> True."""
        spec_config = _make_spec_config(
            eagle_config={"use_aux_hidden_state": True}
        )
        assert get_eagle3_use_aux_hidden_state_from_config(spec_config) is True

    def test_returns_false_when_explicitly_false(self):
        """eagle_config["use_aux_hidden_state"] = False -> False.

        This is the key regression case: models like nvidia/gpt-oss-120b-Eagle3-v2
        set this to False, signalling that no auxiliary hidden states should be
        requested from the target model.
        """
        spec_config = _make_spec_config(
            eagle_config={"use_aux_hidden_state": False}
        )
        assert get_eagle3_use_aux_hidden_state_from_config(spec_config) is False

    def test_returns_false_when_object_config_explicitly_false(self):
        """eagle_config as object with use_aux_hidden_state=False -> False.

        Some HF configs store eagle_config as a sub-config object rather than
        a plain dict.
        """
        eagle_obj = SimpleNamespace(use_aux_hidden_state=False)
        spec_config = _make_spec_config(eagle_config=eagle_obj)
        assert get_eagle3_use_aux_hidden_state_from_config(spec_config) is False

    def test_returns_true_when_object_config_explicitly_true(self):
        """eagle_config as object with use_aux_hidden_state=True -> True."""
        eagle_obj = SimpleNamespace(use_aux_hidden_state=True)
        spec_config = _make_spec_config(eagle_config=eagle_obj)
        assert get_eagle3_use_aux_hidden_state_from_config(spec_config) is True

    def test_returns_true_when_no_draft_model_config(self):
        """No draft_model_config -> safe default True."""
        spec_config = _make_spec_config(has_draft=False)
        assert get_eagle3_use_aux_hidden_state_from_config(spec_config) is True

    def test_returns_true_when_spec_config_none(self):
        """None spec_config -> safe default True."""
        assert get_eagle3_use_aux_hidden_state_from_config(None) is True
