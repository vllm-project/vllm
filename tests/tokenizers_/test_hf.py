# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from copy import deepcopy
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import CachedHfTokenizer, get_cached_tokenizer


@pytest.mark.parametrize("model_id", ["gpt2", "zai-org/chatglm3-6b"])
def test_cached_tokenizer(model_id: str):
    reference_tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    reference_tokenizer.add_special_tokens({"cls_token": "<CLS>"})
    reference_tokenizer.add_special_tokens({"additional_special_tokens": ["<SEP>"]})

    cached_tokenizer = get_cached_tokenizer(deepcopy(reference_tokenizer))
    _check_consistency(cached_tokenizer, reference_tokenizer)

    pickled_tokenizer = pickle.dumps(cached_tokenizer)
    unpickled_tokenizer = pickle.loads(pickled_tokenizer)
    _check_consistency(unpickled_tokenizer, reference_tokenizer)


def _check_consistency(target: TokenizerLike, expected: TokenizerLike):
    assert isinstance(target, type(expected))

    # Cached attributes
    assert target.all_special_ids == expected.all_special_ids
    assert target.all_special_tokens == expected.all_special_tokens
    assert target.get_vocab() == expected.get_vocab()
    assert len(target) == len(expected)

    # Other attributes
    assert getattr(target, "padding_side", None) == getattr(
        expected, "padding_side", None
    )

    assert target.encode("prompt") == expected.encode("prompt")


# ---------------------------------------------------------------------------
# Tests for improved tokenizer error messages (Issue #38024)
# ---------------------------------------------------------------------------


class TestTokenizerErrorMessages:
    @patch("vllm.tokenizers.hf.AutoTokenizer")
    def test_error_message_suggests_upgrade_transformers(self, mock_auto):
        """When the tokenizer class is not found, the error message should
        suggest upgrading transformers in addition to --trust-remote-code."""
        mock_auto.from_pretrained.side_effect = ValueError(
            "Tokenizer class TokenizersBackend does not exist "
            "or is not currently imported."
        )

        with pytest.raises(RuntimeError, match="trust_remote_code") as exc_info:
            CachedHfTokenizer.from_pretrained("test-model")

        assert "pip install --upgrade transformers" in str(exc_info.value)

    @patch("vllm.tokenizers.hf.AutoTokenizer")
    def test_error_message_on_execute_tokenizer_file(self, mock_auto):
        """The upgrade hint should also appear for the 'requires you to
        execute the tokenizer file' error variant."""
        mock_auto.from_pretrained.side_effect = ValueError(
            "CustomTok requires you to execute the tokenizer file"
        )

        with pytest.raises(RuntimeError, match="upgrade") as exc_info:
            CachedHfTokenizer.from_pretrained("test-model")

        error_msg = str(exc_info.value)
        assert "trust_remote_code" in error_msg
        assert "pip install --upgrade transformers" in error_msg

    @patch("vllm.tokenizers.hf.AutoTokenizer")
    def test_unrelated_valueerror_is_reraised(self, mock_auto):
        """ValueError messages that are NOT about missing tokenizer classes
        should propagate unchanged."""
        mock_auto.from_pretrained.side_effect = ValueError(
            "Some completely different error"
        )

        with pytest.raises(ValueError, match="completely different"):
            CachedHfTokenizer.from_pretrained("test-model")
