# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import (
    CachedHfTokenizer,
    _load_tokenizers_backend_fast_fallback,
    get_cached_tokenizer,
)


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


class TestTokenizersBackendFastFallback:
    """Tests for the TokenizersBackendFast fallback logic (issue #36443)."""

    TOKENIZERS_BACKEND_FAST_ERROR = ValueError(
        "Tokenizer class TokenizersBackendFast does not exist "
        "or is not currently imported."
    )

    def test_fallback_tries_tokenizers_backend_fast_import(self):
        """When AutoTokenizer fails to resolve TokenizersBackendFast,
        the fallback should try importing it from transformers."""
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerFast)

        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_tokenizer

        with patch("vllm.tokenizers.hf.PreTrainedTokenizerFast") as mock_ptf:
            fake_module = MagicMock()
            fake_module.TokenizersBackendFast = mock_cls
            with patch.dict(
                "sys.modules",
                {"transformers": fake_module},
            ):
                result = _load_tokenizers_backend_fast_fallback("fake/model")

            assert result == mock_tokenizer
            mock_cls.from_pretrained.assert_called_once_with("fake/model")
            mock_ptf.from_pretrained.assert_not_called()

    def test_fallback_to_pretrained_tokenizer_fast(self):
        """When TokenizersBackendFast is not importable, fall back to
        PreTrainedTokenizerFast."""
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerFast)

        with patch("vllm.tokenizers.hf.PreTrainedTokenizerFast") as mock_ptf:
            mock_ptf.from_pretrained.return_value = mock_tokenizer

            # Use a fake transformers module without TokenizersBackendFast
            # so that `from transformers import TokenizersBackendFast` raises
            # ImportError inside the fallback function.
            fake_module = MagicMock(spec=["__name__"])
            with patch.dict("sys.modules", {"transformers": fake_module}):
                result = _load_tokenizers_backend_fast_fallback("fake/model")

            assert result == mock_tokenizer
            mock_ptf.from_pretrained.assert_called_once_with("fake/model")

    def test_cached_hf_tokenizer_catches_tokenizers_backend_fast_error(self):
        """CachedHfTokenizer.from_pretrained should catch the
        TokenizersBackendFast ValueError and invoke the fallback."""
        mock_tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
        mock_tokenizer.all_special_ids = []
        mock_tokenizer.all_special_tokens = []
        mock_tokenizer.get_vocab.return_value = {"a": 0}
        mock_tokenizer.__len__ = lambda self: 1
        mock_tokenizer.__class__ = PreTrainedTokenizerFast

        with (
            patch(
                "vllm.tokenizers.hf.AutoTokenizer.from_pretrained",
                side_effect=self.TOKENIZERS_BACKEND_FAST_ERROR,
            ),
            patch(
                "vllm.tokenizers.hf._load_tokenizers_backend_fast_fallback",
                return_value=mock_tokenizer,
            ) as mock_fallback,
            patch(
                "vllm.tokenizers.hf.get_sentence_transformer_tokenizer_config",
                return_value=None,
            ),
        ):
            result = CachedHfTokenizer.from_pretrained("fake/model")

            mock_fallback.assert_called_once()
            assert result is not None

    def test_other_value_errors_still_raise(self):
        """Non-TokenizersBackendFast ValueErrors should propagate normally."""
        other_error = ValueError("some other tokenizer error")

        with (
            patch(
                "vllm.tokenizers.hf.AutoTokenizer.from_pretrained",
                side_effect=other_error,
            ),
            pytest.raises(ValueError, match="some other tokenizer error"),
        ):
            CachedHfTokenizer.from_pretrained("fake/model")

    def test_unknown_class_error_suggests_trust_remote_code(self):
        """Unknown tokenizer class errors should still suggest
        --trust-remote-code when trust_remote_code=False."""
        unknown_error = ValueError(
            "Tokenizer class MyCustomTokenizer does not exist "
            "or is not currently imported."
        )

        with (
            patch(
                "vllm.tokenizers.hf.AutoTokenizer.from_pretrained",
                side_effect=unknown_error,
            ),
            pytest.raises(RuntimeError, match="trust_remote_code"),
        ):
            CachedHfTokenizer.from_pretrained("fake/model", trust_remote_code=False)
