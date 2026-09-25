# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import _get_protocol_attrs  # type: ignore

import pybase64
import pytest
from transformers import (
    PreTrainedTokenizerBase,
    TokenizersBackend,
)

from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.hf import HfTokenizer
from vllm.tokenizers.mistral import MistralTokenizer


@pytest.mark.parametrize("side", [None, "left"])
def test_kimi_audio_padding_side(tmp_path, side):
    """Kimi Audio honors explicit left padding and keeps its right default."""
    from vllm.renderers import TokenizeParams
    from vllm.tokenizers.kimi_audio import KimiAudioTokenizer

    (tmp_path / "tiktoken.model").write_text(
        "".join(f"{pybase64.b64encode(bytes([i])).decode()} {i}\n" for i in range(256))
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {"151644": {"content": "<|endoftext|>"}}})
    )
    kwargs = {} if side is None else {"padding_side": side}
    tokenizer = KimiAudioTokenizer.from_pretrained(tmp_path, **kwargs)
    tokens = tokenizer.encode("hi", add_special_tokens=False)
    result = TokenizeParams(
        max_total_tokens=8, pad_prompt_tokens=4
    ).apply_post_tokenization(tokenizer, {"prompt_token_ids": tokens})
    padding = [tokenizer.pad_token_id] * (4 - len(tokens))
    assert result["prompt_token_ids"] == (
        padding + tokens if side == "left" else tokens + padding
    )


def _get_missing_attrs(obj: object, target: type):
    return [k for k in _get_protocol_attrs(target) if not hasattr(obj, k)]


def _assert_tokenizer_like(tokenizer: object):
    missing_attrs = _get_missing_attrs(tokenizer, TokenizerLike)
    assert not missing_attrs, f"Missing attrs: {missing_attrs}"


def test_tokenizer_like_protocol():
    tokenizer = get_tokenizer("openai-community/gpt2")
    assert isinstance(tokenizer, TokenizersBackend)
    _assert_tokenizer_like(tokenizer)

    tokenizer = get_tokenizer(
        "mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_mode="mistral",
    )
    assert isinstance(tokenizer, MistralTokenizer)
    _assert_tokenizer_like(tokenizer)

    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3", tokenizer_mode="deepseek_v32")
    assert isinstance(tokenizer, HfTokenizer)

    # Verify it's a fast tokenizer (required for FastIncrementalDetokenizer)
    assert isinstance(tokenizer, TokenizersBackend)
    assert "DSV32" in tokenizer.__class__.__name__
    _assert_tokenizer_like(tokenizer)


@pytest.mark.parametrize(
    "tokenizer_name", ["facebook/opt-125m", "openai-community/gpt2"]
)
def test_tokenizer_revision(tokenizer_name: str):
    # Assume that "main" branch always exists
    tokenizer = get_tokenizer(tokenizer_name, revision="main")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # Assume that "never" branch always does not exist
    with pytest.raises(OSError, match="not a valid git identifier"):
        get_tokenizer(tokenizer_name, revision="never")


@pytest.mark.parametrize("tokenizer_name", ["BAAI/bge-base-en"])
@pytest.mark.parametrize("n_tokens", [510])
def test_special_tokens(tokenizer_name: str, n_tokens: int):
    tokenizer = get_tokenizer(tokenizer_name, revision="main")

    prompts = "[UNK]" * n_tokens
    prompt_token_ids = tokenizer.encode(prompts)
    assert len(prompt_token_ids) == n_tokens + 2
