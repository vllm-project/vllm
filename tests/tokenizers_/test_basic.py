# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path
from typing import _get_protocol_attrs  # type: ignore

import pytest
from tokenizers import Tokenizer
from tokenizers import decoders as tokenizers_decoders
from tokenizers import models as tokenizers_models
from tokenizers import pre_tokenizers as tokenizers_pre_tokenizers
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.grok2 import Grok2Tokenizer
from vllm.tokenizers.hf import CachedHfTokenizer, HfTokenizer
from vllm.tokenizers.mistral import MistralTokenizer

BYTELEVEL_SPACE = "\u0120"
BYTELEVEL_NEWLINE = "\u010a"


def _get_missing_attrs(obj: object, target: type):
    return [k for k in _get_protocol_attrs(target) if not hasattr(obj, k)]


def _assert_tokenizer_like(tokenizer: object):
    missing_attrs = _get_missing_attrs(tokenizer, TokenizerLike)
    assert not missing_attrs, f"Missing attrs: {missing_attrs}"


def test_tokenizer_like_protocol():
    tokenizer = get_tokenizer("gpt2", use_fast=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    _assert_tokenizer_like(tokenizer)

    tokenizer = get_tokenizer(
        "mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_mode="mistral",
    )
    assert isinstance(tokenizer, MistralTokenizer)
    _assert_tokenizer_like(tokenizer)

    tokenizer = get_tokenizer("xai-org/grok-2", tokenizer_mode="grok2")
    assert isinstance(tokenizer, Grok2Tokenizer)
    _assert_tokenizer_like(tokenizer)

    tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3", tokenizer_mode="deepseek_v32")
    assert isinstance(tokenizer, HfTokenizer)

    # Verify it's a fast tokenizer (required for FastIncrementalDetokenizer)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert "DSV32" in tokenizer.__class__.__name__
    _assert_tokenizer_like(tokenizer)

    tokenizer = get_tokenizer(
        "Qwen/Qwen-VL",
        tokenizer_mode="qwen_vl",
        trust_remote_code=True,
    )
    assert isinstance(tokenizer, HfTokenizer)
    assert "WithoutImagePad" in tokenizer.__class__.__name__


def test_deepseek_v3_ignores_bad_tokenizer_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    tokenizer = Tokenizer(
        tokenizers_models.BPE(
            vocab={
                "<unk>": 0,
                "According": 1,
                f"{BYTELEVEL_SPACE}to": 2,
                BYTELEVEL_NEWLINE: 3,
                f"{BYTELEVEL_SPACE}US": 4,
            },
            merges=[],
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = tokenizers_pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = tokenizers_decoders.ByteLevel()
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    # This metadata is inconsistent with the byte-level tokenizer JSON. Loading
    # through LlamaTokenizerFast can surface byte-level marker glyphs in
    # decoded text.
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizerFast",
                "unk_token": "<unk>",
            }
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v3",
                "architectures": ["DeepseekV3ForCausalLM"],
            }
        )
    )

    def fail_if_hf_tokenizer_is_used(*args, **kwargs):
        raise AssertionError("deepseek_v3 should bypass tokenizer_class metadata")

    monkeypatch.setattr(
        CachedHfTokenizer, "from_pretrained", fail_if_hf_tokenizer_is_used
    )

    tokenizer = get_tokenizer(str(tmp_path), trust_remote_code=True)

    assert isinstance(tokenizer, TokenizersBackend)
    assert tokenizer.convert_ids_to_tokens([1, 2, 3, 4]) == [
        "According",
        f"{BYTELEVEL_SPACE}to",
        BYTELEVEL_NEWLINE,
        f"{BYTELEVEL_SPACE}US",
    ]
    assert tokenizer.decode([1, 2, 3, 4]) == "According to\n US"


@pytest.mark.parametrize("tokenizer_name", ["facebook/opt-125m", "gpt2"])
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
