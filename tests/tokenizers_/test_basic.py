# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path
from typing import _get_protocol_attrs  # type: ignore

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TokenizersBackend,
)

from vllm.tokenizers import TokenizerLike, get_tokenizer
from vllm.tokenizers.hf import HfTokenizer
from vllm.tokenizers.mistral import MistralTokenizer


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


def test_gte_qwen2_eos_token():
    """GTE pools the EOS token, which tokenization must append."""
    tokenizer = get_tokenizer(
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
    )
    prompt = "The chef prepared a delicious meal."
    text_ids = tokenizer.encode(prompt, add_special_tokens=False)

    assert tokenizer.encode(prompt) == text_ids + [tokenizer.eos_token_id]


@pytest.fixture
def gte_tokenizer_files(tmp_path: Path):
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "hello": 1, "world": 2, "[EOS]": 3}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    TokenizersBackend(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    ).save_pretrained(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config.update(
        tokenizer_class="Qwen2Tokenizer",
        auto_map={
            "AutoTokenizer": [
                "tokenization_qwen.Qwen2Tokenizer",
                "tokenization_qwen.Qwen2TokenizerFast",
            ]
        },
        add_eos_token=True,
    )
    config_path.write_text(json.dumps(config))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "is_causal": False})
    )
    (tmp_path / "tokenization_qwen.py").write_text(
        """from tokenizers import processors
from transformers import TokenizersBackend

class Qwen2Tokenizer(TokenizersBackend):
    pass

class Qwen2TokenizerFast(TokenizersBackend):
    slow_tokenizer_class = Qwen2Tokenizer
    padding_side = "left"
    truncation_side = "right"
    preserve_template = False

    @classmethod
    def convert_to_native_format(cls, **kwargs):
        return TokenizersBackend.convert_to_native_format(**kwargs)

    def __init__(self, *args, add_eos_token=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.truncation_side = "right"
        self._add_eos_token = add_eos_token
        if not self.preserve_template:
            self.update_post_processor()

    def update_post_processor(self):
        suffix = [f"{self.eos_token}:0"] if self.add_eos_token else []
        pair_suffix = [f"{self.eos_token}:1"] if self.add_eos_token else []
        self.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single=["$A:0", *suffix],
            pair=["$A:0", *suffix, "$B:1", *pair_suffix],
            special_tokens=[(self.eos_token, self.eos_token_id)] if suffix else [],
        )

    @property
    def add_eos_token(self):
        return self._add_eos_token
"""
    )
    return tmp_path


@pytest.mark.parametrize("declared_eos", [True, False])
@pytest.mark.parametrize("explicit_eos", [None, True, False])
@pytest.mark.parametrize("explicit_bos", [None, True, False])
def test_gte_legacy_eos(gte_tokenizer_files, declared_eos, explicit_eos, explicit_bos):
    config_path = gte_tokenizer_files / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["add_eos_token"] = declared_eos
    config_path.write_text(json.dumps(config))
    kwargs = {} if explicit_eos is None else {"add_eos_token": explicit_eos}
    if explicit_bos is not None:
        # The original GTE tokenizer ignores BOS, even when a token is supplied.
        kwargs.update(add_bos_token=explicit_bos, bos_token="[UNK]")
    tokenizer = get_tokenizer(gte_tokenizer_files, trust_remote_code=True, **kwargs)
    suffix = [3] if (declared_eos if explicit_eos is None else explicit_eos) else []

    assert tokenizer.padding_side == "left"
    assert tokenizer.encode("hello world") == [1, 2] + suffix
    assert tokenizer.encode("hello", text_pair="world") == [1] + suffix + [2] + suffix
    assert tokenizer.encode("hello world", add_special_tokens=False) == [1, 2]


def test_gte_legacy_auto_map(gte_tokenizer_files):
    config_path = gte_tokenizer_files / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["auto_map"] = config["auto_map"]["AutoTokenizer"]
    config_path.write_text(json.dumps(config))
    tokenizer = get_tokenizer(gte_tokenizer_files, trust_remote_code=True)
    assert tokenizer.encode("hello world") == [1, 2, 3]


@pytest.mark.parametrize("serialized_eos", [True, False])
def test_gte_preserves_serialized_template(gte_tokenizer_files, serialized_eos):
    path = gte_tokenizer_files / "tokenizer.json"
    backend = Tokenizer.from_file(str(path))
    backend.post_processor = processors.TemplateProcessing(
        single="$A [EOS]" if serialized_eos else "$A",
        special_tokens=[("[EOS]", 3)],
    )
    backend.save(str(path))
    tokenizer = get_tokenizer(gte_tokenizer_files, trust_remote_code=True)
    assert tokenizer.encode("hello world") == [1, 2] + ([3] if serialized_eos else [])


def test_gte_tokenizer_roundtrip(gte_tokenizer_files, tmp_path):
    tokenizer = get_tokenizer(gte_tokenizer_files, trust_remote_code=True)
    saved = tmp_path / "saved"
    tokenizer.save_pretrained(saved)
    assert "auto_map" in json.loads((saved / "tokenizer_config.json").read_text())
    (saved / "config.json").write_text(
        (gte_tokenizer_files / "config.json").read_text()
    )
    reloaded = get_tokenizer(saved, trust_remote_code=True)
    assert reloaded.encode("hello world") == [1, 2, 3]
    for side, expected in [("left", [2, 1, 3]), ("right", [1, 2, 3])]:
        reloaded.truncation_side = side
        assert (
            reloaded.encode("hello world hello", max_length=3, truncation=True)
            == expected
        )


@pytest.mark.parametrize("fixed_initializer", [False, True])
def test_gte_preserves_custom_initialization(gte_tokenizer_files, fixed_initializer):
    path = gte_tokenizer_files / "tokenization_qwen.py"
    source = path.read_text().replace(
        'truncation_side = "right"', 'truncation_side = "left"'
    )
    if fixed_initializer:
        source = source.replace("preserve_template = False", "preserve_template = True")
        tokenizer_path = gte_tokenizer_files / "tokenizer.json"
        backend = Tokenizer.from_file(str(tokenizer_path))
        backend.post_processor = processors.TemplateProcessing(
            single="$A:0 [EOS]:0",
            pair="$A:0 [EOS]:0 $B:1 [EOS]:1",
            special_tokens=[("[EOS]", 3)],
        )
        backend.save(str(tokenizer_path))
    path.write_text(source)
    tokenizer = get_tokenizer(
        gte_tokenizer_files, trust_remote_code=True, runner_type="pooling"
    )
    assert tokenizer.__class__.__mro__[1].__name__ == "Qwen2TokenizerFast"
    assert tokenizer.truncation_side == "left"
    assert tokenizer.encode("hello world hello", max_length=3, truncation=True) == [
        2,
        1,
        3,
    ]
    if fixed_initializer:
        reference = AutoTokenizer.from_pretrained(
            gte_tokenizer_files, trust_remote_code=True
        )
        assert reference.encode("hello world hello", max_length=3, truncation=True) == [
            2,
            1,
            3,
        ]
        assert (
            tokenizer.backend_tokenizer.to_str() == reference.backend_tokenizer.to_str()
        )
        assert tokenizer.init_kwargs["auto_map"] == reference.init_kwargs["auto_map"]
