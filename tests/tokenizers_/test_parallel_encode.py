# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that a long prompt tokenized in parallel pieces gets the same token
IDs as the normal tokenizer call, and that other cases use the normal call."""

import copy
import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pytest
from tokenizers import Tokenizer, pre_tokenizers
from transformers import TokenizersBackend

from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.hf import maybe_make_thread_pool
from vllm.tokenizers.parallel_encode import (
    _CUT_PATTERN,
    MIN_PARALLEL_ENCODE_CHARS,
    _cut_into_pieces,
    _encode_pieces,
    try_parallel_encode,
)

# Qwen3 uses the pre-tokenizer regex of Qwen2 to Qwen3. Qwen3.5 uses the regex
# that also puts combining marks into words.
QWEN_TOKENIZERS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]

# The keyword arguments that `TokenizeParams.get_encode_kwargs` gives when the
# prompt is not truncated.
NO_TRUNCATION = {"truncation": False, "max_length": None, "add_special_tokens": True}

# Words and snippets for the test text. They put cut places next to special
# tokens, digits, punctuation, CJK and other scripts, emojis, combining marks
# and line breaks, and they add places that look like cut places but are not
# ones, such as a letter after a tab, a no-break space or a next-line char.
WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "A", "I",
    "x", "vLLM", "Qwen", "tokenizer", "prompt", "don't", "it's", "we'll",
    "I'm", "they're", "café", "naïve", "über", "Straße", "résumé", "ﬁle",
    "e\u0301t\u0301e\u0301", "0", "7", "42", "3.14159", "2026-10-14", "1,000,000",
    "12345678901234567890", "0x1F", "v1.2.3", "x86_64", "GPU0",
]  # fmt: skip
SNIPPETS = [
    "\n", "\r\n", "\r", "\n\n", "\n\n\n", "\t", "  ", "   ", " \n", "\n ",
    "\n\t", "\t\n", " \r\n ", ".", ",", "!", "?", "...", " - ", "--", "(", ")",
    "[x]", '{"a": 1, "b": [2, 3]}', " #", " //", "'s", "'ll", " '", "\"",
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<|im_start", "im_end|>", "<|im_start|>user\n",
    "<|im_end|>\n<|im_start|>assistant\n", "中文分词测试", " 日本語のテキスト",
    "한국어 문장", " مرحبا", " שלום", " Ελληνικά", " Привет мир", "😀", " 👍🏽",
    "👨‍👩‍👧‍👦", " 🇫🇮", "\u0301", " \u0301x", "\u00a0A", "\u3000B", "\u2028C",
    "\x85D", "\x0bE", "\tF", "def f(x):\n    return x + 1\n",
    "for i in range(10):\n\tprint(i)\n", "https://example.com/a?b=c&d=1",
    "\n1. First\n2. Second\n", "\nThe end.\n",
]  # fmt: skip


def make_text(num_chars: int, seed: int = 0) -> str:
    """Return a random text of about `num_chars` characters."""
    rng = random.Random(seed)
    parts: list[str] = []
    length = 0
    while length < num_chars:
        part = " " + rng.choice(WORDS) if rng.random() < 0.7 else rng.choice(SNIPPETS)
        parts.append(part)
        length += len(part)
    return "".join(parts)


@pytest.fixture(scope="module")
def long_text() -> str:
    return make_text(3_000_000)


_loaded: dict[tuple[str, str], TokenizersBackend] = {}


def load(name: str, truncation_side: str = "left") -> TokenizersBackend:
    """Load a tokenizer the way vLLM does, once per name and side."""
    key = (name, truncation_side)
    if key not in _loaded:
        _loaded[key] = get_tokenizer(name, truncation_side=truncation_side)
    return _loaded[key]


@pytest.mark.parametrize("name", QWEN_TOKENIZERS)
def test_long_text_matches_normal_call(long_text: str, name: str):
    tokenizer = load(name)
    # The text must really be cut, or the comparison below proves nothing.
    assert len(_cut_into_pieces(long_text)) > 40
    expected = tokenizer(long_text, **NO_TRUNCATION)["input_ids"]
    assert try_parallel_encode(tokenizer, long_text, NO_TRUNCATION) == expected


@pytest.mark.parametrize("name", QWEN_TOKENIZERS)
def test_every_cut_place_is_a_token_boundary(name: str):
    # Pieces of at least `PARALLEL_ENCODE_PIECE_CHARS` characters test only a
    # few cut places. Cutting at all of them tests tens of thousands.
    tokenizer = load(name)
    text = make_text(MIN_PARALLEL_ENCODE_CHARS, seed=1)
    cuts = [match.start() for match in _CUT_PATTERN.finditer(text)]
    assert len(cuts) > 20_000
    pieces = [text[a:b] for a, b in itertools.pairwise([0, *cuts, len(text)])]
    expected = tokenizer(text, **NO_TRUNCATION)["input_ids"]
    assert _encode_pieces(tokenizer, pieces) == expected


@pytest.mark.parametrize("name", QWEN_TOKENIZERS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_truncation_matches_normal_call(long_text: str, name: str, side: str):
    tokenizer = load(name, truncation_side=side)
    text = long_text[: 4 * MIN_PARALLEL_ENCODE_CHARS]
    full = tokenizer(text, **NO_TRUNCATION)["input_ids"]

    kwargs = {"truncation": True, "max_length": len(full) // 3}
    expected = tokenizer(text, **kwargs)["input_ids"]
    assert len(expected) == len(full) // 3
    assert try_parallel_encode(tokenizer, text, kwargs) == expected

    # The normal call also truncates when only `max_length` is given.
    kwargs = {"max_length": len(full) // 3}
    assert tokenizer(text, **kwargs)["input_ids"] == expected
    assert try_parallel_encode(tokenizer, text, kwargs) == expected

    # With truncation turned off, `max_length` does nothing.
    kwargs = {"truncation": False, "max_length": len(full) // 3}
    assert try_parallel_encode(tokenizer, text, kwargs) == full

    # vLLM passes one token more than the limit, so that a prompt that is too
    # long still fails the length check. A prompt within the limit is kept.
    kwargs = {"truncation": True, "max_length": len(full) + 1}
    assert try_parallel_encode(tokenizer, text, kwargs) == full


def test_thread_pool_tokenizer_in_threads(long_text: str):
    # HfRenderer wraps its tokenizer with `maybe_make_thread_pool` and calls
    # it from several threads at the same time.
    tokenizer = maybe_make_thread_pool(copy.deepcopy(load(QWEN_TOKENIZERS[0])), 2)
    text = long_text[: 4 * MIN_PARALLEL_ENCODE_CHARS]
    expected = tokenizer(text, **NO_TRUNCATION)["input_ids"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: try_parallel_encode(tokenizer, text, NO_TRUNCATION),
                range(8),
            )
        )
    assert all(result == expected for result in results)


def test_normal_call_for_short_text_or_no_cut_place():
    tokenizer = load(QWEN_TOKENIZERS[0])

    short_text = make_text(MIN_PARALLEL_ENCODE_CHARS - 1000)
    assert len(short_text) < MIN_PARALLEL_ENCODE_CHARS
    assert try_parallel_encode(tokenizer, short_text, NO_TRUNCATION) is None

    # CJK text has no space or line break in front of an ASCII letter or
    # digit, and the only cut place below is too early to make two pieces.
    for text in (
        "中文分词测试" * MIN_PARALLEL_ENCODE_CHARS,
        "x" * MIN_PARALLEL_ENCODE_CHARS,
        "a b" + "中" * MIN_PARALLEL_ENCODE_CHARS,
    ):
        assert try_parallel_encode(tokenizer, text, NO_TRUNCATION) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {**NO_TRUNCATION, "return_offsets_mapping": True},
        {**NO_TRUNCATION, "padding": "max_length"},
        {"truncation": "only_first", "max_length": 10},
        {"truncation": True, "max_length": None},
        {"truncation": True, "max_length": 0},
    ],
)
def test_normal_call_for_unsupported_arguments(kwargs: dict):
    tokenizer = load(QWEN_TOKENIZERS[0])
    text = make_text(MIN_PARALLEL_ENCODE_CHARS + 10_000)
    assert try_parallel_encode(tokenizer, text, kwargs) is None


def _with_config(name: str, change) -> TokenizersBackend:
    """Return a tokenizer built from the Rust configuration of `name` after
    `change` edits the configuration in place."""
    config = json.loads(load(name).backend_tokenizer.to_str())
    change(config)
    return TokenizersBackend(
        tokenizer_object=Tokenizer.from_str(json.dumps(config)),
        truncation_side="left",
    )


def _set_regex(regex: str):
    def change(config):
        config["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = regex

    return change


def _set(path: tuple, value):
    def change(config):
        node = config
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return change


def _add_token(**fields):
    def change(config):
        token = {
            "id": len(config["model"]["vocab"]) + 1000,
            "content": "<|extra|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        config["added_tokens"].append({**token, **fields})

    return change


def test_tokenizer_from_unchanged_config_is_supported():
    # The tests below change one field of this configuration. Without the
    # change, the tokenizer must be supported.
    tokenizer = _with_config(QWEN_TOKENIZERS[0], lambda config: None)
    text = make_text(MIN_PARALLEL_ENCODE_CHARS + 10_000)
    token_ids = try_parallel_encode(tokenizer, text, NO_TRUNCATION)
    assert token_ids is not None
    assert token_ids == tokenizer(text, **NO_TRUNCATION)["input_ids"]


@pytest.mark.parametrize(
    "change",
    [
        # The regex of Llama 3 groups digits by three. It is not in the list.
        _set_regex(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ),
        _set(("normalizer",), {"type": "Lowercase"}),
        _set(("pre_tokenizer", "pretokenizers", 1, "add_prefix_space"), True),
        _set(("model", "dropout"), 0.1),
        _set(
            ("post_processor",),
            {
                "type": "BertProcessing",
                "cls": ["<|endoftext|>", 151643],
                "sep": ["<|endoftext|>", 151643],
            },
        ),
        _add_token(lstrip=True),
        _add_token(single_word=True),
        # A cut place inside the content of an added token.
        _add_token(content="<|tool call|>"),
        _add_token(content="<|line|>\nA"),
    ],
    ids=[
        "llama3_regex",
        "lowercase_normalizer",
        "add_prefix_space",
        "bpe_dropout",
        "bert_post_processor",
        "lstrip_token",
        "single_word_token",
        "token_with_space",
        "token_with_line_break",
    ],
)
def test_normal_call_for_unsupported_config(change):
    tokenizer = _with_config(QWEN_TOKENIZERS[0], change)
    text = make_text(MIN_PARALLEL_ENCODE_CHARS + 10_000)
    assert try_parallel_encode(tokenizer, text, NO_TRUNCATION) is None


class _TokenInPythonTokenizer(TokenizersBackend):
    """A tokenizer class that adds a token in Python, which the Rust
    configuration does not show."""

    def __call__(self, text, *args, add_special_tokens=True, **kwargs):
        encoding = super().__call__(
            text, *args, add_special_tokens=add_special_tokens, **kwargs
        )
        if add_special_tokens and isinstance(text, str):
            encoding["input_ids"] = [0, *encoding["input_ids"]]
        return encoding


def test_normal_call_for_tokens_added_in_python():
    config = load(QWEN_TOKENIZERS[0]).backend_tokenizer.to_str()
    tokenizer = _TokenInPythonTokenizer(tokenizer_object=Tokenizer.from_str(config))
    text = make_text(MIN_PARALLEL_ENCODE_CHARS + 10_000)
    assert try_parallel_encode(tokenizer, text, NO_TRUNCATION) is None


class _KeepText:
    """A pre-tokenizer step written in Python that leaves the text as it is."""

    def pre_tokenize(self, pretok):
        pass


def test_normal_call_for_pre_tokenizer_in_python():
    # A pre-tokenizer step written in Python cannot be serialized, so the
    # configuration of the tokenizer cannot be checked.
    tokenizer = _with_config(QWEN_TOKENIZERS[0], lambda config: None)
    backend = tokenizer.backend_tokenizer
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.PreTokenizer.custom(_KeepText()),
            backend.pre_tokenizer[0],
            backend.pre_tokenizer[1],
        ]
    )
    text = make_text(MIN_PARALLEL_ENCODE_CHARS + 10_000)
    assert try_parallel_encode(tokenizer, text, NO_TRUNCATION) is None


def test_normal_call_for_other_tokenizers():
    # GPT-2 splits text with the regex of its ByteLevel pre-tokenizer, which
    # is not checked.
    tokenizer = get_tokenizer("openai-community/gpt2")
    text = make_text(MIN_PARALLEL_ENCODE_CHARS + 10_000)
    assert try_parallel_encode(tokenizer, text, NO_TRUNCATION) is None
