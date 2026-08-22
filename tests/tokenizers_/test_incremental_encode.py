# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the exact incremental prompt-encoding cache.

These tests only require ``transformers``; the module under test is
dependency-free by design, so it is loaded directly from its file when a
full vLLM installation is not available.
"""

import random

import pytest
from transformers import AutoTokenizer

try:
    from vllm.tokenizers.incremental_encode import IncrementalEncodeCache
except ImportError:  # pragma: no cover - minimal environments without vLLM
    import importlib.util
    from pathlib import Path

    _MODULE_PATH = (
        Path(__file__).resolve().parents[2]
        / "vllm"
        / "tokenizers"
        / "incremental_encode.py"
    )
    _spec = importlib.util.spec_from_file_location("incremental_encode", _MODULE_PATH)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    IncrementalEncodeCache = _module.IncrementalEncodeCache

TOKENIZER_IDS = [
    "openai-community/gpt2",
    "Qwen/Qwen2.5-0.5B",
    "deepseek-ai/DeepSeek-V3",
]

# Deliberately adversarial turn material: code, unicode (CJK, emoji with
# ZWJ, RTL, combining marks), whitespace runs, and literal special-token
# strings that must not be spliced through.
TURN_SNIPPETS = [
    "Here is a Python helper:\n\n```python\ndef merge(a: dict, b: dict) -> dict:"
    '\n    """Merge b into a."""\n    out = {**a}\n    for k, v in b.items():'
    "\n        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) else v"
    "\n    return out\n```\nIt handles nested dicts recursively.",
    "多轮对话的关键在于前缀缓存。日本語のテキストも含めてテストします。"
    "한국어 문장도 섞어 봅니다. 這是繁體中文。",
    "Emoji stress: 👩‍👩‍👧‍👦 🇯🇵 🧑🏽‍💻 étude résumé שלום مرحبا done.",
    "Whitespace run follows:" + " " * 137 + "\n\n\t\t\n" + " " * 41 + "end.",
    "Literal special tokens: <|endoftext|> <|im_start|>user<|im_end|> "
    "<|User|> <|Assistant|> <|begin▁of▁sentence|> and a bare <| marker.",
    "Log excerpt:\nERROR 2026-07-03T12:00:01Z worker[3] step=41952 "
    "loss=0.001234 grad_norm=17.3\n" * 3,
]

CACHE_KWARGS = dict(min_chars=2048, backup_chars=512, margin_chars=64)


@pytest.fixture(scope="module", params=TOKENIZER_IDS)
def tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


def _build_turns(num_turns: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    system = (
        "<|im_start|>system\nYou are a meticulous assistant. " * 20
        + "Rules:\n"
        + "\n".join(f"{i}. Always be exact." for i in range(40))
        + "<|im_end|>\n"
    )
    turns = [system]
    for i in range(num_turns):
        snippet = TURN_SNIPPETS[i % len(TURN_SNIPPETS)]
        turns.append(
            f"<|im_start|>user\nTurn {i}: {snippet} "
            f"{rng.getrandbits(64):x}<|im_end|>\n"
            f"<|im_start|>assistant\nReply {i}: {snippet[::-1][:200]}<|im_end|>\n"
        )
    return turns


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_multi_turn_token_exact(tokenizer, add_special_tokens: bool):
    """Growing conversation: every turn must be token-exact vs full encode."""
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    text = ""
    for turn in _build_turns(num_turns=8):
        text += turn
        actual = cache.encode(tokenizer, text, add_special_tokens=add_special_tokens)
        expected = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        if len(text) < cache.min_chars:
            assert actual is None
        else:
            assert actual == expected, f"mismatch at len(text)={len(text)}"
    # The growth pattern must actually exercise the splice path.
    assert cache.stats.hits >= 4
    assert cache.stats.misses == 1


def test_identical_prompt_is_served_from_cache(tokenizer):
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    text = "".join(_build_turns(num_turns=3))
    first = cache.encode(tokenizer, text)
    second = cache.encode(tokenizer, text)
    assert first == second == tokenizer(text)["input_ids"]
    assert cache.stats.hits == 1


def test_interleaved_conversations(tokenizer):
    """Multiple conversations growing concurrently share the LRU."""
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    all_turns = [_build_turns(num_turns=6, seed=c + 1) for c in range(3)]
    texts = ["", "", ""]
    for i in range(6):
        for c in range(3):
            texts[c] += all_turns[c][i]
            actual = cache.encode(tokenizer, texts[c])
            if actual is not None:
                assert actual == tokenizer(texts[c])["input_ids"]
    assert cache.stats.hits > 0


def test_truncation_within_limit_matches_full_encode(tokenizer):
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    text = "".join(_build_turns(num_turns=4))
    limit = len(tokenizer(text)["input_ids"]) + 10
    actual = cache.encode(tokenizer, text, truncation=True, max_length=limit)
    expected = tokenizer(text, truncation=True, max_length=limit)["input_ids"]
    assert actual == expected


def test_truncation_overflow_falls_back(tokenizer):
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    text = "".join(_build_turns(num_turns=4))
    assert cache.encode(tokenizer, text, truncation=True, max_length=16) is None
    # The cache was still warmed by the attempt.
    assert cache.encode(tokenizer, text) == tokenizer(text)["input_ids"]
    assert cache.stats.hits == 1


def test_unsupported_kwargs_are_rejected(tokenizer):
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    text = "".join(_build_turns(num_turns=4))
    assert cache.encode(tokenizer, text, return_tensors="pt") is None
    assert cache.encode(tokenizer, text, truncation=True) is None


def test_short_prompts_are_skipped(tokenizer):
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    assert cache.encode(tokenizer, "short prompt") is None
    assert cache.stats.misses == 0


def test_cache_salt_partitions_entries(tokenizer):
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    turns = _build_turns(num_turns=4)
    text = "".join(turns[:-1])
    cache.encode(tokenizer, text, cache_salt="salt-a")
    grown = text + turns[-1]
    # Different salt: the salt-a entry must not be visible.
    assert (
        cache.encode(tokenizer, grown, cache_salt="salt-b")
        == tokenizer(grown)["input_ids"]
    )
    assert cache.stats.hits == 0
    assert cache.stats.misses == 2
    # Same salt: the prefix entry is visible again.
    grown_more = grown + turns[1]
    assert (
        cache.encode(tokenizer, grown_more, cache_salt="salt-b")
        == tokenizer(grown_more)["input_ids"]
    )
    assert cache.stats.misses == 2


def test_lru_is_bounded(tokenizer):
    cache = IncrementalEncodeCache(max_entries=2, **CACHE_KWARGS)
    for seed in range(5):
        text = "".join(_build_turns(num_turns=3, seed=seed))
        cache.encode(tokenizer, text)
    assert len(cache._entries) == 2


def test_non_fast_tokenizer_is_unsupported():
    class _SlowTokenizer:
        is_fast = False

    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    text = "".join(_build_turns(num_turns=4))
    assert cache.encode(_SlowTokenizer(), text) is None
    assert cache.stats.misses == 0


def test_append_directly_after_special_token(tokenizer):
    """The seam guard must not break exactness when new content starts with
    or directly follows special-token markers."""
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    text = "".join(_build_turns(num_turns=3)) + "<|im_start|>assistant\n"
    assert cache.encode(tokenizer, text) == tokenizer(text)["input_ids"]
    text += "<|endoftext|>" * 40
    assert cache.encode(tokenizer, text) == tokenizer(text)["input_ids"]


def test_whitespace_run_across_seam(tokenizer):
    """A giant whitespace run spanning the backup window must either splice
    exactly or fall back — never corrupt the output."""
    cache = IncrementalEncodeCache(**CACHE_KWARGS)
    base = "".join(_build_turns(num_turns=3)) + " " * 2000
    assert cache.encode(tokenizer, base) == tokenizer(base)["input_ids"]
    grown = base + " " * 500 + "after the run"
    assert cache.encode(tokenizer, grown) == tokenizer(grown)["input_ids"]
