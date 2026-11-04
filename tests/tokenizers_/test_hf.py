# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, TokenizersBackend

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import (
    CachedHfTokenizer,
    ThreadSafeHFTokenizerMixin,
    _copy_for_pool,
    get_cached_tokenizer,
    maybe_make_thread_pool,
)


@pytest.fixture
def local_tokenizer_dir(tmp_path):
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    return tmp_path


def test_cached_tokenizer_ignores_unrelated_jinja(local_tokenizer_dir):
    template = "{{ messages[0].content }}"
    (local_tokenizer_dir / "custom_chat.jinja").write_text(template)

    tokenizer = CachedHfTokenizer.from_pretrained(local_tokenizer_dir)
    assert tokenizer.chat_template is None


def test_cached_tokenizer_preserves_standard_chat_template(local_tokenizer_dir):
    (local_tokenizer_dir / "chat_template.jinja").write_text("standard")
    (local_tokenizer_dir / "custom_chat.jinja").write_text("fallback")

    tokenizer = CachedHfTokenizer.from_pretrained(local_tokenizer_dir)
    assert tokenizer.chat_template == "standard"


@pytest.mark.parametrize("model_id", ["openai-community/gpt2", "zai-org/chatglm3-6b"])
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


@pytest.mark.parametrize("model_id", ["openai-community/gpt2"])
def test_thread_pool_tokenizer_pickle(model_id: str):
    """Regression test for issue #45433: the thread-pool tokenizer wrapper
    reconstructs through maybe_make_thread_pool on unpickling, which used to
    fall off the end and return None."""
    reference_tokenizer = AutoTokenizer.from_pretrained(model_id)

    pooled_tokenizer = maybe_make_thread_pool(deepcopy(reference_tokenizer))
    assert pooled_tokenizer is not None
    assert isinstance(pooled_tokenizer, ThreadSafeHFTokenizerMixin)

    unpickled_tokenizer = pickle.loads(pickle.dumps(pooled_tokenizer))
    assert unpickled_tokenizer is not None
    assert isinstance(unpickled_tokenizer, ThreadSafeHFTokenizerMixin)
    assert unpickled_tokenizer.encode("prompt") == reference_tokenizer.encode("prompt")  # type: ignore[attr-defined]

    # Idempotence: wrapping an already-pooled tokenizer returns it unchanged.
    assert maybe_make_thread_pool(pooled_tokenizer) is pooled_tokenizer  # type: ignore[type-var]


@pytest.mark.parametrize("model_id", ["openai-community/gpt2"])
def test_thread_pool_tokenizer_copies_only_backend(model_id: str):
    """Pool copies get their own Rust backend tokenizer and share the
    Python-side state, which the pooled methods only read."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer_copy = _copy_for_pool(tokenizer)

    assert tokenizer_copy._tokenizer is not tokenizer._tokenizer
    assert tokenizer_copy.init_kwargs is tokenizer.init_kwargs
    assert tokenizer_copy.encode("prompt") == tokenizer.encode("prompt")


@pytest.mark.parametrize("model_id", ["openai-community/gpt2"])
def test_thread_pool_tokenizer_concurrent_calls(model_id: str):
    """Concurrent calls that set different truncation lengths on the Rust
    backend tokenizer must each get their own backend, or they raise
    `RuntimeError: Already borrowed` or truncate to another call's length."""
    reference_tokenizer = AutoTokenizer.from_pretrained(model_id)
    pooled_tokenizer = maybe_make_thread_pool(deepcopy(reference_tokenizer), copies=4)

    calls = [("hello world " * (1 + i % 32), 4 + i % 8) for i in range(512)]

    def encode(tokenizer, prompt: str, max_length: int) -> list[int]:
        return tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(
            executor.map(lambda call: encode(pooled_tokenizer, *call), calls)
        )

    assert results == [encode(reference_tokenizer, *call) for call in calls]


@pytest.mark.parametrize("model_id", ["openai-community/gpt2"])
def test_cached_tokenizer_max_token_id_is_a_valid_id(model_id: str):
    """max_token_id is the largest valid id, one less than vocab_size."""
    reference_tokenizer = AutoTokenizer.from_pretrained(model_id)
    cached_tokenizer = get_cached_tokenizer(deepcopy(reference_tokenizer))

    expected = max(
        max(reference_tokenizer.get_vocab().values()),
        reference_tokenizer.vocab_size - 1,
    )
    assert cached_tokenizer.max_token_id == expected
    assert cached_tokenizer.max_token_id == max(
        reference_tokenizer.get_vocab().values()
    )
