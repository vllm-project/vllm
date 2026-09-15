# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy as copy_module
import pickle
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, TokenizersBackend

import vllm.tokenizers.hf as hf_module
from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.hf import (
    ThreadSafeHFTokenizerMixin,
    get_cached_tokenizer,
    maybe_make_thread_pool,
)


def _make_local_tokenizer() -> TokenizersBackend:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return TokenizersBackend(tokenizer_object=backend, unk_token="[UNK]")


def _make_pooled_tokenizer(monkeypatch, encode, copies: int = 1):
    deepcopy_calls: list[None] = []

    def counted_deepcopy(_):
        deepcopy_calls.append(None)
        return SimpleNamespace(encode=encode)

    monkeypatch.setattr(
        hf_module,
        "copy",
        SimpleNamespace(copy=copy_module.copy, deepcopy=counted_deepcopy),
    )
    return maybe_make_thread_pool(_make_local_tokenizer(), copies), deepcopy_calls


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
    assert unpickled_tokenizer.encode("prompt") == reference_tokenizer.encode("prompt")

    # Idempotence: wrapping an already-pooled tokenizer returns it unchanged.
    assert maybe_make_thread_pool(pooled_tokenizer) is pooled_tokenizer


def test_thread_pool_discards_overflow_copies(monkeypatch: pytest.MonkeyPatch):
    active_barrier = [threading.Barrier(1)]

    def blocking_encode(_):
        active_barrier[0].wait(timeout=10)
        return [1]

    pooled_tokenizer, deepcopy_calls = _make_pooled_tokenizer(
        monkeypatch, blocking_encode
    )
    assert len(deepcopy_calls) == 1

    def run_burst(workers: int):
        active_barrier[0] = threading.Barrier(workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = list(
                executor.map(lambda _: pooled_tokenizer.encode("hello"), range(workers))
            )
        assert outputs == [[1]] * workers

    run_burst(3)
    assert len(deepcopy_calls) == 3

    run_burst(3)
    assert len(deepcopy_calls) == 5


def test_thread_pool_returns_copy_when_call_raises_queue_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    should_raise = True

    def encode_that_raises_once(_):
        nonlocal should_raise
        if should_raise:
            should_raise = False
            raise queue.Empty("raised by tokenizer")
        return [1]

    pooled_tokenizer, deepcopy_calls = _make_pooled_tokenizer(
        monkeypatch, encode_that_raises_once
    )
    with pytest.raises(queue.Empty, match="raised by tokenizer"):
        pooled_tokenizer.encode("hello")

    assert pooled_tokenizer.encode("hello") == [1]
    assert len(deepcopy_calls) == 1
