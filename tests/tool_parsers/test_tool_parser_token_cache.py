# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from unittest.mock import MagicMock

import pytest

from vllm.tool_parsers.abstract_tool_parser import ToolParser


@pytest.fixture(autouse=True)
def clear_token_cache():
    """Clear class-level caches between tests."""
    ToolParser._token_id_cache.clear()
    ToolParser._token_str_cache.clear()
    yield
    ToolParser._token_id_cache.clear()
    ToolParser._token_str_cache.clear()


def _make_tokenizer():
    """Create a mock tokenizer that simulates encode/decode."""
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, **kw: [ord(c) for c in text[:5]]
    tokenizer.decode.side_effect = lambda ids: "".join(chr(i) for i in ids)
    tokenizer.get_vocab.return_value = {}
    return tokenizer


def test_cached_encode_returns_correct_result():
    tokenizer = _make_tokenizer()
    parser = ToolParser(tokenizer)

    result = parser._cached_encode("<tool>")
    assert result == [ord("<"), ord("t"), ord("o"), ord("o"), ord("l")]


def test_cached_encode_avoids_repeated_calls():
    tokenizer = _make_tokenizer()
    parser = ToolParser(tokenizer)

    parser._cached_encode("<tool>")
    parser._cached_encode("<tool>")
    parser._cached_encode("<tool>")

    tokenizer.encode.assert_called_once()


def test_cached_decode_returns_correct_result():
    tokenizer = _make_tokenizer()
    parser = ToolParser(tokenizer)

    result = parser._cached_decode(60)
    assert result == "<"


def test_cached_decode_avoids_repeated_calls():
    tokenizer = _make_tokenizer()
    parser = ToolParser(tokenizer)

    parser._cached_decode(60)
    parser._cached_decode(60)
    parser._cached_decode(60)

    tokenizer.decode.assert_called_once()


def test_cache_shared_across_parser_instances():
    tokenizer = _make_tokenizer()
    parser1 = ToolParser(tokenizer)
    parser1._cached_encode("<tool>")

    parser2 = ToolParser(tokenizer)
    parser2._cached_encode("<tool>")

    tokenizer.encode.assert_called_once()


def test_different_tokenizers_get_separate_cache_entries():
    tok1 = _make_tokenizer()
    tok2 = _make_tokenizer()

    parser1 = ToolParser(tok1)
    parser2 = ToolParser(tok2)

    parser1._cached_encode("<tool>")
    parser2._cached_encode("<tool>")

    tok1.encode.assert_called_once()
    tok2.encode.assert_called_once()


def test_concurrent_parser_instantiation_no_race():
    """Simulate the race condition from issue #34932.

    Multiple threads instantiate Hermes2ProToolParser concurrently
    with a shared tokenizer. Without caching, this causes
    RuntimeError('Already borrowed') from PyO3's RefCell.
    """
    tokenizer = _make_tokenizer()

    errors: list[Exception] = []

    def create_parser():
        try:
            ToolParser(tokenizer)._cached_encode("<tool_call>")
            ToolParser(tokenizer)._cached_decode(60)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=create_parser) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent access raised errors: {errors}"
    # Should have been called only once despite 50 threads
    tokenizer.encode.assert_called_once()


def test_concurrent_encode_with_slow_tokenizer():
    """Ensure lock prevents concurrent tokenizer access.

    Uses a slow mock tokenizer to increase the window for race conditions.
    """
    tokenizer = _make_tokenizer()
    call_count = 0
    lock = threading.Lock()

    original_encode = tokenizer.encode.side_effect

    def slow_encode(text, **kw):
        nonlocal call_count
        with lock:
            call_count += 1
        time.sleep(0.01)
        return original_encode(text, **kw)

    tokenizer.encode.side_effect = slow_encode

    errors: list[Exception] = []

    def encode_token():
        try:
            parser = ToolParser(tokenizer)
            parser._cached_encode("<tool_call>")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=encode_token) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # The slow tokenizer should only be called once due to caching
    assert call_count == 1, f"Expected 1 tokenizer call, got {call_count}"
