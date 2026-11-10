# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential tests: the Rust block hasher in `vllm._rust_core` must produce
byte-identical serializations and hashes to the Python implementation."""

import pickle
import random
from collections.abc import Callable
from typing import Any

import cbor2
import numpy as np
import pytest
import torch

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils import hashing
from vllm.utils.hashing import sha256, sha256_cbor, xxhash, xxhash_cbor
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
)
from vllm.v1.request import Request

rust_core = pytest.importorskip("vllm._rust_core")

pytestmark = pytest.mark.cpu_test

HASH_FNS: dict[str, Callable[[Any], bytes]] = {
    "sha256": sha256,
    "sha256_cbor": sha256_cbor,
    "xxhash": xxhash,
    "xxhash_cbor": xxhash_cbor,
}
ALGOS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            name.startswith("xxhash") and hashing._xxhash is None,
            reason="xxhash is not installed",
        ),
    )
    for name in HASH_FNS
]

# Encoding-width boundaries of pickle (BININT1/2/4, LONG1) and CBOR.
EDGE_INTS = [
    0, 1, 23, 24, 255, 256, 65535, 65536, 2**31 - 1, -1, -24, -25, -128, -129,
    -256, -257, -(2**31), 2**31, -(2**31) - 1, 2**32, 2**63 - 1, 2**63, -(2**63),
    2**64 - 1, 2**64, -(2**64), -(2**64) - 1, 2**127 - 1, -(2**127),
]  # fmt: skip
# Lengths around pickle's 64 KiB frame target, also outside of frames.
EDGE_LENS = [0, 1, 23, 24, 255, 256, 65535, 65536, 65537, 70000]


def _random_value(rng: random.Random, depth: int, pool: list[Any]) -> Any:
    kind = rng.randrange(8 if depth < 4 else 6)
    if kind == 0:
        return rng.choice([None, True, False])
    if kind == 1:
        return rng.choice(
            [rng.choice(EDGE_INTS), rng.randint(-(2**127), 2**127 - 1)]
            + [rng.randint(-300, 70000)] * 2
        )
    if kind == 2:
        pool.append(bytes(rng.randrange(256) for _ in range(rng.randrange(40))))
        return pool[-1]
    if kind == 3:
        alphabet = "abé中\U0001f600"
        pool.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(20))))
        return pool[-1]
    if kind == 4 and pool:
        # The same object again, which pickle memoizes.
        return rng.choice(pool)
    if kind == 5:
        n = rng.choice(EDGE_LENS[:4] + [300])
        return rng.choice([b"x" * n, "y" * n])
    items = tuple(_random_value(rng, depth + 1, pool) for _ in range(rng.randrange(6)))
    pool.append(items)
    return items


def _random_values(seed: int, n: int) -> list[Any]:
    rng = random.Random(seed)
    values: list[Any] = []
    for _ in range(n):
        pool: list[Any] = []
        values.append(_random_value(rng, 0, pool))
    return values


def _python_serialize(algo: str, obj: Any) -> bytes:
    if algo.endswith("cbor"):
        return cbor2.dumps(obj, canonical=True)
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _large_values() -> list[Any]:
    """Values whose pickles span several frames or bypass them."""
    shared = "shared"
    values: list[Any] = [shared, (shared, shared), ((), ()), ("", ""), (b"", b"")]
    for n in EDGE_LENS:
        values += [b"a" * n, "b" * n, (1, b"a" * n, 2), (1, "é" * n, 2)]
    # Tuples ending exactly at, and around, a frame boundary for each int width.
    for width, token in [(2, 5), (3, 300), (5, 70000)]:
        for n in range(65536 // width - 2, 65536 // width + 3):
            values += [tuple([token] * n), (b"p" * 32, tuple([token] * n), None)]
    values.append(tuple(range(100_000)))
    # Memo references past index 255 (LONG_BINGET).
    strs = tuple(f"s{i}" for i in range(300))
    values.append((strs, strs[299], strs[3]))
    return values


@pytest.mark.parametrize("algo", ["sha256", "sha256_cbor"])
def test_serialize_value_matches_python(algo: str):
    """The serialized bytes, not only the digests, are identical."""
    for obj in _random_values(seed=0, n=3000) + EDGE_INTS + _large_values():
        assert rust_core.serialize_value(algo, obj) == _python_serialize(algo, obj), (
            repr(obj)[:200]
        )


@pytest.mark.parametrize("algo", ALGOS)
def test_hash_value_matches_python(algo: str):
    hash_fn = HASH_FNS[algo]
    for obj in _random_values(seed=1, n=1000) + EDGE_INTS:
        assert rust_core.hash_value(algo, obj) == hash_fn(obj), repr(obj)[:200]


def test_hash_value_rejects_unsupported_types():
    for obj in [1.5, [1], {"a": 1}, 2**200, np.int32(1)]:
        with pytest.raises((TypeError, OverflowError)):
            rust_core.hash_value("sha256", obj)
    with pytest.raises(ValueError):
        rust_core.hash_value("md5", None)


def _random_tokens(rng: random.Random, n: int) -> list[int]:
    # Negative ids don't occur in practice, but are valid int32 values.
    lo = rng.choice([0, 0, 0, -(2**31)])
    hi = rng.choice([255, 65535, 151_000, 2**31 - 1])
    return [rng.randint(lo, hi) for _ in range(n)]


@pytest.mark.parametrize("algo", ALGOS)
def test_hash_block_tokens_matches_python(algo: str):
    hash_fn = HASH_FNS[algo]
    init_none_hash(hash_fn)
    none_hash = kv_cache_utils.NONE_HASH
    rng = random.Random(2)
    lengths = list(range(0, 40)) + [rng.randrange(5001) for _ in range(40)]
    lengths += [13106, 13107, 13108, 21845, 32767, 32768]
    extra_keys_choices = [
        None,
        (("cache_salt", "salt"),),
        (("lora", "name", "/path"), ("mm", "abc", 3)),
        (None, 5),
        (None, 1, 2, 3, 4),
    ]
    for n in lengths:
        tokens = _random_tokens(rng, n)
        parent = rng.choice([None, b"", hash_fn(n)])
        extra_keys = rng.choice(extra_keys_choices)
        expected = hash_block_tokens(hash_fn, parent, tokens, extra_keys)
        actual = rust_core.hash_block_tokens(
            algo, parent, tokens, extra_keys, none_hash=none_hash
        )
        assert actual == expected, (n, parent, extra_keys)


def _make_request(
    prompt_token_ids: list[int],
    block_hasher: Callable[[Request], list[bytes]],
    cache_salt: str | None = None,
    lora_request: LoRARequest | None = None,
    mm_positions: list[PlaceholderRange] | None = None,
    prompt_embeds: torch.Tensor | None = None,
) -> Request:
    mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            mm_position=position,
            identifier=f"hash_{i}",
            modality="image",
        )
        for i, position in enumerate(mm_positions or [])
    ]
    return Request(
        request_id="0",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        mm_features=mm_features or None,
        lora_request=lora_request,
        cache_salt=cache_salt,
        block_hasher=block_hasher,
        prompt_embeds=prompt_embeds,
    )


class _Hashers:
    """A Rust hasher wrapping the Python one, counting fallback calls."""

    def __init__(self, algo: str, block_size: int, lookahead: int):
        self.python = get_request_block_hasher(block_size, HASH_FNS[algo], lookahead)
        self.num_fallbacks = 0

        def fallback(request: Request) -> list[bytes]:
            self.num_fallbacks += 1
            return self.python(request)

        self.rust = rust_core.RustRequestBlockHasher(
            block_size, algo, None, lookahead, fallback
        )


def _assert_same_hashes(
    hashers: _Hashers, prompt: list[int], outputs: list[list[int]], **kwargs
) -> None:
    """Hash the prompt, then the outputs as they are appended step by step."""
    expected = _make_request(prompt, hashers.python, **kwargs)
    actual = _make_request(prompt, hashers.rust, **kwargs)
    assert actual.block_hashes == expected.block_hashes
    for step in outputs:
        expected.append_output_token_ids(step)
        actual.append_output_token_ids(step)
        assert actual.block_hashes == expected.block_hashes


def _random_outputs(rng: random.Random, num_steps: int) -> list[list[int]]:
    return [_random_tokens(rng, rng.choice([1, 1, 1, 4, 17])) for _ in range(num_steps)]


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("lookahead", [0, 1, 3])
def test_request_block_hasher_matches_python(algo: str, lookahead: int):
    init_none_hash(HASH_FNS[algo])
    rng = random.Random(3)
    for block_size in [1, 3, 16, 64]:
        hashers = _Hashers(algo, block_size, lookahead)
        for _ in range(8):
            prompt = _random_tokens(rng, rng.randrange(1, 600))
            _assert_same_hashes(hashers, prompt, _random_outputs(rng, 40))
        assert hashers.num_fallbacks == 0


@pytest.mark.parametrize("algo", ALGOS)
def test_request_block_hasher_long_prompt(algo: str):
    """A prompt whose blocks' pickles exceed the 64 KiB frame target."""
    init_none_hash(HASH_FNS[algo])
    rng = random.Random(4)
    hashers = _Hashers(algo, 16384, 0)
    _assert_same_hashes(hashers, _random_tokens(rng, 40000), [[1] * 16384, [2]])
    assert hashers.num_fallbacks == 0


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("lookahead", [0, 2])
def test_request_block_hasher_extra_keys(algo: str, lookahead: int):
    """Cache salts and LoRA are hashed in Rust; multi-modal inputs and prompt
    embeds go to the Python fallback until the new blocks are past them."""
    init_none_hash(HASH_FNS[algo])
    rng = random.Random(5)
    name = "adapter-é"
    rust_cases: list[dict[str, Any]] = [
        dict(cache_salt="salt"),
        dict(cache_salt="s" * 300),
        dict(lora_request=LoRARequest("adapter", 1, "/models/adapter")),
        # The same object as name and path is pickled as a memo reference.
        dict(lora_request=LoRARequest(name, 2, name)),
        dict(lora_request=LoRARequest("adapter", 3, "/p"), cache_salt="salt"),
    ]
    for kwargs in rust_cases:
        hashers = _Hashers(algo, 4, lookahead)
        prompt = _random_tokens(rng, 50)
        _assert_same_hashes(hashers, prompt, _random_outputs(rng, 20), **kwargs)
        assert hashers.num_fallbacks == 0, kwargs

    fallback_cases: list[dict[str, Any]] = [
        dict(mm_positions=[PlaceholderRange(offset=2, length=10)]),
        dict(
            mm_positions=[
                PlaceholderRange(offset=0, length=3),
                PlaceholderRange(offset=5, length=30),
            ],
            cache_salt="salt",
        ),
        dict(prompt_embeds=torch.randn(30, 4)),
        # Strings that may be the very key literals.
        dict(cache_salt="cache_salt"),
        dict(lora_request=LoRARequest("lora", 4, "/path")),
    ]
    for kwargs in fallback_cases:
        hashers = _Hashers(algo, 4, lookahead)
        prompt = _random_tokens(rng, 50)
        if "prompt_embeds" in kwargs:
            prompt = None  # type: ignore[assignment]
        outputs = _random_outputs(rng, 30)
        if prompt is None:
            expected = _make_request(prompt, hashers.python, **kwargs)
            actual = _make_request(prompt, hashers.rust, **kwargs)
            assert actual.block_hashes == expected.block_hashes
        else:
            _assert_same_hashes(hashers, prompt, outputs, **kwargs)
        assert hashers.num_fallbacks > 0, kwargs


def test_request_block_hasher_reads_none_hash_at_call_time():
    """`get_request_block_hasher` may run before `init_none_hash`."""
    hashers = _Hashers("sha256", 4, 0)
    for seed_fn in [sha256, sha256_cbor]:
        init_none_hash(seed_fn)
        expected = _make_request(list(range(10)), hashers.python)
        actual = _make_request(list(range(10)), hashers.rust)
        assert actual.block_hashes == expected.block_hashes


def test_request_block_hasher_sequence_tokens():
    """Token ids that aren't an int32 array are read as a sequence."""
    init_none_hash(sha256)
    hashers = _Hashers("sha256", 4, 1)
    request = _make_request(list(range(10)), hashers.python)
    request.block_hashes.clear()
    expected = hashers.python(request)
    request._all_token_ids = list(request._all_token_ids)
    assert hashers.rust(request) == expected
    assert hashers.num_fallbacks == 0


def test_get_request_block_hasher_uses_rust(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_RUST_CORE", "1")
    hasher = get_request_block_hasher(16, sha256, 0)
    assert isinstance(hasher, rust_core.RustRequestBlockHasher)
    # Unknown hash functions keep the Python hasher.
    hasher = get_request_block_hasher(16, lambda x: sha256(x), 0)
    assert not isinstance(hasher, rust_core.RustRequestBlockHasher)
    monkeypatch.setenv("VLLM_USE_RUST_CORE", "0")
    hasher = get_request_block_hasher(16, sha256, 0)
    assert not isinstance(hasher, rust_core.RustRequestBlockHasher)
