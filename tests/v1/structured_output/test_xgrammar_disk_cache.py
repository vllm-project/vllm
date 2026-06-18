# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the optional xgrammar compiled-grammar disk cache.

These exercise the env-gated, default-off, bounded, (serialization-version +
tokenizer)-keyed persistent disk cache wired into ``XgrammarBackend``:
round-trip faithfulness, hit/miss = recompile-or-not, robust fallback on a
deserialize failure, whole-cache invalidation on a serialization-version bump,
the tokenizer-dependent key, and the zero-overhead default-off path.
"""

import json
from unittest import mock

import pytest
import torch
import xgrammar
from transformers import AutoTokenizer

import vllm.envs as envs
from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

pytestmark = pytest.mark.cpu_test

TOKENIZER = "gpt2"
VOCAB_SIZE = 50257
SCHEMA_A = json.dumps({"type": "object", "properties": {"name": {"type": "string"}}})
SCHEMA_B = json.dumps({"type": "object", "properties": {"age": {"type": "integer"}}})
GRAMMAR_SPEC = 'root ::= "yes" | "no"'


def _make_backend(monkeypatch, cache_root, *, enabled=True):
    """Build an XgrammarBackend over ``cache_root`` with the disk cache on/off.

    Mirrors tests/v1/structured_output/test_backend_guidance.py construction.
    ``envs.__getattr__`` is uncached by default, so ``monkeypatch.setenv``
    takes effect immediately; the ``cache_clear`` guard is defensive in case a
    prior test enabled the opt-in env cache.
    """
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(cache_root))
    if enabled:
        monkeypatch.setenv("VLLM_XGRAMMAR_DISK_CACHE", "1")
    else:
        monkeypatch.delenv("VLLM_XGRAMMAR_DISK_CACHE", raising=False)
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()

    config = StructuredOutputsConfig(backend="xgrammar")
    vllm_config = VllmConfig(structured_outputs_config=config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return XgrammarBackend(vllm_config, tokenizer=tokenizer, vocab_size=VOCAB_SIZE)


def test_disk_cache_persists_across_backends_and_round_trips(monkeypatch, tmp_path):
    """Compiling writes a persistent entry; a fresh backend over the same dir
    serves it from disk (no recompile) and the deserialized grammar is
    functionally identical to the freshly compiled one."""
    b1 = _make_backend(monkeypatch, tmp_path)
    g1 = b1.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
    key = b1._disk_key(StructuredOutputOptions.JSON, SCHEMA_A)
    assert key in b1._disk_cache

    b2 = _make_backend(monkeypatch, tmp_path)
    with mock.patch.object(
        b2.compiler, "compile_json_schema", wraps=b2.compiler.compile_json_schema
    ) as spy:
        g2 = b2.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
        spy.assert_not_called()  # served from disk, not recompiled

    # Identical token mask at the start state proves the deserialized grammar
    # is functionally identical, not merely that a file exists.
    bm1 = b1.allocate_token_bitmask(1)
    bm2 = b2.allocate_token_bitmask(1)
    g1.fill_bitmask(bm1, 0)
    g2.fill_bitmask(bm2, 0)
    assert torch.equal(bm1, bm2)


def test_disk_cache_round_trips_non_json_grammar(monkeypatch, tmp_path):
    """The cache is grammar-type-agnostic: a GBNF GRAMMAR round-trips too."""
    b1 = _make_backend(monkeypatch, tmp_path)
    g1 = b1.compile_grammar(StructuredOutputOptions.GRAMMAR, GRAMMAR_SPEC)
    key = b1._disk_key(StructuredOutputOptions.GRAMMAR, GRAMMAR_SPEC)
    assert key in b1._disk_cache

    b2 = _make_backend(monkeypatch, tmp_path)
    with mock.patch.object(
        b2.compiler, "compile_grammar", wraps=b2.compiler.compile_grammar
    ) as spy:
        g2 = b2.compile_grammar(StructuredOutputOptions.GRAMMAR, GRAMMAR_SPEC)
        spy.assert_not_called()  # served from disk, not recompiled

    bm1 = b1.allocate_token_bitmask(1)
    bm2 = b2.allocate_token_bitmask(1)
    g1.fill_bitmask(bm1, 0)
    g2.fill_bitmask(bm2, 0)
    assert torch.equal(bm1, bm2)


def test_disk_cache_key_partitioning_avoids_and_forces_recompile(monkeypatch, tmp_path):
    """Same key -> disk hit (no recompile); different key -> recompile."""
    backend = _make_backend(monkeypatch, tmp_path)
    with mock.patch.object(
        backend.compiler,
        "compile_json_schema",
        wraps=backend.compiler.compile_json_schema,
    ) as spy:
        backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
        assert spy.call_count == 1  # miss -> compile + write
        backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
        assert spy.call_count == 1  # same key -> disk hit, no recompile
        backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_B)
        assert spy.call_count == 2  # different key -> recompile


def test_disk_cache_recompiles_when_deserialize_fails(monkeypatch, tmp_path):
    """A deserialize failure must fall back to a fresh compile, never 500."""
    b1 = _make_backend(monkeypatch, tmp_path)
    b1.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)  # writes an entry

    b2 = _make_backend(monkeypatch, tmp_path)
    with (
        mock.patch.object(
            xgrammar.CompiledGrammar,
            "deserialize_json",
            side_effect=xgrammar.DeserializeVersionError("simulated version skew"),
        ),
        mock.patch.object(
            b2.compiler,
            "compile_json_schema",
            wraps=b2.compiler.compile_json_schema,
        ) as spy,
    ):
        grammar = b2.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
        spy.assert_called_once()  # deserialize raised -> recompiled

    # The recompiled grammar is valid and usable.
    bitmask = b2.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)


def test_disk_cache_recompiles_when_read_raises(monkeypatch, tmp_path):
    """A raising cache read (e.g. corrupt SQLite db) must fall back to a fresh
    compile, never 500 -- the read is guarded, not just deserialize."""
    backend = _make_backend(monkeypatch, tmp_path)
    with mock.patch.object(
        backend._disk_cache, "get", side_effect=RuntimeError("corrupt db")
    ):
        grammar = backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)

    # The fallback grammar is valid and usable.
    bitmask = backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)


def test_disk_cache_wiped_on_serialization_version_bump(monkeypatch, tmp_path):
    """A serialization-version change wipes the whole cache on next open."""
    b1 = _make_backend(monkeypatch, tmp_path)
    b1.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
    key = b1._disk_key(StructuredOutputOptions.JSON, SCHEMA_A)
    assert key in b1._disk_cache

    monkeypatch.setattr(xgrammar, "get_serialization_version", lambda: "v-bumped")
    b2 = _make_backend(monkeypatch, tmp_path)
    assert key not in b2._disk_cache
    assert b2._disk_cache.get("__version__") == "v-bumped"


def test_disk_key_depends_on_tokenizer_fingerprint(monkeypatch, tmp_path):
    """The cache key must change when the tokenizer fingerprint changes."""
    assert isinstance(xgrammar.get_serialization_version(), str)
    assert xgrammar.get_serialization_version()  # non-empty

    backend = _make_backend(monkeypatch, tmp_path)
    key = backend._disk_key(StructuredOutputOptions.JSON, SCHEMA_A)
    backend._tokenizer_key = "a-different-tokenizer-fingerprint"
    other = backend._disk_key(StructuredOutputOptions.JSON, SCHEMA_A)
    assert key != other


def test_disk_cache_disabled_by_default(monkeypatch, tmp_path):
    """Off by default: no cache object, no disk writes, no behavior change."""
    backend = _make_backend(monkeypatch, tmp_path, enabled=False)
    assert backend._disk_cache is None
    assert backend._tokenizer_key is None

    backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
    assert not (tmp_path / "xgrammar_cache").exists()
