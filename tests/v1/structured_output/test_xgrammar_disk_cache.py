# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the optional xgrammar compiled-grammar disk cache.

These exercise the env-gated, default-off, bounded, (serialization-version +
tokenizer)-keyed persistent disk cache wired into ``XgrammarBackend``:
round-trip faithfulness, hit/miss = recompile-or-not, robust fallback on a
deserialize failure, whole-cache invalidation on a serialization-version bump,
the tokenizer-dependent key, and the zero-overhead default-off path.

Two structural choices keep these hermetic and crash-free:

* **One backend per test.** ``compile_grammar`` consults the disk cache *before*
  compiling, so a second ``compile_grammar`` call for the same grammar already
  takes the persisted-on-disk path (deserialize instead of recompile) -- no
  second backend is needed to exercise persistence.
* **A forked process per test** (``fork_new_process_for_each_test``). Each test
  therefore constructs exactly one cache-enabled ``XgrammarBackend`` in its own
  process, which also gives every test an independent, pristine env + cache.
"""

import json
from unittest import mock

import pytest
import torch
import xgrammar
from transformers import AutoTokenizer

import vllm.envs as envs
from tests.utils import fork_new_process_for_each_test
from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
from vllm.v1.structured_output.utils import get_xgrammar_disk_cache

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


def _start_mask(backend, grammar):
    """Token bitmask at the grammar's start state (proves functional identity)."""
    bitmask = backend.allocate_token_bitmask(1)
    grammar.fill_bitmask(bitmask, 0)
    return bitmask


@fork_new_process_for_each_test
def test_disk_cache_persists_and_round_trips(monkeypatch, tmp_path):
    """Compiling writes a persistent entry; recompiling the same grammar serves
    it from disk (no recompile) and the deserialized grammar is functionally
    identical to the freshly compiled one."""
    backend = _make_backend(monkeypatch, tmp_path)
    fresh = backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
    key = backend._disk_key(StructuredOutputOptions.JSON, SCHEMA_A)
    assert key in backend._disk_cache  # cold miss wrote the entry

    with mock.patch.object(
        backend.compiler,
        "compile_json_schema",
        wraps=backend.compiler.compile_json_schema,
    ) as spy:
        from_disk = backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
        spy.assert_not_called()  # served from disk, not recompiled

    # Identical token mask at the start state proves the deserialized grammar
    # is functionally identical, not merely that a file exists.
    assert torch.equal(_start_mask(backend, fresh), _start_mask(backend, from_disk))


@fork_new_process_for_each_test
def test_disk_cache_round_trips_non_json_grammar(monkeypatch, tmp_path):
    """The cache is grammar-type-agnostic: a GBNF GRAMMAR round-trips too."""
    backend = _make_backend(monkeypatch, tmp_path)
    fresh = backend.compile_grammar(StructuredOutputOptions.GRAMMAR, GRAMMAR_SPEC)
    key = backend._disk_key(StructuredOutputOptions.GRAMMAR, GRAMMAR_SPEC)
    assert key in backend._disk_cache

    with mock.patch.object(
        backend.compiler, "compile_grammar", wraps=backend.compiler.compile_grammar
    ) as spy:
        from_disk = backend.compile_grammar(
            StructuredOutputOptions.GRAMMAR, GRAMMAR_SPEC
        )
        spy.assert_not_called()  # served from disk, not recompiled

    assert torch.equal(_start_mask(backend, fresh), _start_mask(backend, from_disk))


@fork_new_process_for_each_test
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


@fork_new_process_for_each_test
def test_disk_cache_recompiles_when_deserialize_fails(monkeypatch, tmp_path):
    """A deserialize failure on a hit must fall back to a fresh compile."""
    backend = _make_backend(monkeypatch, tmp_path)
    backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)  # writes an entry

    with (
        mock.patch.object(
            xgrammar.CompiledGrammar,
            "deserialize_json",
            side_effect=xgrammar.DeserializeVersionError("simulated version skew"),
        ),
        mock.patch.object(
            backend.compiler,
            "compile_json_schema",
            wraps=backend.compiler.compile_json_schema,
        ) as spy,
    ):
        grammar = backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
        spy.assert_called_once()  # disk hit -> deserialize raised -> recompiled

    # The recompiled grammar is valid and usable.
    _start_mask(backend, grammar)


@fork_new_process_for_each_test
def test_disk_cache_recompiles_when_read_raises(monkeypatch, tmp_path):
    """A raising cache read (e.g. corrupt SQLite db) must fall back to a fresh
    compile, never 500 -- the read is guarded, not just deserialize."""
    backend = _make_backend(monkeypatch, tmp_path)
    with mock.patch.object(
        backend._disk_cache, "get", side_effect=RuntimeError("corrupt db")
    ):
        grammar = backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)

    # The fallback grammar is valid and usable.
    _start_mask(backend, grammar)


@fork_new_process_for_each_test
def test_disk_cache_wiped_on_serialization_version_bump(monkeypatch, tmp_path):
    """A serialization-version change wipes the whole cache when it is next
    opened, directly exercising the version-keying in
    ``get_xgrammar_disk_cache``.

    ``xgr`` is mocked in the utils module so the helper reports a controllable
    serialization version without invoking xgrammar; the cache itself is real.
    """
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("VLLM_XGRAMMAR_DISK_CACHE", "1")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()

    import vllm.v1.structured_output.utils as so_utils

    fake_xgr = mock.MagicMock()  # capability hasattr() checks pass trivially
    fake_xgr.get_serialization_version.return_value = "v-1"
    monkeypatch.setattr(so_utils, "xgr", fake_xgr)

    # Open at version "v-1" and seed an entry.
    cache = get_xgrammar_disk_cache()
    assert cache.get("__version__") == "v-1"
    cache.set("some-grammar-key", "serialized-grammar")
    cache.close()

    # A serialization-version change must wipe the whole cache on next open.
    fake_xgr.get_serialization_version.return_value = "v-2"
    reopened = get_xgrammar_disk_cache()
    assert "some-grammar-key" not in reopened  # bump wiped the stale entry
    assert reopened.get("__version__") == "v-2"


@fork_new_process_for_each_test
def test_disk_key_depends_on_tokenizer_fingerprint(monkeypatch, tmp_path):
    """The cache key must change when the tokenizer fingerprint changes."""
    assert isinstance(xgrammar.get_serialization_version(), str)
    assert xgrammar.get_serialization_version()  # non-empty

    backend = _make_backend(monkeypatch, tmp_path)
    key = backend._disk_key(StructuredOutputOptions.JSON, SCHEMA_A)
    backend._tokenizer_key = "a-different-tokenizer-fingerprint"
    other = backend._disk_key(StructuredOutputOptions.JSON, SCHEMA_A)
    assert key != other


@fork_new_process_for_each_test
def test_disk_cache_disabled_by_default(monkeypatch, tmp_path):
    """Off by default: no cache object, no disk writes, no behavior change."""
    backend = _make_backend(monkeypatch, tmp_path, enabled=False)
    assert backend._disk_cache is None
    assert backend._tokenizer_key is None

    backend.compile_grammar(StructuredOutputOptions.JSON, SCHEMA_A)
    assert not (tmp_path / "xgrammar_cache").exists()
