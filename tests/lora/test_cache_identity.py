# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.lora.cache_identity import (
    compute_lora_cache_key,
    ensure_lora_cache_key,
    is_content_versioned,
)
from vllm.lora.request import LoRARequest

pytestmark = pytest.mark.skip_global_cleanup


def _write_adapter(
    d, *, config: bytes, weights: bytes, name="adapter_model.safetensors"
):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_bytes(config)
    (d / name).write_bytes(weights)
    return str(d)


def test_same_content_same_key(tmp_path):
    # Reproducibility: identical contents -> identical key (cross-process safe).
    a = _write_adapter(tmp_path / "a", config=b'{"r":16}', weights=b"W")
    b = _write_adapter(tmp_path / "b", config=b'{"r":16}', weights=b"W")
    assert compute_lora_cache_key(a) == compute_lora_cache_key(b)


def test_changed_weights_change_key(tmp_path):
    # The #42125 property: same path, different weight bytes -> different key.
    d = tmp_path / "adapter"
    p = _write_adapter(d, config=b'{"r":16}', weights=b"WEIGHTS_A")
    key_a = compute_lora_cache_key(p)
    (d / "adapter_model.safetensors").write_bytes(b"WEIGHTS_B")
    key_b = compute_lora_cache_key(p)
    assert key_a != key_b


def test_changed_config_changes_key(tmp_path):
    a = _write_adapter(tmp_path / "a", config=b'{"r":16}', weights=b"W")
    b = _write_adapter(tmp_path / "b", config=b'{"r":32}', weights=b"W")
    assert compute_lora_cache_key(a) != compute_lora_cache_key(b)


def test_is_3d_flag_changes_key(tmp_path):
    p = _write_adapter(tmp_path / "a", config=b'{"r":16}', weights=b"W")
    assert compute_lora_cache_key(p, is_3d_lora_weight=False) != compute_lora_cache_key(
        p, is_3d_lora_weight=True
    )


def test_tensorizer_config_changes_key(tmp_path):
    p = _write_adapter(tmp_path / "a", config=b'{"r":16}', weights=b"W")
    # tensorizer_config_dict reroutes the weight source; differing configs must
    # not alias even with the same lora_path.
    k1 = compute_lora_cache_key(p, tensorizer_config_dict={"tensorizer_dir": "/x"})
    k2 = compute_lora_cache_key(p, tensorizer_config_dict={"tensorizer_dir": "/y"})
    assert k1 != k2


def test_length_prefix_no_concat_collision(tmp_path):
    # Without length-prefixed framing, (config=b"AB", weights=b"C") and
    # (config=b"A", weights=b"BC") would hash the same byte stream.
    a = _write_adapter(tmp_path / "a", config=b"AB", weights=b"C")
    b = _write_adapter(tmp_path / "b", config=b"A", weights=b"BC")
    assert compute_lora_cache_key(a) != compute_lora_cache_key(b)


def test_missing_sources_fall_back_to_path(tmp_path):
    # Unreadable/empty path -> deterministic path-only identity; different
    # paths differ, identical path matches.
    p1 = str(tmp_path / "nope-1")
    p2 = str(tmp_path / "nope-2")
    assert compute_lora_cache_key(p1) == compute_lora_cache_key(p1)
    assert compute_lora_cache_key(p1) != compute_lora_cache_key(p2)


def test_weight_source_precedence_safetensors_over_bin(tmp_path):
    # The loader prefers adapter_model.safetensors over .bin; the identity must
    # mirror that, so a change to the lower-precedence .bin is invisible.
    d = tmp_path / "a"
    d.mkdir()
    (d / "adapter_config.json").write_bytes(b'{"r":16}')
    (d / "adapter_model.safetensors").write_bytes(b"ST")
    (d / "adapter_model.bin").write_bytes(b"BIN1")
    k1 = compute_lora_cache_key(str(d))
    (d / "adapter_model.bin").write_bytes(b"BIN2")
    assert compute_lora_cache_key(str(d)) == k1


def test_weight_source_bin_only_tracks_bin(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    (d / "adapter_config.json").write_bytes(b'{"r":16}')
    (d / "adapter_model.bin").write_bytes(b"BIN_A")
    k1 = compute_lora_cache_key(str(d))
    (d / "adapter_model.bin").write_bytes(b"BIN_B")
    assert compute_lora_cache_key(str(d)) != k1


def test_ensure_fills_when_missing(tmp_path):
    p = _write_adapter(tmp_path / "a", config=b'{"r":16}', weights=b"W")
    req = LoRARequest(lora_name="x", lora_int_id=1, lora_path=p)
    assert req.lora_cache_key is None
    ensure_lora_cache_key(req)
    assert req.lora_cache_key == compute_lora_cache_key(p)


def test_ensure_noop_when_present(tmp_path):
    p = _write_adapter(tmp_path / "a", config=b'{"r":16}', weights=b"W")
    req = LoRARequest(
        lora_name="x", lora_int_id=1, lora_path=p, lora_cache_key="preset"
    )
    ensure_lora_cache_key(req)
    assert req.lora_cache_key == "preset"


def test_config_canonicalization_ignores_formatting(tmp_path):
    # Same JSON content, different whitespace / key order -> same identity.
    a = _write_adapter(tmp_path / "a", config=b'{"r": 16, "alpha": 32}', weights=b"W")
    b = _write_adapter(tmp_path / "b", config=b'{"alpha":32,"r":16}', weights=b"W")
    assert compute_lora_cache_key(a) == compute_lora_cache_key(b)


def test_relative_path_is_path_only(tmp_path, monkeypatch):
    # A relative path must NOT hash on-disk contents (worker may resolve it from
    # a different CWD); it degrades to a deterministic path-only identity.
    _write_adapter(tmp_path / "adapter", config=b'{"r":16}', weights=b"WEIGHTS_A")
    monkeypatch.chdir(tmp_path)
    k1 = compute_lora_cache_key("adapter")
    (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"WEIGHTS_B")
    # content changed but the relative path is path-only -> key unchanged
    assert compute_lora_cache_key("adapter") == k1


def test_is_content_versioned(tmp_path, monkeypatch):
    p = _write_adapter(tmp_path / "a", config=b'{"r":16}', weights=b"W")
    assert is_content_versioned(p) is True
    assert (
        is_content_versioned(str(tmp_path / "missing")) is False
    )  # absolute, no files
    monkeypatch.chdir(tmp_path)
    assert is_content_versioned("a") is False  # relative -> path-only


def test_ensure_handles_none():
    ensure_lora_cache_key(None)  # must not raise
