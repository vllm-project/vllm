# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum

import pytest

from vllm.config.cache import CacheConfig
from vllm.config.utils import get_hash_factors, hash_factors, normalize_value

# Helpers


def endswith_fqname(obj, suffix: str) -> bool:
    # normalize_value(type) returns fully-qualified name
    # Compare suffix to avoid brittle import paths.
    out = normalize_value(obj)
    return isinstance(out, str) and out.endswith(suffix)


def expected_path(p_str: str = ".") -> str:
    import pathlib

    p = pathlib.Path(p_str)
    return p.expanduser().resolve().as_posix()


# Minimal dataclass to test get_hash_factors.
# Avoid importing heavy vLLM configs.
@dataclass
class SimpleConfig:
    a: object
    b: object | None = None


class DummyLogprobsMode(Enum):
    RAW_LOGITS = "raw_logits"


def test_hash_factors_deterministic():
    """Test that hash_factors produces consistent SHA-256 hashes"""
    factors = {"a": 1, "b": "test"}
    hash1 = hash_factors(factors)
    hash2 = hash_factors(factors)

    assert hash1 == hash2
    # Dict key insertion order should not affect the hash.
    factors_reordered = {"b": "test", "a": 1}
    assert hash_factors(factors_reordered) == hash1
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)


@pytest.mark.parametrize(
    "inp, expected",
    [
        (None, None),
        (True, True),
        (1, 1),
        (1.0, 1.0),
        ("x", "x"),
        (b"ab", "6162"),
        (bytearray(b"ab"), "6162"),
        ([1, 2], (1, 2)),
        ({"b": 2, "a": 1}, (("a", 1), ("b", 2))),
    ],
)
def test_normalize_value_matrix(inp, expected):
    """Parametric input→expected normalization table."""
    assert normalize_value(inp) == expected


def test_normalize_value_enum():
    # Enums normalize to (module.QualName, value).
    # DummyLogprobsMode uses a string payload.
    out = normalize_value(DummyLogprobsMode.RAW_LOGITS)
    assert isinstance(out, tuple)
    assert out[0].endswith("DummyLogprobsMode")
    # Expect string payload 'raw_logits'.
    assert out[1] == "raw_logits"


def test_normalize_value_set_order_insensitive():
    # Sets are unordered; normalize_value sorts elements for determinism.
    assert normalize_value({3, 1, 2}) == normalize_value({1, 2, 3})


def test_normalize_value_path_normalization():
    from pathlib import Path  # local import to avoid global dependency

    # Paths expand/resolve to absolute strings.
    # Stabilizes hashing across working dirs.
    assert normalize_value(Path(".")) == expected_path(".")


def test_normalize_value_uuid_and_to_json():
    # Objects may normalize via uuid() or to_json_string().
    class HasUUID:
        def uuid(self):
            return "test-uuid"

    class ToJson:
        def to_json_string(self):
            return '{"x":1}'

    assert normalize_value(HasUUID()) == "test-uuid"
    assert normalize_value(ToJson()) == '{"x":1}'


@pytest.mark.parametrize(
    "bad",
    [
        (lambda x: x),
        (type("CallableInstance", (), {"__call__": lambda self: 0}))(),
        (lambda: (lambda: 0))(),  # nested function instance
    ],
)
def test_error_cases(bad):
    """Inputs expected to raise TypeError."""
    # Reject functions/lambdas/callable instances
    # to avoid under-hashing.
    with pytest.raises(TypeError):
        normalize_value(bad)


def test_enum_vs_int_disambiguation():
    # int stays primitive
    nf_int = normalize_value(1)
    assert nf_int == 1

    # enum becomes ("module.QualName", value)
    nf_enum = normalize_value(DummyLogprobsMode.RAW_LOGITS)
    assert isinstance(nf_enum, tuple) and len(nf_enum) == 2
    enum_type, enum_val = nf_enum
    assert enum_type.endswith(".DummyLogprobsMode")
    assert enum_val == "raw_logits"

    # Build factor dicts from configs with int vs enum
    f_int = get_hash_factors(SimpleConfig(1), set())
    f_enum = get_hash_factors(SimpleConfig(DummyLogprobsMode.RAW_LOGITS), set())
    # The int case remains a primitive value
    assert f_int["a"] == 1
    # The enum case becomes a tagged tuple ("module.QualName", "raw_logits")
    assert isinstance(f_enum["a"], tuple) and f_enum["a"][1] == "raw_logits"
    # Factor dicts must differ so we don't collide primitives with Enums.
    assert f_int != f_enum
    # Hash digests must differ correspondingly
    assert hash_factors(f_int) != hash_factors(f_enum)

    # Hash functions produce stable hex strings
    h_int = hash_factors(f_int)
    h_enum = hash_factors(f_enum)
    assert isinstance(h_int, str) and len(h_int) == 64
    assert isinstance(h_enum, str) and len(h_enum) == 64


def test_classes_are_types():
    """Types normalize to FQNs; include real vLLM types."""
    # Only classes allowed; functions/lambdas are rejected.
    # Canonical form is the fully-qualified name.
    assert isinstance(normalize_value(str), str)

    class LocalDummy:
        pass

    assert endswith_fqname(LocalDummy, ".LocalDummy")


def test_envs_compile_factors_stable():
    """Test that envs.compile_factors() hash is stable across fresh initializations.

    Uses subprocesses to ensure env vars with dynamic defaults (like UUIDs)
    are freshly generated each time, verifying they're properly ignored.
    """
    import subprocess
    import sys

    code = """
import sys
import logging
logging.disable(logging.CRITICAL)
from vllm import envs
from vllm.config.utils import hash_factors
print(hash_factors(envs.compile_factors()))
"""

    def get_hash_in_subprocess():
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**dict(__import__("os").environ), "VLLM_LOGGING_LEVEL": "ERROR"},
        )
        return result.stdout.strip()

    hash1 = get_hash_in_subprocess()
    hash2 = get_hash_in_subprocess()

    assert hash1 == hash2, (
        "compile_factors hash differs between fresh initializations - "
        "dynamic env vars may not be properly ignored"
    )


def test_cache_config_hash_ignores_kv_cache_sizing_knobs():
    """kv_cache_memory_bytes only sizes the KV cache allocation (like
    gpu_memory_utilization, which is already ignored); it does not affect
    the compiled computation graph. If it leaks into the hash, setting the
    documented fast-boot knob silently invalidates the torch.compile cache
    and forces a full recompile.
    """
    base_hash = CacheConfig().compute_hash()
    assert CacheConfig(kv_cache_memory_bytes=1 << 30).compute_hash() == base_hash
    assert CacheConfig(gpu_memory_utilization=0.5).compute_hash() == base_hash


def test_envs_compile_factors_relocation_invariant(tmp_path):
    """Relocating HOME or the XDG roots must not change the compile-cache
    env hash.

    Location-derived env vars (VLLM_XLA_CACHE_PATH from XDG_CACHE_HOME,
    VLLM_CONFIG_ROOT from XDG_CONFIG_HOME/HOME) carry no information about
    compiled artifacts, only about where directories live. When they leak
    into compile_factors(), a cache produced under one HOME/XDG layout
    silently misses under another - which defeats copying or pre-baking a
    compile cache into a container image.
    """
    import os
    import subprocess
    import sys

    code = """
import sys
import logging
logging.disable(logging.CRITICAL)
from vllm import envs
from vllm.config.utils import hash_factors
print(hash_factors(envs.compile_factors()))
"""

    def hash_with(extra_env):
        env = {**dict(os.environ), "VLLM_LOGGING_LEVEL": "ERROR"}
        # Drop explicit overrides so the derived defaults are what is
        # exercised, then apply the relocation under test.
        for key in ("VLLM_XLA_CACHE_PATH", "VLLM_CONFIG_ROOT", "VLLM_CACHE_ROOT"):
            env.pop(key, None)
        env.update(extra_env)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        return result.stdout.strip()

    xdg_cache = tmp_path / "relocated-xdg-cache"
    xdg_config = tmp_path / "relocated-xdg-config"
    new_home = tmp_path / "relocated-home"
    for d in (xdg_cache, xdg_config, new_home):
        d.mkdir()

    base = hash_with({})
    relocated_xdg = hash_with(
        {"XDG_CACHE_HOME": str(xdg_cache), "XDG_CONFIG_HOME": str(xdg_config)}
    )
    relocated_home = hash_with({"HOME": str(new_home)})

    assert relocated_xdg == base, (
        "XDG_CACHE_HOME/XDG_CONFIG_HOME relocation changed the compile-cache "
        "env hash - a location-only derived var is leaking into the key"
    )
    assert relocated_home == base, (
        "HOME relocation changed the compile-cache env hash - a "
        "location-only derived var is leaking into the key"
    )
