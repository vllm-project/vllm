# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from dataclasses import dataclass
from typing import Optional

import pytest

from vllm.config import LogprobsMode
from vllm.config.utils import get_hash_factors, hash_factors, normalize_value

# Helpers

def _endswith_fqname(obj, suffix: str) -> bool:
    # normalize_value(type) returns fully-qualified name
    # Compare suffix to avoid brittle import paths.
    out = normalize_value(obj)
    return isinstance(out, str) and out.endswith(suffix)


def _expected_path(p_str: str = ".") -> str:
    import pathlib
    p = pathlib.Path(p_str)
    return str(p.expanduser().resolve())


# Minimal dataclass to test get_hash_factors.
# Avoid importing heavy vLLM configs.
@dataclass
class SimpleConfig:
    a: object
    b: Optional[object] = None


def test_hash_factors_deterministic():
    """Test that hash_factors produces consistent SHA-256 hashes"""
    factors = {"a": 1, "b": "test"}
    hash1 = hash_factors(factors)
    hash2 = hash_factors(factors)

    assert hash1 == hash2
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
    # LogprobsMode uses a string payload.
    out = normalize_value(LogprobsMode.RAW_LOGITS)
    assert isinstance(out, tuple)
    assert out[0].endswith("LogprobsMode")
    # Expect string payload 'raw_logits'.
    assert out[1] == "raw_logits"


def test_normalize_value_set_order_insensitive():
    # Sets are unordered; normalize_value sorts elements for determinism.
    assert normalize_value({3, 1, 2}) == normalize_value({1, 2, 3})


def test_normalize_value_path_normalization():
    from pathlib import Path  # local import to avoid global dependency
    # Paths expand/resolve to absolute strings.
    # Stabilizes hashing across working dirs.
    assert normalize_value(Path(".")) == _expected_path(".")


def test_normalize_value_uuid_and_to_json():
    # Objects may normalize via uuid() or to_json_string().
    class HasUUID:
        def uuid(self):
            return "test-uuid"

    class ToJson:
        def to_json_string(self):
            return "{\"x\":1}"

    assert normalize_value(HasUUID()) == "test-uuid"
    assert normalize_value(ToJson()) == "{\"x\":1}"


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
    nf_enum = normalize_value(LogprobsMode.RAW_LOGITS)
    assert isinstance(nf_enum, tuple) and len(nf_enum) == 2
    enum_type, enum_val = nf_enum
    assert enum_type.endswith(".LogprobsMode")
    assert enum_val == "raw_logits"

    # Build factor dicts from configs with int vs enum
    f_int = get_hash_factors(SimpleConfig(1), set())
    f_enum = get_hash_factors(SimpleConfig(LogprobsMode.RAW_LOGITS), set())
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

    assert _endswith_fqname(LocalDummy, ".LocalDummy")

    from vllm.v1.sample.logits_processor.interface import LogitsProcessor
    from vllm.executor.executor_base import ExecutorBase
    from vllm.compilation.inductor_pass import InductorPass

    assert _endswith_fqname(LogitsProcessor, ".LogitsProcessor")
    assert _endswith_fqname(ExecutorBase, ".ExecutorBase")
    assert _endswith_fqname(InductorPass, ".InductorPass")
