# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for lmcache.v1.utils.json_utils.

Covers make_json_safe (recursive sanitization) and safe_asdict
(dataclass -> JSON-safe dict) across primitive, container,
nested, non-serializable, and invalid-input scenarios.
"""

# Standard
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.utils.json_utils import make_json_safe, safe_asdict

# ---------------------------------------------------------------- #
#  make_json_safe
# ---------------------------------------------------------------- #


class TestMakeJsonSafe:
    @pytest.mark.parametrize(
        "value",
        ["s", 1, 1.5, True, False, None],
    )
    def test_primitives_unchanged(self, value: Any):
        """JSON-native primitives are returned unchanged."""
        assert make_json_safe(value) == value
        assert type(make_json_safe(value)) is type(value)

    def test_dict_recursively_sanitized(self):
        obj = {"a": 1, "b": Path("/tmp/x"), "c": {"d": (1, 2)}}
        result = make_json_safe(obj)
        assert result == {"a": 1, "b": "/tmp/x", "c": {"d": [1, 2]}}

    def test_tuple_becomes_list(self):
        assert make_json_safe((1, 2, 3)) == [1, 2, 3]

    def test_list_recursively_sanitized(self):
        assert make_json_safe([1, Path("/a"), [Path("/b")]]) == [
            1,
            "/a",
            ["/b"],
        ]

    def test_non_serializable_falls_back_to_str(self):
        class Opaque:
            def __str__(self) -> str:
                return "opaque-repr"

        assert make_json_safe(Opaque()) == "opaque-repr"

    def test_nested_mixed_structures(self):
        obj = {"items": [{"p": Path("/x")}, (Path("/y"),)]}
        assert make_json_safe(obj) == {
            "items": [{"p": "/x"}, ["/y"]],
        }


# ---------------------------------------------------------------- #
#  safe_asdict
# ---------------------------------------------------------------- #


@dataclass
class _FakeCfg:
    name: str = "n"
    port: int = 8080
    path: Path = Path("/tmp/cfg")
    nested: dict = field(default_factory=lambda: {"k": Path("/v")})


@dataclass
class _EmptyCfg:
    pass


class TestSafeAsdict:
    def test_dataclass_converted_with_non_serializable_fields(self):
        result = safe_asdict(_FakeCfg())
        assert result == {
            "name": "n",
            "port": 8080,
            "path": "/tmp/cfg",
            "nested": {"k": "/v"},
        }

    def test_empty_dataclass(self):
        assert safe_asdict(_EmptyCfg()) == {}

    def test_non_dataclass_raises(self):
        with pytest.raises(TypeError, match="Expected a dataclass"):
            safe_asdict({"not": "a dataclass"})

    def test_dataclass_class_rejected(self):
        """A dataclass *type* (not instance) must be rejected."""
        with pytest.raises(TypeError, match="Expected a dataclass"):
            safe_asdict(_FakeCfg)

    @pytest.mark.parametrize("bad", [None, 42, "str", [1, 2]])
    def test_primitive_inputs_rejected(self, bad: Any):
        with pytest.raises(TypeError, match="Expected a dataclass"):
            safe_asdict(bad)
