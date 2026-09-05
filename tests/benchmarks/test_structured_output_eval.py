# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the structured-output serving benchmark's correctness metric.

The benchmark module imports a live-serving stack at module load time; the
stubs below cover the pieces that are irrelevant to scoring so the real
module can be imported and its real `evaluate` exercised.
"""

import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

BENCH_PATH = (
    Path(__file__).parent.parent.parent
    / "benchmarks"
    / "benchmark_serving_structured_output.py"
)

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


def _stub_module(name: str, attrs: dict) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _load_module():
    # Stub the live-serving imports (and vllm itself) so the real module can
    # be imported and scored without a server stack. Keys inserted here are
    # removed again after load so the rest of the pytest process is unaffected.
    stubs: dict[str, types.ModuleType] = {}
    for name in ("datasets",):
        stubs[name] = MagicMock()
    stubs["backend_request_func"] = _stub_module(
        "backend_request_func",
        {
            "ASYNC_REQUEST_FUNCS": {},
            "RequestFuncInput": MagicMock(),
            "RequestFuncOutput": MagicMock(),
            "get_tokenizer": MagicMock(),
        },
    )
    for name in ("vllm", "vllm.v1", "vllm.v1.structured_output"):
        stubs[name] = types.ModuleType(name)
    stubs["vllm.tokenizers"] = _stub_module(
        "vllm.tokenizers", {"get_tokenizer": MagicMock()}
    )
    stubs["vllm.v1.structured_output.backend_xgrammar"] = _stub_module(
        "vllm.v1.structured_output.backend_xgrammar",
        {"has_xgrammar_unsupported_json_features": MagicMock()},
    )

    inserted = [name for name in stubs if name not in sys.modules]
    for name in inserted:
        sys.modules[name] = stubs[name]
    try:
        spec = importlib.util.spec_from_file_location(
            "benchmark_serving_structured_output", BENCH_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load benchmark module from {BENCH_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in inserted:
            sys.modules.pop(name, None)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load_module()


def _score(bench, ret):
    return bench.evaluate(ret, Namespace(structure_type="json"))


def test_schema_ignoring_backend_scores_zero(bench):
    # A backend that ignores guided decoding and emits {} for everything used
    # to score 100; it must now fail schema validation.
    ret = [{"generated": "{}", "expected": None, "schema": SCHEMA, "structured": True}]
    assert _score(bench, ret) == 0.0


def test_conforming_object_scores_hundred(bench):
    ret = [
        {
            "generated": '{"name": "John Smith", "age": 34}',
            "expected": None,
            "schema": SCHEMA,
            "structured": True,
        }
    ]
    assert _score(bench, ret) == 100.0


def test_wrong_value_type_fails_validation(bench):
    ret = [
        {
            "generated": '{"name": "John Smith", "age": "thirty-four"}',
            "expected": None,
            "schema": SCHEMA,
            "structured": True,
        }
    ]
    assert _score(bench, ret) == 0.0


def test_schema_given_as_string_is_accepted(bench):
    ret = [
        {
            "generated": '{"name": "John Smith", "age": 34}',
            "expected": None,
            "schema": json.dumps(SCHEMA),
            "structured": True,
        }
    ]
    assert _score(bench, ret) == 100.0


def test_reference_completion_semantic_equality(bench):
    # Key order and whitespace differ; the content matches.
    ret = [
        {
            "generated": '{\n  "age": 34,\n  "name": "John Smith"\n}',
            "expected": '{"name": "John Smith", "age": 34}',
            "schema": SCHEMA,
            "structured": True,
        }
    ]
    assert _score(bench, ret) == 100.0


def test_reference_completion_mismatch_fails(bench):
    ret = [
        {
            "generated": '{"name": "Jane Doe", "age": 34}',
            "expected": '{"name": "John Smith", "age": 34}',
            "schema": SCHEMA,
            "structured": True,
        }
    ]
    assert _score(bench, ret) == 0.0


def test_unstructured_request_keeps_parseability_semantics(bench):
    # Under --structured-output-ratio < 1 the unstructured share was never
    # asked to follow a schema; grading stays at parseability for them.
    ret = [
        {
            "generated": 'here you go: {"anything": "free-form"}',
            "expected": None,
            "schema": SCHEMA,
            "structured": False,
        }
    ]
    assert _score(bench, ret) == 100.0


def test_unparsable_response_fails(bench):
    ret = [
        {
            "generated": "no json here",
            "expected": None,
            "schema": SCHEMA,
            "structured": True,
        }
    ]
    assert _score(bench, ret) == 0.0


def test_string_values_keep_their_spaces(bench):
    # The old grader stripped every space before parsing, so a payload with
    # spaces inside a string was graded as different content than produced.
    ret = [
        {
            "generated": '{"name": "John Smith", "age": 34}',
            "expected": '{"name": "John Smith", "age": 34}',
            "schema": SCHEMA,
            "structured": True,
        }
    ]
    assert _score(bench, ret) == 100.0
