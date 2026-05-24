# SPDX-License-Identifier: Apache-2.0
"""TDD tests for P49 — interface contract validation helpers.

Covers:
- `validate_impl` — pass when all required attrs+methods present
- Missing required attr → raises `GenesisInterfaceMismatch` with details
- Wrong-type required attr → raises with got/expected info
- Missing required method → raises with method name
- ANY sentinel accepts any non-None
- String type matching (by class name) — avoids eager imports
- Tuple of expected types matches any of them
- Optional attrs absent → log note, no raise
- `validate_method_signature` — min param count, expected param names
- `assert_shape_compat` — ndim, min_shape, dtype checks; fallback
  when torch not available (module still importable on CPU-only)
- `describe_impl` — produces JSON-safe snapshot
- Full-path integration: scenario where a "future upstream refactor"
  removes an attr and our guard catches it before rebind happens

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import pytest


class TestValidateImplPasses:
    def test_all_required_present(self):
        from vllm._genesis.interface_guard import validate_impl

        class Good:
            num_heads = 32
            num_kv_heads = 4
            head_size = 128

            def _continuation_prefill(self, *a, **kw):
                return None

            def _flash_attn_varlen(self, *a, **kw):
                return None

        # Should not raise
        validate_impl(
            Good,
            role="Good",
            required_attrs={
                "num_heads": int,
                "num_kv_heads": int,
                "head_size": int,
            },
            required_methods=[
                "_continuation_prefill",
                "_flash_attn_varlen",
            ],
        )

    def test_any_sentinel(self):
        from vllm._genesis.interface_guard import validate_impl, ANY

        class X:
            config = "anything"
        validate_impl(X, role="X", required_attrs={"config": ANY})

    def test_string_type_match(self):
        from vllm._genesis.interface_guard import validate_impl

        class FakeTQConfig:
            pass

        class Impl:
            tq_config = FakeTQConfig()

        validate_impl(
            Impl, role="Impl",
            required_attrs={"tq_config": "FakeTQConfig"},
        )

    def test_tuple_of_types(self):
        from vllm._genesis.interface_guard import validate_impl

        class X:
            val = 42  # int or float accepted
        validate_impl(X, role="X", required_attrs={"val": (int, float)})


class TestValidateImplFailures:
    def test_missing_required_attr(self):
        from vllm._genesis.interface_guard import (
            validate_impl, GenesisInterfaceMismatch,
        )

        class Incomplete:
            num_heads = 32
            # missing num_kv_heads

        with pytest.raises(GenesisInterfaceMismatch) as excinfo:
            validate_impl(
                Incomplete, role="Incomplete",
                required_attrs={"num_heads": int, "num_kv_heads": int},
            )
        assert "num_kv_heads" in str(excinfo.value)
        assert "Incomplete" in str(excinfo.value)

    def test_wrong_type_attr(self):
        from vllm._genesis.interface_guard import (
            validate_impl, GenesisInterfaceMismatch,
        )

        class Wrong:
            num_heads = "not an int"

        with pytest.raises(GenesisInterfaceMismatch) as excinfo:
            validate_impl(
                Wrong, role="Wrong",
                required_attrs={"num_heads": int},
            )
        assert "num_heads" in str(excinfo.value)
        assert "str" in str(excinfo.value).lower()

    def test_missing_required_method(self):
        from vllm._genesis.interface_guard import (
            validate_impl, GenesisInterfaceMismatch,
        )

        class NoMethod:
            pass

        with pytest.raises(GenesisInterfaceMismatch) as excinfo:
            validate_impl(
                NoMethod, role="NoMethod",
                required_methods=["_continuation_prefill"],
            )
        assert "_continuation_prefill" in str(excinfo.value)

    def test_multiple_errors_aggregated(self):
        from vllm._genesis.interface_guard import (
            validate_impl, GenesisInterfaceMismatch,
        )

        class Broken:
            wrong_type = "str"
            # missing expected_attr
            # missing expected_method

        try:
            validate_impl(
                Broken, role="Broken",
                required_attrs={
                    "wrong_type": int,
                    "expected_attr": int,
                },
                required_methods=["expected_method"],
            )
        except GenesisInterfaceMismatch as e:
            assert "expected_attr" in str(e)
            assert "expected_method" in str(e)
            assert "wrong_type" in str(e)
            return
        pytest.fail("expected GenesisInterfaceMismatch")


class TestOptionalAttrs:
    def test_absent_optional_doesnt_raise(self, caplog):
        from vllm._genesis.interface_guard import validate_impl
        import logging

        class Bare:
            required_a = 1

        caplog.set_level(logging.INFO)
        # Should not raise
        validate_impl(
            Bare, role="Bare",
            required_attrs={"required_a": int},
            optional_attrs={"optional_b": int},
        )
        # Log should note the optional absence
        assert any(
            "optional" in r.message.lower() for r in caplog.records
        )


class TestMethodSignature:
    def test_min_param_count_pass(self):
        from vllm._genesis.interface_guard import validate_method_signature

        class X:
            def m(self, a, b, c):
                pass

        validate_method_signature(
            X, "m", role="X.m", expected_min_params=3,
        )

    def test_min_param_count_fail(self):
        from vllm._genesis.interface_guard import (
            validate_method_signature, GenesisInterfaceMismatch,
        )

        class X:
            def m(self, a):
                pass

        with pytest.raises(GenesisInterfaceMismatch) as excinfo:
            validate_method_signature(
                X, "m", role="X.m", expected_min_params=3,
            )
        assert "params" in str(excinfo.value).lower()

    def test_expected_param_names_pass(self):
        from vllm._genesis.interface_guard import validate_method_signature

        class X:
            def m(self, query, key, value):
                pass

        validate_method_signature(
            X, "m", role="X.m",
            expected_min_params=3,
            expected_param_names=["query", "key", "value"],
        )

    def test_expected_param_names_missing_raises(self):
        from vllm._genesis.interface_guard import (
            validate_method_signature, GenesisInterfaceMismatch,
        )

        class X:
            def m(self, q, k):
                pass

        with pytest.raises(GenesisInterfaceMismatch) as excinfo:
            validate_method_signature(
                X, "m", role="X.m",
                expected_min_params=2,
                expected_param_names=["query", "key"],  # renamed!
            )
        assert "query" in str(excinfo.value)


class TestAssertShapeCompat:
    def test_non_tensor_raises(self):
        from vllm._genesis.interface_guard import (
            assert_shape_compat, GenesisInterfaceMismatch,
        )
        with pytest.raises(GenesisInterfaceMismatch):
            assert_shape_compat(
                "not a tensor", role="test", expected_ndim=2,
            )

    def test_ndim_pass(self):
        import torch
        from vllm._genesis.interface_guard import assert_shape_compat
        t = torch.zeros(1, 2, 3)
        assert_shape_compat(t, role="test", expected_ndim=3)

    def test_ndim_fail(self):
        import torch
        from vllm._genesis.interface_guard import (
            assert_shape_compat, GenesisInterfaceMismatch,
        )
        t = torch.zeros(1, 2, 3)
        with pytest.raises(GenesisInterfaceMismatch):
            assert_shape_compat(t, role="test", expected_ndim=4)

    def test_min_shape_pass(self):
        import torch
        from vllm._genesis.interface_guard import assert_shape_compat
        t = torch.zeros(10, 20)
        assert_shape_compat(t, role="test", min_shape=(1, 5))

    def test_min_shape_fail(self):
        import torch
        from vllm._genesis.interface_guard import (
            assert_shape_compat, GenesisInterfaceMismatch,
        )
        t = torch.zeros(10, 3)
        with pytest.raises(GenesisInterfaceMismatch):
            assert_shape_compat(t, role="test", min_shape=(10, 5))

    def test_dtype_check(self):
        import torch
        from vllm._genesis.interface_guard import (
            assert_shape_compat, GenesisInterfaceMismatch,
        )
        t = torch.zeros(2, 2, dtype=torch.float32)
        # Pass
        assert_shape_compat(
            t, role="test", expected_dtype=torch.float32,
        )
        # Fail
        with pytest.raises(GenesisInterfaceMismatch):
            assert_shape_compat(
                t, role="test", expected_dtype=torch.float16,
            )


class TestDescribeImpl:
    def test_snapshot_contains_class_and_attrs(self):
        from vllm._genesis.interface_guard import describe_impl

        class X:
            a = 1
            b = "text"

            def m(self):
                pass

        snap = describe_impl(X, role="X")
        assert snap["role"] == "X"
        assert snap["class"] == "type"
        assert "a" in snap["attrs"]
        assert "m" in snap["methods"]
        assert snap["attrs"]["a"] == "int"

    def test_snapshot_is_json_serialisable(self):
        import json
        from vllm._genesis.interface_guard import describe_impl

        class X:
            v = 42

        snap = describe_impl(X, role="X")
        json.dumps(snap)  # must not raise


class TestFullDriftScenario:
    """End-to-end: simulate a future upstream that DROPPED an attr
    our P38 body relies on, and assert that the guard catches it
    before rebind."""

    def test_future_upstream_drift_is_caught(self):
        from vllm._genesis.interface_guard import (
            validate_impl, GenesisInterfaceMismatch,
        )

        # Simulated "future-upstream refactored" class: missing
        # `_mse_bytes`, which our P38 body reads.
        class FutureTQImpl:
            num_heads = 40
            num_kv_heads = 4
            head_size = 256
            # _mse_bytes missing — upstream renamed to _packed_mse!
            _packed_mse = 8

            def _continuation_prefill(self, *a, **kw):
                pass

            def _flash_attn_varlen(self, *a, **kw):
                pass

        # Genesis apply() would guard on attrs our body READS. In
        # reality we only check method presence in v7.8 P38 (instance
        # attrs are set in __init__, class-level check can't see them).
        # This test demonstrates how a MORE-STRICT guard would catch
        # the drift if we extended required_attrs.
        with pytest.raises(GenesisInterfaceMismatch) as excinfo:
            validate_impl(
                FutureTQImpl, role="FutureTQImpl",
                required_attrs={
                    "_mse_bytes": int,    # missing!
                    "_val_data_bytes": int,  # also missing
                },
                required_methods=["_continuation_prefill"],
            )
        assert "_mse_bytes" in str(excinfo.value)
        assert "_val_data_bytes" in str(excinfo.value)
