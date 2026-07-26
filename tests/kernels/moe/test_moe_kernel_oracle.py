# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

This file exercises two layers:

1. The base class's generic `select_backend` logic — the priority
   loop, `is_supported_config` iteration, and user-explicit backend
   override behave correctly. Driven by an in-test stub oracle so the
   tests don't depend on any concrete oracle's data or platform
   conditions.

2. Each concrete oracle — `backend_enum_cls`,
   `get_priority_backends`, `backend_to_kernel_cls`, `map_backend`,
   and with its specific guards.
"""

from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    Int8MoeBackend,
    Int8MoEKernelOracle,
)
from vllm.model_executor.layers.fused_moe.oracle.w4a8_int8 import (
    W4A8Int8MoeBackend,
    W4A8Int8MoEKernelOracle,
)

# ---------------------------------------------------------------------------
# Stubs for exercising the base class's generic select_backend logic.
# ---------------------------------------------------------------------------


class _StubBackend(Enum):
    A = "A"
    B = "B"


class _StubSupportedExperts:
    """Stub experts class whose `is_supported_config` always returns True."""

    @staticmethod
    def is_supported_config(cls, config, weight_key, activation_key, fmt):
        return True, None


class _StubUnsupportedExperts:
    @staticmethod
    def is_supported_config(cls, config, weight_key, activation_key, fmt):
        return False, "stub: not supported"


class _StubOracle(MoEKernelOracle[_StubBackend]):
    """In-test stub oracle for exercising base class behaviour."""

    def __init__(self, priority, kernel_map, mapping=None):
        self._priority = priority
        self._kernel_map = kernel_map
        self._mapping = mapping or {}

    def backend_enum_cls(self):
        return _StubBackend

    def get_priority_backends(self, moe_config):
        return self._priority

    def backend_to_kernel_cls(self, backend):
        return self._kernel_map[backend]

    def map_backend(self, runner_backend):
        if runner_backend not in self._mapping:
            raise ValueError(f"unknown: {runner_backend}")
        return self._mapping[runner_backend]


def _stub_moe_config(moe_backend="auto", use_batched=False):
    config = MagicMock()
    config.moe_backend = moe_backend
    config.moe_parallel_config.use_batched_activation_format = use_batched
    return config


class TestGenericSelectBackend:
    """The base class's generic `select_backend` should
    - iterate `get_priority_backends` in order
    - for each backend, iterate `backend_to_kernel_cls` candidates,
      asking each one's `is_supported_config` until one accepts
    - return the first supported (backend, kernel_class) tuple
    - raise NotImplementedError when none of the candidates supports
      the configuration
    - honour a user-explicit `moe_backend` override (raise on unsupported)
    """

    def test_first_supported_backend_wins(self) -> None:
        oracle = _StubOracle(
            priority=[_StubBackend.A, _StubBackend.B],
            kernel_map={
                _StubBackend.A: [_StubSupportedExperts],
                _StubBackend.B: [_StubSupportedExperts],
            },
        )
        backend, k_cls = oracle.select_backend(_stub_moe_config())
        assert backend == _StubBackend.A
        assert k_cls is _StubSupportedExperts

    def test_skips_unsupported_backend(self) -> None:
        oracle = _StubOracle(
            priority=[_StubBackend.A, _StubBackend.B],
            kernel_map={
                _StubBackend.A: [_StubUnsupportedExperts],
                _StubBackend.B: [_StubSupportedExperts],
            },
        )
        backend, k_cls = oracle.select_backend(_stub_moe_config())
        assert backend == _StubBackend.B
        assert k_cls is _StubSupportedExperts

    def test_skips_unsupported_class_within_backend(self) -> None:
        oracle = _StubOracle(
            priority=[_StubBackend.A],
            kernel_map={
                _StubBackend.A: [_StubUnsupportedExperts, _StubSupportedExperts],
            },
        )
        backend, k_cls = oracle.select_backend(_stub_moe_config())
        assert backend == _StubBackend.A
        assert k_cls is _StubSupportedExperts

    def test_all_unsupported_raises(self) -> None:
        oracle = _StubOracle(
            priority=[_StubBackend.A, _StubBackend.B],
            kernel_map={
                _StubBackend.A: [_StubUnsupportedExperts],
                _StubBackend.B: [_StubUnsupportedExperts],
            },
        )
        with pytest.raises(NotImplementedError, match="No _StubOracle backend"):
            oracle.select_backend(_stub_moe_config())

    def test_user_explicit_override_picks_requested(self) -> None:
        oracle = _StubOracle(
            priority=[_StubBackend.A],
            kernel_map={
                _StubBackend.A: [_StubSupportedExperts],
                _StubBackend.B: [_StubSupportedExperts],
            },
            mapping={"b": _StubBackend.B},
        )
        backend, _ = oracle.select_backend(_stub_moe_config(moe_backend="b"))
        assert backend == _StubBackend.B

    def test_user_explicit_unsupported_raises(self) -> None:
        oracle = _StubOracle(
            priority=[_StubBackend.A],
            kernel_map={
                _StubBackend.A: [_StubUnsupportedExperts],
            },
            mapping={"a": _StubBackend.A},
        )
        with pytest.raises(ValueError, match="does not support"):
            oracle.select_backend(_stub_moe_config(moe_backend="a"))


class TestInt8OracleDataDeclarations:
    """Int8MoEKernelOracle declares TRITON + HUMMING + CPU backends, maps
    `triton` from the user-facing override, and prioritises CPU on CPU
    platforms."""

    def test_backend_enum_cls(self) -> None:
        assert Int8MoEKernelOracle().backend_enum_cls() is Int8MoeBackend

    def test_get_priority_backends_non_cpu(self) -> None:
        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.int8.current_platform.is_cpu",
            return_value=False,
        ):
            assert Int8MoEKernelOracle().get_priority_backends(_stub_moe_config()) == [
                Int8MoeBackend.TRITON,
                Int8MoeBackend.HUMMING,
                Int8MoeBackend.CPU,
            ]

    def test_get_priority_backends_cpu_moves_cpu_front(self) -> None:
        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.int8.current_platform.is_cpu",
            return_value=True,
        ):
            assert Int8MoEKernelOracle().get_priority_backends(_stub_moe_config()) == [
                Int8MoeBackend.CPU,
                Int8MoeBackend.TRITON,
                Int8MoeBackend.HUMMING,
            ]

    def test_backend_to_kernel_cls_triton(self) -> None:
        out = Int8MoEKernelOracle().backend_to_kernel_cls(Int8MoeBackend.TRITON)
        assert out == [TritonExperts]

    def test_map_backend_triton(self) -> None:
        assert Int8MoEKernelOracle().map_backend("triton") is Int8MoeBackend.TRITON

    def test_map_backend_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            Int8MoEKernelOracle().map_backend("flashinfer")


class TestW4A8Int8OracleDataDeclarations:
    """W4A8Int8MoEKernelOracle is a CPU-only oracle with a single
    backend (CPU_INT4) that requires monolithic experts."""

    def test_backend_enum_cls(self) -> None:
        assert W4A8Int8MoEKernelOracle().backend_enum_cls() is W4A8Int8MoeBackend

    def test_map_backend_cpu(self) -> None:
        assert (
            W4A8Int8MoEKernelOracle().map_backend("cpu") is W4A8Int8MoeBackend.CPU_INT4
        )

    def test_map_backend_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            W4A8Int8MoEKernelOracle().map_backend("triton")

    def test_select_backend_raises_on_non_cpu(self) -> None:
        with (
            patch(
                "vllm.model_executor.layers.fused_moe.oracle.w4a8_int8."
                "current_platform.is_cpu",
                return_value=False,
            ),
            pytest.raises(NotImplementedError, match="CPU platforms"),
        ):
            W4A8Int8MoEKernelOracle().select_backend(_stub_moe_config())

    def test_make_kernel_requires_monolithic_experts(self) -> None:
        # Pass a non-monolithic stub class to trigger the guard.
        class _NonMonolithicExperts:
            pass

        with pytest.raises(ValueError, match="monolithic"):
            W4A8Int8MoEKernelOracle().make_kernel(
                MagicMock(),
                _stub_moe_config(),
                _NonMonolithicExperts,
                W4A8Int8MoeBackend.CPU_INT4,
            )
