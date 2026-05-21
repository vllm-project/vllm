# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

These tests deliberately avoid constructing a full FusedMoEConfig (which
is expensive) and instead exercise:
  1. ABC contract enforcement (cannot instantiate without all abstract
     methods).
  2. Default implementations on the base class.
  3. Bit-identical delegation from UnquantizedMoEKernelOracle to the
     existing module-level functions in oracle/unquantized.py.
"""

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import (
    MoEKernelOracle,
    UnquantizedMoEKernelOracle,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)


class TestABCContract:
    """ABC enforcement: subclasses must implement all abstract methods."""

    def test_cannot_instantiate_base_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            MoEKernelOracle()

    def test_cannot_instantiate_with_missing_abstract_method(self) -> None:
        # Stub subclass that forgets `make_kernel`.
        class BrokenOracle(MoEKernelOracle[UnquantizedMoeBackend]):
            def backend_enum_cls(self):
                return UnquantizedMoeBackend

            def get_priority_backends(self, moe_config):
                return []

            def backend_to_kernel_cls(self, backend):
                raise NotImplementedError

            def map_backend(self, runner_backend):
                raise NotImplementedError

            def select_backend(self, moe_config):
                raise NotImplementedError

            # NOTE: deliberately no `make_kernel` implementation.

        with pytest.raises(TypeError, match="abstract"):
            BrokenOracle()


class TestBaseDefaults:
    """Optional methods on the base class have sane defaults."""

    def test_make_quant_config_default_raises(self) -> None:
        oracle = UnquantizedMoEKernelOracle()
        with pytest.raises(
            NotImplementedError, match="does not implement make_quant_config"
        ):
            oracle.make_quant_config()

    def test_convert_to_kernel_format_default_is_identity(self) -> None:
        """A subclass that doesn't override convert_to_kernel_format gets
        the identity default. Unquantized DOES override, so we test the
        default via a minimal stub subclass."""

        class StubOracle(MoEKernelOracle[UnquantizedMoeBackend]):
            def backend_enum_cls(self):
                return UnquantizedMoeBackend

            def get_priority_backends(self, moe_config):
                return []

            def backend_to_kernel_cls(self, backend):
                raise NotImplementedError

            def map_backend(self, runner_backend):
                raise NotImplementedError

            def select_backend(self, moe_config):
                raise NotImplementedError

            def make_kernel(
                self,
                quant_config,
                moe_config,
                backend,
                experts_cls,
                routing_tables=None,
            ):
                raise NotImplementedError

        import torch

        w13 = torch.empty(2, 3)
        w2 = torch.empty(3, 2)
        out13, out2 = StubOracle().convert_to_kernel_format(
            UnquantizedMoeBackend.TRITON, layer=None, w13_weight=w13, w2_weight=w2
        )
        assert out13 is w13
        assert out2 is w2


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

    def test_backend_enum_cls_returns_unquantized_enum(self) -> None:
        assert UnquantizedMoEKernelOracle().backend_enum_cls() is (
            UnquantizedMoeBackend
        )

    def test_backend_to_kernel_cls_triton_matches_module_level(self) -> None:
        # TRITON backend's mapping is unconditional (no platform check),
        # so we can compare without constructing a FusedMoEConfig.
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            backend_to_kernel_cls,
        )

        cls_via_module = backend_to_kernel_cls(UnquantizedMoeBackend.TRITON)
        cls_via_oracle = UnquantizedMoEKernelOracle().backend_to_kernel_cls(
            UnquantizedMoeBackend.TRITON
        )
        assert cls_via_module is cls_via_oracle is TritonExperts

    def test_map_backend_triton(self) -> None:
        oracle = UnquantizedMoEKernelOracle()
        assert oracle.map_backend("triton") is UnquantizedMoeBackend.TRITON
        assert oracle.map_backend("aiter") is UnquantizedMoeBackend.AITER

    def test_map_backend_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            UnquantizedMoEKernelOracle().map_backend("not_a_real_backend")

    def test_select_backend_delegates(self) -> None:
        """`select_backend` should call the module-level function with
        the moe_config argument forwarded unchanged."""
        sentinel_moe_config = object()
        sentinel_result = (UnquantizedMoeBackend.TRITON, TritonExperts)

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "select_unquantized_moe_backend",
            return_value=sentinel_result,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().select_backend(sentinel_moe_config)

        mocked.assert_called_once_with(sentinel_moe_config)
        assert out is sentinel_result

    def test_get_priority_backends_delegates(self) -> None:
        sentinel_moe_config = object()
        sentinel_list: list[UnquantizedMoeBackend] = [UnquantizedMoeBackend.TRITON]

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "_get_priority_backends",
            return_value=sentinel_list,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().get_priority_backends(
                sentinel_moe_config
            )

        mocked.assert_called_once_with(sentinel_moe_config)
        assert out is sentinel_list

    def test_convert_to_kernel_format_delegates(self) -> None:
        import torch

        w13 = torch.empty(2, 3)
        w2 = torch.empty(3, 2)
        sentinel = (torch.empty(0), torch.empty(0))

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "convert_to_unquantized_kernel_format",
            return_value=sentinel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().convert_to_kernel_format(
                UnquantizedMoeBackend.AITER, layer=None, w13_weight=w13, w2_weight=w2
            )

        mocked.assert_called_once_with(UnquantizedMoeBackend.AITER, None, w13, w2)
        assert out is sentinel

    def test_make_kernel_delegates(self) -> None:
        quant_config = object()
        moe_config = object()
        experts_cls = TritonExperts
        sentinel_kernel = object()

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "make_unquantized_moe_kernel",
            return_value=sentinel_kernel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().make_kernel(
                quant_config,
                moe_config,
                UnquantizedMoeBackend.TRITON,
                experts_cls,
            )

        mocked.assert_called_once_with(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            experts_cls,
            None,  # routing_tables default
        )
        assert out is sentinel_kernel
