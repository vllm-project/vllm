# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

These tests deliberately avoid constructing a full FusedMoEConfig (which
is expensive) and instead exercise:
  1. Default implementations on the base class.
  2. Bit-identical delegation from UnquantizedMoEKernelOracle to the
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

    def test_map_backend_triton(self) -> None:
        oracle = UnquantizedMoEKernelOracle()
        assert oracle.map_backend("triton") is UnquantizedMoeBackend.TRITON
        assert oracle.map_backend("aiter") is UnquantizedMoeBackend.AITER

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
