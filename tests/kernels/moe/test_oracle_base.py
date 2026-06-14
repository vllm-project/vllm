# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC and UnquantizedOracle."""

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    UnquantizedOracle,
)


class TestMoEKernelOracleABC:
    """Verify the ABC enforces the 4-method contract."""

    def test_cannot_instantiate_abc(self):
        """MoEKernelOracle is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            MoEKernelOracle()  # type: ignore[abstract]

    def test_concrete_subclass_requires_all_methods(self):
        """A subclass missing any abstract method fails to instantiate."""

        class IncompleteOracle(MoEKernelOracle[UnquantizedMoeBackend]):
            pass

        with pytest.raises(TypeError):
            IncompleteOracle()  # type: ignore[abstract]

    def test_full_subclass_instantiates(self):
        """UnquantizedOracle implements all abstract methods."""

        oracle = UnquantizedOracle()
        assert oracle is not None

    def test_abstract_methods_present(self):
        """All 4 core methods are available on the concrete class."""
        oracle = UnquantizedOracle()
        assert hasattr(oracle, "select_backend")
        assert hasattr(oracle, "convert_to_kernel_format")
        assert hasattr(oracle, "make_quant_config")
        assert hasattr(oracle, "make_kernel")


class TestUnquantizedOracleSharedHelpers:
    """Shared helper methods from the base class."""

    def test_oracle_name(self):
        assert UnquantizedOracle._oracle_name() == "Unquantized"

    def test_make_log_backend_basic(self):
        msg = UnquantizedOracle._make_log_backend(UnquantizedMoeBackend.TRITON)
        assert "TRITON" in msg
        assert "Unquantized" in msg

    def test_make_log_backend_with_available_list(self):
        msg = UnquantizedOracle._make_log_backend(
            UnquantizedMoeBackend.TRITON,
            available_backends=[
                UnquantizedMoeBackend.TRITON,
                UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            ],
        )
        assert "TRITON" in msg
        assert "potential backends" in msg
        assert "FlashInfer TRTLLM" in msg

    def test_make_log_unsupported_with_reason(self):
        msg = UnquantizedOracle._make_log_unsupported(
            UnquantizedMoeBackend.CPU, "CPU-only platform"
        )
        assert "CPU" in msg
        assert "CPU-only platform" in msg

    def test_make_log_unsupported_without_reason(self):
        msg = UnquantizedOracle._make_log_unsupported(UnquantizedMoeBackend.CPU, None)
        assert "CPU" in msg
        assert "does not support the deployment configuration" in msg


class TestUnquantizedOracleStaticMethods:
    """Static utility methods on UnquantizedOracle."""

    def test_backend_to_kernel_cls_known_backend(self):
        cls = UnquantizedOracle.backend_to_kernel_cls(UnquantizedMoeBackend.TRITON)
        assert cls is not None
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        assert cls is TritonExperts

    def test_backend_to_kernel_cls_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            UnquantizedOracle.backend_to_kernel_cls(UnquantizedMoeBackend.CPU)

    def test_map_backend_known(self):
        backend = UnquantizedOracle.map_backend("triton")
        assert backend == UnquantizedMoeBackend.TRITON

    def test_map_backend_unknown(self):
        backend = UnquantizedOracle.map_backend("nonexistent")
        assert backend is None


class TestUnquantizedOracleModuleWrappers:
    """Module-level wrapper functions preserve backward compatibility."""

    def test_select_wrapper_exists(self):
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            select_unquantized_moe_backend,
        )

        assert callable(select_unquantized_moe_backend)

    def test_convert_wrapper_exists(self):
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            convert_to_unquantized_kernel_format,
        )

        assert callable(convert_to_unquantized_kernel_format)

    def test_make_kernel_wrapper_exists(self):
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            make_unquantized_moe_kernel,
        )

        assert callable(make_unquantized_moe_kernel)


class TestUnquantizedOracleSelectBackend:
    """select_backend behaves correctly for special platform paths."""

    @patch("vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform")
    def test_cpu_platform_returns_cpu_none(self, mock_platform):
        mock_platform.is_cpu.return_value = True
        mock_platform.is_tpu.return_value = False
        mock_platform.is_out_of_tree.return_value = False

        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
        )

        moe_config = FusedMoEConfig(
            num_experts=8,
            top_k=2,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        )
        oracle = UnquantizedOracle()
        backend, experts_cls = oracle.select_backend(moe_config)

        assert backend == UnquantizedMoeBackend.CPU
        assert experts_cls is None

    @patch("vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform")
    def test_tpu_platform_returns_tpu_none(self, mock_platform):
        mock_platform.is_cpu.return_value = False
        mock_platform.is_tpu.return_value = True
        mock_platform.is_out_of_tree.return_value = False

        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
        )

        moe_config = FusedMoEConfig(
            num_experts=8,
            top_k=2,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        )
        oracle = UnquantizedOracle()
        backend, experts_cls = oracle.select_backend(moe_config)

        assert backend == UnquantizedMoeBackend.TPU
        assert experts_cls is None
