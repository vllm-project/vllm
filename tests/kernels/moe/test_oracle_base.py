# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC and UnquantizedMoEKernelOracle."""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    UnquantizedMoEKernelOracle,
)


def make_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=16,
        intermediate_size=32,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )


class TestMoEKernelOracleABC:
    """Verify the ABC enforces the oracle contract."""

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
        """UnquantizedMoEKernelOracle implements all abstract methods."""

        oracle = UnquantizedMoEKernelOracle()
        assert oracle is not None

    def test_abstract_methods_present(self):
        """Core methods are available on the concrete class."""
        oracle = UnquantizedMoEKernelOracle()
        assert hasattr(oracle, "backend_enum_cls")
        assert hasattr(oracle, "get_priority_backends")
        assert hasattr(oracle, "backend_to_kernel_cls")
        assert hasattr(oracle, "map_backend")
        assert hasattr(oracle, "select_backend")
        assert hasattr(oracle, "convert_to_kernel_format")
        assert hasattr(oracle, "make_quant_config")
        assert hasattr(oracle, "make_kernel")


class TestUnquantizedOracleBaseDefaults:
    """Base/default behaviour exposed by the concrete oracle."""

    def test_backend_enum_cls(self):
        oracle = UnquantizedMoEKernelOracle()
        assert oracle.backend_enum_cls() is UnquantizedMoeBackend

    def test_unquantized_oracle_has_no_quant_config(self):
        oracle = UnquantizedMoEKernelOracle()
        with pytest.raises(NotImplementedError, match="UnquantizedMoEKernelOracle"):
            oracle.make_quant_config()


class TestUnquantizedOracleStaticMethods:
    """Utility methods on UnquantizedMoEKernelOracle."""

    def test_backend_to_kernel_cls_known_backend(self):
        oracle = UnquantizedMoEKernelOracle()
        classes = oracle.backend_to_kernel_cls(UnquantizedMoeBackend.TRITON)

        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        assert classes == [TritonExperts]

    def test_backend_to_kernel_cls_unknown_backend_raises(self):
        oracle = UnquantizedMoEKernelOracle()
        with pytest.raises(ValueError, match="Unknown"):
            oracle.backend_to_kernel_cls(UnquantizedMoeBackend.CPU)

    def test_map_backend_known(self):
        oracle = UnquantizedMoEKernelOracle()
        backend = oracle.map_backend("triton")
        assert backend == UnquantizedMoeBackend.TRITON

    def test_map_backend_unknown_raises(self):
        oracle = UnquantizedMoEKernelOracle()
        with pytest.raises(ValueError, match="not supported for unquantized MoE"):
            oracle.map_backend("nonexistent")  # type: ignore[arg-type]


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

        moe_config = make_moe_config()
        oracle = UnquantizedMoEKernelOracle()
        backend, experts_cls = oracle.select_backend(moe_config)

        assert backend == UnquantizedMoeBackend.CPU
        assert experts_cls is None

    @patch("vllm.model_executor.layers.fused_moe.oracle.unquantized.current_platform")
    def test_tpu_platform_returns_tpu_none(self, mock_platform):
        mock_platform.is_cpu.return_value = False
        mock_platform.is_tpu.return_value = True
        mock_platform.is_out_of_tree.return_value = False

        moe_config = make_moe_config()
        oracle = UnquantizedMoEKernelOracle()
        backend, experts_cls = oracle.select_backend(moe_config)

        assert backend == UnquantizedMoeBackend.TPU
        assert experts_cls is None
