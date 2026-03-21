# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Minimal structural tests for the MoE oracle class hierarchy.

Verifies that all oracle classes correctly inherit from MoEKernelOracle
and that module-level wrapper functions delegate to oracle instances.
No GPU or model weights required.
"""

import pytest

from vllm.model_executor.layers.fused_moe.oracle.base import MoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    Fp8MoEKernelOracle,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    Mxfp4MoEKernelOracle,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
    Mxfp8MoEKernelOracle,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    NvFp4MoEKernelOracle,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    UnquantizedMoEKernelOracle,
)

ALL_ORACLE_CLASSES = [
    Fp8MoEKernelOracle,
    NvFp4MoEKernelOracle,
    Mxfp8MoEKernelOracle,
    Mxfp4MoEKernelOracle,
    UnquantizedMoEKernelOracle,
]


@pytest.mark.parametrize("oracle_cls", ALL_ORACLE_CLASSES)
def test_oracle_inherits_from_base(oracle_cls):
    """Each oracle class must inherit from MoEKernelOracle."""
    assert issubclass(oracle_cls, MoEKernelOracle)


@pytest.mark.parametrize("oracle_cls", ALL_ORACLE_CLASSES)
def test_oracle_is_instantiable(oracle_cls):
    """Each oracle class must be instantiable (all abstract methods implemented)."""
    oracle = oracle_cls()
    assert isinstance(oracle, MoEKernelOracle)


@pytest.mark.parametrize("oracle_cls", ALL_ORACLE_CLASSES)
def test_oracle_has_required_methods(oracle_cls):
    """Each oracle must implement the 4 standard methods + 2 helpers."""
    oracle = oracle_cls()
    # 4 standard methods from the issue
    assert callable(getattr(oracle, "select_backend", None))
    assert callable(getattr(oracle, "convert_to_kernel_format", None))
    assert callable(getattr(oracle, "make_quant_config", None))
    assert callable(getattr(oracle, "make_kernel", None))
    # 2 helper methods
    assert callable(getattr(oracle, "backend_to_kernel_cls", None))
    assert callable(getattr(oracle, "map_backend", None))


def test_fp8_module_wrapper_delegates_to_oracle():
    """Module-level wrapper must delegate to the oracle singleton."""
    from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_mod

    assert isinstance(fp8_mod._oracle, Fp8MoEKernelOracle)
    # map_fp8_backend should delegate to _oracle.map_backend
    backend = fp8_mod.map_fp8_backend("triton")
    assert backend == Fp8MoeBackend.TRITON


def test_nvfp4_module_wrapper_delegates_to_oracle():
    """Module-level wrapper must delegate to the oracle singleton."""
    from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as nvfp4_mod

    assert isinstance(nvfp4_mod._oracle, NvFp4MoEKernelOracle)
    backend = nvfp4_mod.map_nvfp4_backend("marlin")
    assert backend == NvFp4MoeBackend.MARLIN


def test_mxfp4_module_wrapper_delegates_to_oracle():
    """Module-level wrapper must delegate to the oracle singleton."""
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mxfp4_mod

    assert isinstance(mxfp4_mod._oracle, Mxfp4MoEKernelOracle)
    backend = mxfp4_mod.map_mxfp4_backend("triton")
    assert backend == Mxfp4MoeBackend.TRITON


def test_unquantized_module_wrapper_delegates_to_oracle():
    """Module-level wrapper must delegate to the oracle singleton."""
    from vllm.model_executor.layers.fused_moe.oracle import (
        unquantized as unquant_mod,
    )

    assert isinstance(unquant_mod._oracle, UnquantizedMoEKernelOracle)
    backend = unquant_mod.map_unquantized_backend("triton")
    assert backend == UnquantizedMoeBackend.TRITON


def test_map_backend_invalid_raises():
    """map_backend should raise ValueError for unsupported backend strings."""
    oracle = Fp8MoEKernelOracle()
    with pytest.raises(ValueError, match="not supported"):
        oracle.map_backend("nonexistent_backend")
