# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

This file contains a single canonical demonstration that
`UnquantizedMoEKernelOracle` methods delegate one-to-one to the
existing module-level functions in `oracle/unquantized.py`. Each method
on `UnquantizedMoEKernelOracle` follows the same `return module_fn(args)`
pattern, so verifying delegation for one method (`make_kernel`) gives
high confidence in the rest.
"""

from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
    Nvfp4QuantizationEmulationTritonExperts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as nvfp4_oracle
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

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


@pytest.mark.parametrize(
    "strict,expected",
    [(False, NvFp4MoeBackend.MARLIN), (True, NvFp4MoeBackend.EMULATION)],
)
def test_strict_quant_scheme_skips_activation_fallbacks(monkeypatch, strict, expected):
    """Marlin runs NVFP4 W4A4 layers as W4A16. VLLM_STRICT_QUANT_SCHEME keeps the
    backend priority order but skips such fallbacks; emulation honors W4A4."""
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", str(int(strict)))
    kernels = {
        NvFp4MoeBackend.MARLIN: MarlinExperts,
        NvFp4MoeBackend.EMULATION: Nvfp4QuantizationEmulationTritonExperts,
    }
    for k_cls in kernels.values():
        monkeypatch.setattr(
            k_cls, "_supports_current_device", staticmethod(lambda: True)
        )
    monkeypatch.setattr(
        nvfp4_oracle,
        "backend_to_kernel_cls",
        lambda backend: [kernels[backend]] if backend in kernels else [],
    )

    backend, experts_cls = nvfp4_oracle.select_nvfp4_moe_backend(
        make_dummy_moe_config(hidden_dim=256, intermediate_size=256),
        weight_key=kNvfp4Static,
        activation_key=kNvfp4Dynamic,
    )
    assert (backend, experts_cls) == (expected, kernels[expected])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Marlin requires CUDA")
@pytest.mark.parametrize("strict", [False, True])
def test_fp8_marlin_moe_runs_w8a8_as_w8a16_unless_strict(monkeypatch, strict):
    """Marlin only runs unquantized activations, so the FP8 oracle must resolve
    W8A8 layers to W8A16 for it rather than lose the Marlin fallback."""
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", str(int(strict)))
    config = make_dummy_moe_config(hidden_dim=256, intermediate_size=256)
    config.moe_backend = "marlin"

    if strict:
        with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
            select_fp8_moe_backend(config, kFp8Static128BlockSym, kFp8Dynamic128Sym)
    else:
        assert select_fp8_moe_backend(
            config, kFp8Static128BlockSym, kFp8Dynamic128Sym
        ) == (Fp8MoeBackend.MARLIN, MarlinExperts)
