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

from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)


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


def test_fp8_round_up_intermediate_size_for_aiter() -> None:
    """The fp8 MoE intermediate size is rounded up to a multiple of 128 only for
    the AITER backend with gelu_tanh, the one combination that always lands on
    the CK 2-stage GEMM."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        fp8_round_up_hidden_size_and_intermediate_size,
    )

    # Should round
    assert fp8_round_up_hidden_size_and_intermediate_size(
        Fp8MoeBackend.AITER, 2048, 704, MoEActivation.GELU_TANH
    ) == (2048, 768)

    # no-op when already aligned
    assert fp8_round_up_hidden_size_and_intermediate_size(
        Fp8MoeBackend.AITER, 2048, 1024, MoEActivation.GELU_TANH
    ) == (2048, 1024)

    # silu/gelu on AITER keep their size (AITER has 1-stage kernels for them
    # that take non-128-aligned shapes).
    for activation in (MoEActivation.SILU, MoEActivation.GELU):
        assert fp8_round_up_hidden_size_and_intermediate_size(
            Fp8MoeBackend.AITER, 2048, 704, activation
        ) == (2048, 704)

    # Non-AITER backends are untouched even when non-128-aligned.
    assert fp8_round_up_hidden_size_and_intermediate_size(
        Fp8MoeBackend.TRITON, 2048, 704, MoEActivation.GELU_TANH
    ) == (2048, 704)
