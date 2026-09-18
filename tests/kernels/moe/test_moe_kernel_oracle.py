# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE oracle selection and kernel construction."""

from unittest.mock import patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
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


@pytest.mark.parametrize(
    "gemm_type,use_ep,batched,expected",
    [
        ("auto", False, False, "HummingIndexedExperts"),
        ("auto", True, False, "HummingGroupedExperts"),
        ("grouped", False, False, "HummingGroupedExperts"),
        ("indexed", True, False, "HummingIndexedExperts"),
        ("grouped_masked", True, True, "BatchedHummingGroupedExperts"),
        ("indexed", True, True, None),
    ],
)
def test_humming_oracle_matches_parallel_and_activation_format(
    monkeypatch, gemm_type, use_ep, batched, expected
):
    """Selection must honor overrides without mixing batched and standard layouts."""
    pytest.importorskip("humming")
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase,
    )
    from vllm.model_executor.layers.fused_moe.oracle import humming as oracle
    from vllm.utils.humming import HummingInputSchema, ModeloptNvfp4WeightSchema

    monkeypatch.setenv("VLLM_HUMMING_MOE_GEMM_TYPE", gemm_type)
    monkeypatch.setattr(HummingExpertsBase, "_supports_current_device", lambda: True)
    moe = make_dummy_moe_config()
    moe.moe_parallel_config.use_ep = use_ep
    if batched:
        moe.moe_parallel_config.dp_size = 2
        moe.moe_parallel_config.all2all_backend = "deepep_low_latency"
    selected = oracle.select_humming_moe_experts(
        moe, ModeloptNvfp4WeightSchema(), HummingInputSchema()
    )
    assert (selected.__name__ if selected is not None else None) == expected


@pytest.mark.parametrize("force_weight", [False, True])
@pytest.mark.parametrize("force_input", [False, True])
def test_humming_oracle_selects_for_effective_schemas(
    monkeypatch, force_weight, force_input
):
    """Selection must check the schemas used after requantization."""
    pytest.importorskip("humming")
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase,
        HummingIndexedExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle import humming as oracle
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kMxfp4Static,
        kNvfp4Static,
    )

    checkpoint_weight, requantized_weight = object(), object()
    checkpoint_input, requantized_input = object(), object()
    monkeypatch.setattr(
        oracle.humming_utils,
        "weight_schema_to_quant_key",
        lambda schema: {
            checkpoint_weight: kNvfp4Static,
            requantized_weight: kMxfp4Static,
        }[schema],
    )
    monkeypatch.setattr(
        oracle.humming_utils,
        "input_schema_to_quant_key",
        lambda schema: {
            checkpoint_input: None,
            requantized_input: kInt8DynamicTokenSym,
        }[schema],
    )

    monkeypatch.setenv("VLLM_HUMMING_MOE_GEMM_TYPE", "indexed")
    monkeypatch.setattr(HummingExpertsBase, "_supports_current_device", lambda: True)
    expected_weight = kMxfp4Static if force_weight else kNvfp4Static
    expected_input = kInt8DynamicTokenSym if force_input else None
    monkeypatch.setattr(
        HummingExpertsBase,
        "_supports_quant_scheme",
        lambda weight, activation: (weight, activation)
        == (expected_weight, expected_input),
    )
    selected = oracle.select_humming_moe_experts(
        make_dummy_moe_config(),
        checkpoint_weight,
        checkpoint_input,
        force_weight_schema=requantized_weight if force_weight else None,
        force_input_schema=requantized_input if force_input else None,
    )
    assert selected is HummingIndexedExperts


def test_humming_oracle_without_optional_dependency(monkeypatch):
    from vllm.model_executor.layers.fused_moe.oracle import humming as oracle

    monkeypatch.setattr(oracle, "has_humming", lambda: False)
    assert oracle.select_humming_moe_experts(None, None, None) is None
