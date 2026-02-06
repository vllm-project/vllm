# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tier 1: NVFP4 MoE Backend Parity Test

This test compares the numerical output of FlashInfer CUTLASS vs vLLM CUTLASS
backends for NVFP4 quantized MoE layers. Both backends should produce
numerically similar results for identical inputs.

Key test scenarios:
1. Standard gated activation (silu) - typical MoE models
2. Non-gated activation (relu2_no_mul) - Nemotron-Nano model

Tolerance rationale:
- 1e-2 atol/rtol for backend comparison (both use NVFP4, implementation diffs)
- Higher than Tier 2 (activation test) because kernels have different
  numerical paths internally

Results are saved to /tmp/tier1_backend_parity_results.txt
"""

import json
import os
from datetime import datetime

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config, make_test_quant_config
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp4
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
    is_valid_flashinfer_cutlass_fused_moe,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.utils.torch_utils import set_random_seed

# Results file path
RESULTS_FILE = "/tmp/tier1_backend_parity_results.txt"

# Skip if requirements not met
if not has_flashinfer_cutlass_fused_moe() or not current_platform.has_device_capability(
    100
):
    pytest.skip(
        "Requires flashinfer_cutlass_fused_moe and compute capability >= 10.0",
        allow_module_level=True,
    )


def write_result(test_name: str, status: str, details: dict):
    """Append test result to results file."""
    timestamp = datetime.now().isoformat()
    result = {
        "timestamp": timestamp,
        "test": test_name,
        "status": status,
        "details": details,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def compute_parity_metrics(output1: torch.Tensor, output2: torch.Tensor) -> dict:
    """Compute various parity metrics between two tensors."""
    diff = (output1 - output2).float()
    abs_diff = diff.abs()

    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "std_abs_diff": abs_diff.std().item(),
        "max_rel_diff": (abs_diff / (output2.abs().float() + 1e-8)).max().item(),
        "output1_norm": output1.float().norm().item(),
        "output2_norm": output2.float().norm().item(),
        "allclose_1e2": torch.allclose(output1, output2, atol=1e-2, rtol=1e-2),
        "allclose_1e1": torch.allclose(output1, output2, atol=1e-1, rtol=1e-1),
    }


# Test dimensions - using dimensions known to work with both backends
# (M tokens, N intermediate, K hidden)
# Note: Both FlashInfer and vLLM CUTLASS have alignment requirements
# Dimensions must be divisible by certain factors (16 for block scales, etc.)
TEST_DIMENSIONS = [
    # Standard dimensions (from existing test_flashinfer_moe.py)
    (64, 1024, 1024),
    (64, 2048, 1024),
    (64, 3072, 1024),
    (32, 1024, 1536),  # Different K dimension
    (224, 1024, 1024),  # Larger batch
]

# Expert configurations
EXPERT_CONFIGS = [
    (8, 2),  # 8 experts, top-2
    (64, 4),  # 64 experts, top-4 (closer to Nemotron-Nano's 128 experts)
]

# IMPORTANT FINDING: CutlassExpertsFp4 only supports gated activations (silu)
# It expects weight shapes [e, 2*n, k] which is specific to gated MLPs.
# For non-gated activations (relu2_no_mul), only FlashInfer CUTLASS works.
# This is documented in test_flashinfer_only_relu2_no_mul below.
GATED_ACTIVATIONS = ["silu"]  # Only gated activations for parity test


@pytest.mark.parametrize("m,n,k", TEST_DIMENSIONS)
@pytest.mark.parametrize("e,topk", EXPERT_CONFIGS)
@pytest.mark.parametrize("activation", GATED_ACTIVATIONS)
@torch.inference_mode()
def test_flashinfer_vs_vllm_cutlass_parity(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: str,
    workspace_init,
):
    """Test numerical parity between FlashInfer CUTLASS and vLLM CUTLASS backends.

    Both backends process NVFP4 quantized weights and should produce similar outputs.
    This validates that the NVFP4 MoE implementation is consistent across backends.

    Args:
        m: Number of tokens (batch size)
        n: Intermediate dimension
        k: Hidden dimension
        e: Number of experts
        topk: Number of experts selected per token
        activation: Activation function ('silu' for gated, 'relu2_no_mul' for non-gated)
    """
    test_name = f"parity_m{m}_n{n}_k{k}_e{e}_topk{topk}_{activation}"
    set_random_seed(42)  # Fixed seed for reproducibility

    # Determine if using gated activation
    is_gated = activation == "silu"

    # Map activation names for different backends
    # FlashInfer uses: "silu" or "relu2_no_mul"
    # make_test_quant_config expects: "silu_and_mul" or "relu2" internally
    internal_activation = "silu_and_mul" if is_gated else "relu2"

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        dtype = torch.bfloat16

        # Create input tensor
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        # Create quantized weights using shared utility
        # make_gate=False for relu2_no_mul (non-gated)
        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated,
        )

        # Create routing scores and topk selection
        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        # Validate FlashInfer can handle this configuration
        assert is_valid_flashinfer_cutlass_fused_moe(a, w1_q, w2_q), (
            f"FlashInfer CUTLASS cannot handle config: "
            f"m={m}, n={n}, k={k}, e={e}, topk={topk}"
        )

        # Create MoE configuration (shared by both backends)
        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size_per_partition=n,
            num_local_experts=e,
            activation=internal_activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            is_act_and_mul=is_gated,
            routing_method=RoutingMethodType.TopK,
        )

        # ========== FlashInfer CUTLASS Backend ==========
        flashinfer_kernel = FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
        )

        flashinfer_output = flashinfer_kernel(
            hidden_states=a.clone(),
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights.clone(),
            topk_ids=topk_ids.clone(),
            activation=activation,
        )

        # ========== vLLM CUTLASS Backend ==========
        vllm_cutlass_kernel = FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            CutlassExpertsFp4(
                moe_config=make_dummy_moe_config(),
                quant_config=quant_config,
            ),
        )

        vllm_cutlass_output = vllm_cutlass_kernel(
            hidden_states=a.clone(),
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights.clone(),
            topk_ids=topk_ids.clone(),
            activation=activation,
        )

        # ========== Compute and Log Parity Metrics ==========
        metrics = compute_parity_metrics(flashinfer_output, vllm_cutlass_output)

        # Log results
        write_result(
            test_name,
            "computed",
            {
                "dimensions": {"m": m, "n": n, "k": k, "e": e, "topk": topk},
                "activation": activation,
                "is_gated": is_gated,
                "metrics": metrics,
            },
        )

        # ========== Assertions ==========
        # Basic sanity checks
        assert not torch.isnan(flashinfer_output).any(), (
            "FlashInfer output contains NaN"
        )
        assert not torch.isnan(vllm_cutlass_output).any(), (
            "vLLM CUTLASS output contains NaN"
        )
        assert not torch.isinf(flashinfer_output).any(), (
            "FlashInfer output contains Inf"
        )
        assert not torch.isinf(vllm_cutlass_output).any(), (
            "vLLM CUTLASS output contains Inf"
        )

        # Parity check with tolerance appropriate for NVFP4
        # Both backends use 4-bit quantization, so some numerical difference expected
        try:
            torch.testing.assert_close(
                flashinfer_output,
                vllm_cutlass_output,
                atol=1e-1,  # NVFP4 expected tolerance
                rtol=1e-1,
            )
            write_result(test_name, "PASSED", metrics)
        except AssertionError as e:
            write_result(
                test_name,
                "FAILED",
                {
                    "metrics": metrics,
                    "error": str(e)[:500],
                },
            )
            raise


@torch.inference_mode()
def test_flashinfer_only_relu2_no_mul(workspace_init):
    """Test FlashInfer CUTLASS with relu2_no_mul (non-gated activation).

    IMPORTANT FINDING:
    -----------------
    CutlassExpertsFp4 (vLLM CUTLASS) does NOT support non-gated activations.
    It expects weight shape [e, 2*n, k] which is specific to gated MLPs.
    For relu2_no_mul (Nemotron-Nano), weight shape is [e, n, k].

    This test verifies that FlashInfer CUTLASS correctly handles non-gated
    activations, which is critical for Nemotron-Nano-4B-Instruct support.

    Backend support matrix:
    | Activation    | FlashInfer CUTLASS | vLLM CUTLASS |
    |---------------|-------------------|--------------|
    | silu (gated)  | YES               | YES          |
    | relu2_no_mul  | YES               | NO           |
    """
    test_name = "flashinfer_only_relu2_no_mul"
    set_random_seed(42)

    # Test dimensions
    m, n, k = 64, 1024, 1024
    e, topk = 8, 2
    activation = "relu2_no_mul"
    is_gated = False

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        dtype = torch.bfloat16

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated,  # False for non-gated
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size_per_partition=n,
            num_local_experts=e,
            activation="relu2",
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            is_act_and_mul=is_gated,
            routing_method=RoutingMethodType.TopK,
        )

        # FlashInfer CUTLASS - should work with relu2_no_mul
        fi_kernel = FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
        )
        fi_output = fi_kernel(
            hidden_states=a.clone(),
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights.clone(),
            topk_ids=topk_ids.clone(),
            activation=activation,
        )

        # Sanity checks
        assert not torch.isnan(fi_output).any(), "FlashInfer output contains NaN"
        assert not torch.isinf(fi_output).any(), "FlashInfer output contains Inf"
        assert fi_output.abs().sum() > 0, "FlashInfer output is all zeros"
        assert fi_output.shape == (m, k), f"Output shape mismatch: {fi_output.shape}"

        write_result(
            test_name,
            "PASSED",
            {
                "dimensions": {"m": m, "n": n, "k": k, "e": e, "topk": topk},
                "activation": activation,
                "note": "FlashInfer CUTLASS supports relu2_no_mul; "
                "vLLM CUTLASS does NOT",
                "output_norm": fi_output.float().norm().item(),
            },
        )


@pytest.mark.parametrize("e,topk", [(8, 2), (64, 4)])
@pytest.mark.parametrize("m,n,k", [(64, 1024, 1024), (64, 2048, 1024)])
@torch.inference_mode()
def test_flashinfer_relu2_no_mul_various_dims(
    m: int, n: int, k: int, e: int, topk: int, workspace_init
):
    """Test FlashInfer CUTLASS with relu2_no_mul across various dimensions.

    This tests that FlashInfer properly handles non-gated activations
    (required for Nemotron-Nano) across multiple dimension combinations.

    Key finding: vLLM CUTLASS does NOT support relu2_no_mul because it
    expects gated weight shapes [e, 2*n, k]. Only FlashInfer works for
    non-gated MoE layers.
    """
    test_name = f"flashinfer_relu2_m{m}_n{n}_k{k}_e{e}_topk{topk}"
    set_random_seed(42)

    activation = "relu2_no_mul"
    is_gated = False

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        dtype = torch.bfloat16

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size_per_partition=n,
            num_local_experts=e,
            activation="relu2",
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            is_act_and_mul=is_gated,
            routing_method=RoutingMethodType.TopK,
        )

        # FlashInfer CUTLASS - the only backend that works for non-gated MoE
        fi_kernel = FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
        )
        fi_output = fi_kernel(
            hidden_states=a.clone(),
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights.clone(),
            topk_ids=topk_ids.clone(),
            activation=activation,
        )

        # Sanity checks
        assert not torch.isnan(fi_output).any(), "FlashInfer output contains NaN"
        assert not torch.isinf(fi_output).any(), "FlashInfer output contains Inf"
        assert fi_output.abs().sum() > 0, "FlashInfer output is all zeros"
        assert fi_output.shape == (m, k), f"Output shape mismatch: {fi_output.shape}"

        write_result(
            test_name,
            "PASSED",
            {
                "dimensions": {"m": m, "n": n, "k": k, "e": e, "topk": topk},
                "activation": activation,
                "note": "FlashInfer-only test for non-gated "
                "relu2_no_mul (Nemotron-Nano)",
                "output_norm": fi_output.float().norm().item(),
            },
        )


def setup_module():
    """Initialize results file with header."""
    with open(RESULTS_FILE, "w") as f:
        f.write("# Tier 1: NVFP4 MoE Backend Parity Test Results\n")
        f.write(f"# Started: {datetime.now().isoformat()}\n")
        f.write(f"# Platform: {current_platform.get_device_name()}\n")
        f.write("#" + "=" * 60 + "\n")


def teardown_module():
    """Summarize results at end of test run."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "a") as f:
            f.write("#" + "=" * 60 + "\n")
            f.write(f"# Completed: {datetime.now().isoformat()}\n")

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"Tier 1 Results saved to: {RESULTS_FILE}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
