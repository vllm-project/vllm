# SPDX-License-Identifier: Apache-2.0
# This is a test for the AITER ops.
# It tests if the AITER ops are
# 1. correctly registered as custom ops
# 2. correctly defined the relationship between
#    implementation and fake function
# 3. can be used with torch.compile
# This file will be skipped if AITER is not installed
# and the platform is not ROCm.

import importlib.util

import pytest
import torch

# this import statement is needed to ensure the ops are registered
import vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe  # noqa: F401
from vllm.platforms import current_platform

# need to import once to ensure the ops are registered
# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER ops are only available on ROCm with aiter package installed")


def test_rocm_aiter_biased_grouped_topk_custom_op_registration():
    """Test that the custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, 'rocm_aiter_biased_grouped_topk')

    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_biased_grouped_topk)


def test_rocm_aiter_biased_grouped_topk_torch_compile_compatibility():
    """Test that the op can be used with torch.compile."""
    # Create test tensors
    token = 64
    expert = 256
    num_expert_group = 8
    topk = 8
    topk_group = 4
    renormalize = True
    scale_factor = 1.0

    gating_output = torch.randn((token, expert),
                                dtype=torch.bfloat16,
                                device="cuda")
    e_score_correction_bias = torch.randn((expert, ),
                                          dtype=torch.bfloat16,
                                          device="cuda")

    device = gating_output.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk),
                               dtype=torch.float32,
                               device=device)

    # Define a function that uses the op
    def biased_grouped_topk_fn(gating_output, e_score_correction_bias,
                               topk_weights, topk_ids):
        return torch.ops.vllm.rocm_aiter_biased_grouped_topk(
            gating_output, e_score_correction_bias, topk_weights, topk_ids,
            num_expert_group, topk_group, renormalize, scale_factor)

    # Verify the op's fake implementation
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_biased_grouped_topk,
        (gating_output, e_score_correction_bias, topk_weights, topk_ids),
        kwargs={
            "num_expert_group": num_expert_group,
            "topk_group": topk_group,
            "need_renorm": renormalize,
            "routed_scaling_factor": scale_factor
        },
        test_utils=("test_faketensor"))

    # Compile the function with appropriate settings
    compiled_fn = torch.compile(biased_grouped_topk_fn,
                                fullgraph=True,
                                backend="inductor",
                                mode="reduce-overhead",
                                dynamic=False)

    topk_weights_original = torch.empty((token, topk),
                                        dtype=torch.float32,
                                        device=device)
    topk_ids_original = torch.empty((token, topk),
                                    dtype=torch.int32,
                                    device=device)

    topk_weights_compiled = torch.empty((token, topk),
                                        dtype=torch.float32,
                                        device=device)
    topk_ids_compiled = torch.empty((token, topk),
                                    dtype=torch.int32,
                                    device=device)

    # Run both compiled (V1 graph mode) and uncompiled versions (V1 eager mode)
    biased_grouped_topk_fn(gating_output, e_score_correction_bias,
                           topk_weights_original, topk_ids_original)
    compiled_fn(gating_output, e_score_correction_bias, topk_weights_compiled,
                topk_ids_compiled)

    # Sort the results for comparison since the order might not be deterministic
    topk_ids_original, indices_original = torch.sort(topk_ids_original)
    topk_weights_original = torch.gather(topk_weights_original, 1,
                                         indices_original)

    topk_ids_compiled, indices_compiled = torch.sort(topk_ids_compiled)
    topk_weights_compiled = torch.gather(topk_weights_compiled, 1,
                                         indices_compiled)

    # Verify results match
    assert torch.allclose(topk_weights_original,
                          topk_weights_compiled,
                          rtol=1e-2,
                          atol=1e-2)
    assert torch.allclose(topk_ids_original, topk_ids_compiled)
