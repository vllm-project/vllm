# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

from vllm.platforms import current_platform

# need to import once to ensure the ops are registered
# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None
pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER ops are only available on ROCm with aiter package installed")


def test_rocm_aiter_gemm_a8w8_bpreshuffle_custom_op_registration():
    """Test that the custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, 'rocm_aiter_gemm_a8w8_bpreshuffle')

    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_gemm_a8w8_bpreshuffle)


@pytest.mark.parametrize("m", [1, 32, 64])
@pytest.mark.parametrize(
    "nk", [(1280, 8192),
           (8192, 1024)])  # n,k shapes for qkv_proj and attn output
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_rocm_aiter_gemm_a8w8_bpreshuffle_torch_compile_compatibility(
        m: int, nk: tuple[int], out_dtype: torch.dtype):
    n, k = nk
    qinput = torch.randn((m, k), dtype=out_dtype, device="cuda")
    weight = torch.randn((n, k), dtype=out_dtype, device="cuda")
    scale_a = torch.randn_like(qinput,
                               device="cuda").view(torch.float8_e4m3fnuz)
    scale_b = torch.rand_like(weight, dtype=out_dtype, device="cuda")

    # the gemm_a8w8_bpreshuffle kernel requires weight to be shuffled.
    from aiter.ops.shuffle import shuffle_weight
    weight = shuffle_weight(weight)

    def per_token_w8a8_scaled_mm(
        qinput: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_bpreshuffle(
            qinput,
            weight,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype)

    # Verify the op's fake implementation
    torch.library.opcheck(torch.ops.vllm.rocm_aiter_gemm_a8w8_bpreshuffle,
                          (qinput, weight, scale_a, scale_b, out_dtype),
                          kwargs={
                              "qinput": qinput,
                              "weight": weight,
                              "scale_a": scale_a,
                              "scale_b": scale_b,
                              "out_dtype": out_dtype,
                          },
                          test_utils=("test_faketensor"))

    output = per_token_w8a8_scaled_mm(qinput, weight, scale_a, scale_b,
                                      out_dtype)

    # Compile the function with appropriate settings
    compiled_fn = torch.compile(per_token_w8a8_scaled_mm,
                                fullgraph=True,
                                backend="inductor",
                                mode="reduce-overhead",
                                dynamic=False)
    output_compiled = compiled_fn(qinput, weight, scale_a, scale_b, out_dtype)

    assert torch.allclose(output, output_compiled)
