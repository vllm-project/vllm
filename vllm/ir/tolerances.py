# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

ToleranceSpec = dict[torch.dtype, dict[str, float]]

# Default tolerances for comparing IR op implementations against native.
# These are intentionally conservative (permissive) to avoid false failures
# across different hardware and kernel implementations. Ops that need tighter
# or looser bounds should use override_tolerance.
DEFAULT_TOLERANCES: ToleranceSpec = {
    # 52-bit mantissa; machine epsilon ~1.1e-16
    torch.float64: {"atol": 1e-8, "rtol": 1e-8},
    # 23-bit mantissa; machine epsilon ~1.2e-7.
    # Values from PyTorch test_transformers.py reference defaults.
    torch.float32: {"atol": 1e-5, "rtol": 1.3e-6},
    # 10-bit mantissa; machine epsilon ~9.8e-4.
    # Standard tolerance used across vLLM kernel tests.
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    # 7-bit mantissa; machine epsilon ~7.8e-3.
    # Wider rtol than float16 to account for the coarser mantissa.
    torch.bfloat16: {"atol": 1e-3, "rtol": 1.6e-2},
    # 3-bit mantissa; machine epsilon ~6.25e-2.
    # Derived from vLLM fp8 kernel tests (merge_attn_states, silu_mul_fp8).
    torch.float8_e4m3fn: {"atol": 1e-1, "rtol": 1e-1},
    # 2-bit mantissa; machine epsilon ~1.25e-1.
    # Wider than e4m3fn due to the smaller mantissa.
    torch.float8_e5m2: {"atol": 2e-1, "rtol": 2e-1},
    # 1-bit mantissa; machine epsilon ~2.5e-1. Packed pair format (x2).
    # Derived from vLLM fp4 tests (test_silu_mul_nvfp4_quant: atol=3e-1).
    torch.float4_e2m1fn_x2: {"atol": 3e-1, "rtol": 3e-1},
    # Integer quantized; off-by-one from rounding is expected.
    # rtol=0 because relative error is meaningless for small integers.
    torch.int8: {"atol": 1, "rtol": 0},
}
