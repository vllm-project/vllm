# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm native MX linear kernels must not depend on the number of rows."""

import pytest
import torch
from utils import requires_mx

from vllm.utils.torch_utils import set_random_seed

SEED = 0

# E8M0 exponent half-width. With 4-bit operands and a narrow spread the fp32
# partial sums are exact, so a reordering of them would be invisible.
SCALE_SPREAD = 15

# (N, K, use_asm_gemm)
MXFP4_CASES = [
    (4096, 2048, False),
    (2048, 6144, False),
    (4608, 7168, False),
    (8192, 2048, True),
    (8192, 1024, True),
    (8192, 14336, True),
]

MXFP8_CASES = [(4096, 2048), (2048, 6144), (1024, 768), (1536, 1024)]

# The untuned AITER MXFP4 configs switch at M = 9/33/65/129/257/513.
MXFP4_TOKEN_COUNTS = [
    1,
    2,
    8,
    9,
    16,
    32,
    33,
    64,
    65,
    128,
    129,
    256,
    257,
    512,
    513,
    1024,
    2048,
]

MXFP8_TOKEN_COUNTS = [1, 32, 64, 65, 128, 129, 256, 257, 512, 1024, 1025, 2048]

# Row 0 lands in the first tile of every decomposition, so it can stay
# invariant when the rest of the output does not.
CHECK_ROWS = [0, 1, 2, 3, 7, 15, 31]


def _probe(out: torch.Tensor) -> torch.Tensor:
    """The leading rows of an output, kept so the sweep does not hold every GEMM."""
    return out[: CHECK_ROWS[-1] + 1].clone()


def _variant_rows(probes: dict[int, torch.Tensor]) -> list[str]:
    """Rows whose contents changed with the number of rows in the launch."""
    failures = []
    for row in CHECK_ROWS:
        counts = [n for n in sorted(probes) if n > row]
        if not counts:
            continue
        reference = probes[counts[0]][row]
        variant = [n for n in counts if not torch.equal(probes[n][row], reference)]
        if variant:
            failures.append(f"row {row} changed at row counts {variant}")
    return failures


def _mxfp4_weights(n: int, k: int, use_asm_gemm: bool):
    """Weights laid out as AiterMxfp4LinearKernel.process_weights_after_loading."""
    weight = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_scale = torch.randint(
        127 - SCALE_SPREAD,
        128 + SCALE_SPREAD,
        (n, k // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    if not use_asm_gemm:
        return weight, weight_scale.T.contiguous()

    from aiter.ops.shuffle import shuffle_weight

    sm, sn = weight_scale.shape
    weight_scale = (
        weight_scale.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        .permute(0, 3, 5, 2, 4, 1, 6)
        .contiguous()
        .view(sm, sn)
    )
    return shuffle_weight(weight, layout=(16, 16)), weight_scale


def _mxfp8_weights(n: int, k: int):
    """E4M3 weights with E8M0 block scales, as the MXFP8 layer stores them."""
    weight = (torch.randn(n, k, device="cuda") / 8).to(torch.float8_e4m3fn)
    weight_scale = torch.randint(
        127 - SCALE_SPREAD,
        128 + SCALE_SPREAD,
        (n, k // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    return weight, weight_scale


@requires_mx
@pytest.mark.parametrize("n,k,use_asm_gemm", MXFP4_CASES)
def test_mxfp4_linear_is_batch_invariant(n: int, k: int, use_asm_gemm: bool):
    pytest.importorskip("aiter")

    # Importing the module registers torch.ops.vllm.gemm_with_dynamic_quant.
    import vllm.model_executor.kernels.linear.mxfp4.aiter  # noqa: F401

    set_random_seed(SEED)
    weight, weight_scale = _mxfp4_weights(n, k, use_asm_gemm)
    x = torch.randn(max(MXFP4_TOKEN_COUNTS), k, device="cuda", dtype=torch.bfloat16)

    probes = {
        num_tokens: _probe(
            torch.ops.vllm.gemm_with_dynamic_quant(
                x[:num_tokens], weight, weight_scale, use_asm_gemm, torch.bfloat16
            )
        )
        for num_tokens in MXFP4_TOKEN_COUNTS
    }

    failures = _variant_rows(probes)
    assert not failures, (
        f"MXFP4 linear depends on the row count "
        f"(N={n}, K={k}, asm={use_asm_gemm}):\n  " + "\n  ".join(failures)
    )


@requires_mx
@pytest.mark.parametrize("n,k", MXFP8_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_linear_is_batch_invariant(n: int, k: int, dtype: torch.dtype):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_linear,
    )

    set_random_seed(SEED)
    weight, weight_scale = _mxfp8_weights(n, k)
    x = torch.randn(max(MXFP8_TOKEN_COUNTS), k, device="cuda", dtype=dtype)

    probes = {
        num_tokens: _probe(
            _mxfp8_dot_scaled_linear(x[:num_tokens], weight, weight_scale)
        )
        for num_tokens in MXFP8_TOKEN_COUNTS
    }

    failures = _variant_rows(probes)
    assert not failures, (
        f"MXFP8 linear depends on the row count (N={n}, K={k}, {dtype}):\n  "
        + "\n  ".join(failures)
    )
