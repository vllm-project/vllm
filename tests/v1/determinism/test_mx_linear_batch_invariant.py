# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm native MX linear kernels must not depend on the number of rows.

Batch-invariant mode keeps both GEMMs rather than substituting a Triton kernel,
so the tile selectors -- which are keyed on M -- decide the property, and each
selector needed one thing pinned. AITER's MXFP4 selector asks for a four-way
split-K in its M <= 8 bucket and none above it, so ``gemm_with_dynamic_quant``
pins NUM_KSPLIT to 1 and leaves the rest of the tuned tile alone. The native
MXFP8 selector varies BLOCK_K across its M buckets, the one tile parameter that
reorders an output element's accumulation, so ``_mxfp8_dot_scaled_linear`` pins
BLOCK_K. These tests run with the mode on.
"""

import pytest
import torch

from vllm.utils.torch_utils import set_random_seed

from .utils import requires_mx

SEED = 0

# Half-width of the E8M0 block-scale exponent range, either side of 127.
# This is what makes the sweeps able to see anything: MXFP4 operands are four
# bits wide and the block scales are powers of two, so with a narrow spread the
# fp32 accumulation of a whole K row is *exact* and no reordering can change the
# result. test_scale_spread_exposes_reordering keeps this honest.
SCALE_SPREAD = 15

# (N, K, use_asm_gemm). gemm_with_dynamic_quant fans out to four AITER entry
# points and the row counts below reach all of them:
#   asm=False       dynamic_mxfp4_quant + gemm_afp4wfp4, which asks for a
#                   four-way split-K in its M <= 8 bucket
#   (8192, 2048)    per_1x32_f4_quant_hip + gemm_afp4wfp4_preshuffled_weight_scales
#                   for M <= 64 -- shuffle=False below M=32, True from 32 --
#                   and gemm_a4w4 above it
#   (8192, 1024)    gemm_a4w4 at every M, reaching both its asm kernels and the
#                   CK gemm_a4w4_blockscale kernel (M in 513..1024)
MXFP4_CASES = [
    (4096, 2048, False),
    (2048, 6144, False),
    (512, 1024, False),
    (8192, 2048, True),
    (8192, 1024, True),
]

# The native MXFP8 selector keys on all of M, N and K, so these shapes are its
# own -- see the note on its sweep below for what each one reaches.
MXFP8_CASES = [(4096, 2048), (2048, 6144), (1024, 768), (1536, 1024), (1024, 384)]

# The AITER MXFP4 selector switches on M alone -- the config is identical for
# every (N, K) probed -- at 1/9/33/65/129/257/513, and M <= 8 is the only band
# that splits K (NUM_KSPLIT=4). Straddle every boundary.
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

# Together with MXFP8_CASES these counts cover 11 distinct tile configurations,
# spanning every BLOCK_K the MXFP8 selector can choose (128/256/512/1024) and
# BLOCK_M of 16, 64 and 128.
MXFP8_TOKEN_COUNTS = [1, 32, 64, 65, 128, 129, 256, 257, 512, 1024, 1025, 2048]

# Row 0 always sits at offset 0, so it lands in the first tile of every
# decomposition and stays invariant even when the rest of the output does not.
# Checking it alone hides real failures.
CHECK_ROWS = [0, 1, 2, 3, 7, 15, 31]


def _probe(out: torch.Tensor) -> torch.Tensor:
    """The leading rows of an output, kept so the sweep does not hold every GEMM."""
    return out[: CHECK_ROWS[-1] + 1].clone()


def _variant_rows(probes: dict[int, torch.Tensor]) -> list[str]:
    """Rows whose contents changed with the number of rows in the launch."""
    failures = []
    for row in CHECK_ROWS:
        # A row is only comparable across launches that actually contain it.
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


def _reference_operands(fmt: str, n: int, k: int):
    """Dequantized (activation, weight) for the operands the sweeps below use."""
    if fmt == "mxfp4":
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            dequant_mxfp4,
            quant_dequant_mxfp4,
        )

        weight, weight_scale = _mxfp4_weights(n, k, use_asm_gemm=False)
        x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
        return (
            quant_dequant_mxfp4(x).float(),
            dequant_mxfp4(weight, weight_scale.T.contiguous(), torch.bfloat16).float(),
        )

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )

    weight, weight_scale = _mxfp8_weights(n, k)
    x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
    x_q, x_s = mxfp8_e4m3_quantize(x)
    return (
        dequant_mxfp8_to_bf16(x_q, x_s).float(),
        dequant_mxfp8_to_bf16(weight, weight_scale).float(),
    )


@requires_mx
@pytest.mark.parametrize(
    "fmt,n,k",
    [("mxfp4", n, k) for n, k, _ in MXFP4_CASES]
    + [("mxfp8", n, k) for n, k in MXFP8_CASES],
)
def test_scale_spread_exposes_reordering(fmt: str, n: int, k: int):
    """Positive control for the sweeps below, not a test of the operands.

    Summing one K row in the opposite order has to move the bf16 result at
    SCALE_SPREAD, or the invariance sweep over that shape is asserting nothing.
    MXFP4 needs the wide spread for this; MXFP8 carries enough mantissa that it
    holds at any spread. Parametrized over the swept shapes, since sensitivity
    is a property of K as much as of the spread.
    """
    set_random_seed(SEED)
    a, b = _reference_operands(fmt, n, k)

    forward = torch.einsum("mk,nk->mn", a, b).to(torch.bfloat16)
    reverse = torch.einsum("mk,nk->mn", a.flip(-1), b.flip(-1)).to(torch.bfloat16)
    assert not torch.equal(forward, reverse), (
        f"reversing the K order leaves every {fmt} output bit unchanged at "
        f"SCALE_SPREAD={SCALE_SPREAD} (N={n}, K={k}): the fp32 accumulation is "
        f"exact for these operands, so the batch-invariance sweep over this "
        f"shape cannot fail"
    )


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
# The kernel writes the fp32 accumulator out in the activation dtype, so fp16
# rounds the same reordering differently -- and sees more of it. With
# VLLM_BATCH_INVARIANT=0 on gfx950 the unpinned BLOCK_K moves probe rows in
# 4 of the 5 shapes under fp16 against 3 under bf16, K=768 being visible only
# in fp16. K=384 moves in neither, and not for lack of sensitivity: 384 is not
# a multiple of 256, so _select_cfg returns BLOCK_K=128 in every M bucket and
# there is no reordering to pin.
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
