# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 MXFP8 decode GEMM (`gemv.py`) against its PyTorch reference
and cuBLAS.

* Weight decoding, exact: rows of `x` that select single columns return
  the load-time BF16 dequantization of those weight columns exactly (as
  values; a negative-zero weight comes back as +0), over every E4M3 code
  and random scale bytes from 1 to 134.
* The decode-step shapes of DeepSeek-V4.1-Flash at TP8 at 1, 6, 24 and 48
  rows: within one BF16 ulp of the FP64 product of the dequantized weights
  plus an FP32-accumulation term (only the summation order differs), and
  within twice that bound of cuBLAS on the BF16 weights with FP32 split-K
  reductions (PyTorch's default lets cuBLAS reduce split-K partial sums in
  BF16, which on some shapes errs by 20-30 times the bound); bitwise
  repeatable.
* Column splits over warps and CTAs (the ticket path), which change only
  the FP32 summation order, and the self-resetting ticket counters.
* The row-expanded scale copy the loader keeps, and CUDA-graph replay.

The kernel uses SM80 instructions only, so any SM80-or-newer GPU runs it.
"""

import pytest
import torch

from vllm.platforms import current_platform

from . import reference

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="needs an SM80-or-newer CUDA GPU",
)

DEVICE = "cuda"

# Per-rank [N, K] at TP8: fused qkv (q_a, kv, index key), q_b without and
# with the indexer's query projection, wo_b, the shared expert's w13 and w2.
SHAPES = [
    (1792, 5120),
    (4096, 1280),
    (8192, 1280),
    (5120, 1024),
    (576, 5120),
    (5120, 288),
]


def _weight(
    n: int, k: int, low: int, high: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random E4M3 bytes (no NaN codes) and UE8M0 bytes in `[low, high]`."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    codes = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=DEVICE, generator=g)
    codes[(codes & 0x7F) == 0x7F] -= 1
    scales = torch.randint(
        low, high + 1, (n // 32, k // 32), dtype=torch.uint8, device=DEVICE, generator=g
    )
    return codes, scales


def _bound(x: torch.Tensor, weight: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """One BF16 ulp of `want` plus 2^-16 of the sum of absolute products."""
    magnitude = x.double().abs() @ weight.double().abs().t()
    ulp = torch.exp2(torch.floor(torch.log2(want.abs().clamp_min(2.0**-100))) - 7)
    return ulp + magnitude * 2.0**-16


def test_decode_bitwise():
    """Single-column rows of x return dequantized weights exactly."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import gemv

    n, k = 576, 1024
    codes, scales = _weight(n, k, 1, 134, seed=0)
    # Every byte value b at column b of each row of the first 32-row block.
    codes[:32, :256] = torch.arange(256, device=DEVICE, dtype=torch.uint8).view(1, 256)
    codes[(codes & 0x7F) == 0x7F] -= 1
    weight, block_scales = gemv.pack_weight(codes, scales)
    dequantized = reference.mxfp8_weight_dequantize(codes, scales)
    for start in range(0, k, 48):
        columns = torch.arange(start, min(start + 48, k), device=DEVICE)
        x = torch.zeros(columns.numel(), k, dtype=torch.bfloat16, device=DEVICE)
        x[torch.arange(columns.numel()), columns] = 1.0
        got = gemv.mxfp8_gemv(x, weight, block_scales)
        assert torch.equal(got.float(), dequantized[:, columns].t().float())


@pytest.mark.usefixtures("fp32_cublas_reduction")
@pytest.mark.parametrize("rows", [1, 6, 24, 48])
@pytest.mark.parametrize("n,k", SHAPES)
def test_shapes(n: int, k: int, rows: int):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import gemv

    codes, scales = _weight(n, k, 112, 122, seed=n + k)
    weight, block_scales = gemv.pack_weight(codes, scales)
    g = torch.Generator(device=DEVICE).manual_seed(rows)
    x = torch.randn(rows, k, device=DEVICE, generator=g).to(torch.bfloat16)
    got = gemv.mxfp8_gemv(x, weight, block_scales)
    want = reference.mxfp8_linear(x, codes, scales)
    dequantized = reference.mxfp8_weight_dequantize(codes, scales)
    bound = _bound(x, dequantized, want)
    assert ((got.double() - want).abs() <= bound).all()
    cublas = torch.nn.functional.linear(x, dequantized)
    assert ((got.double() - cublas.double()).abs() <= 2 * bound).all()
    again = gemv.mxfp8_gemv(x, weight, block_scales)
    assert torch.equal(got, again)


@pytest.mark.parametrize(
    "config",
    [
        (1, 8, 1),  # 8 warps over K, 20 CTAs per row block
        (2, 4, 4),  # 10 splits
        (4, 2, 10),  # 8 splits, 10 blocks in flight per warp
        (6, 1, 10),  # 16 splits, one warp per tile
        (3, 2, 8),  # odd tiles per CTA: CTAs start mid scale block
        (2, 8, 4),  # 16 warps, 5 splits
    ],
)
def test_splits(config: tuple[int, int, int]):
    """Every split of the columns gives the reference within the bound;
    the counters are back at zero after each launch."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import gemv

    n, k, rows = 576, 5120, 24
    codes, scales = _weight(n, k, 112, 122, seed=3)
    weight, block_scales = gemv.pack_weight(codes, scales)
    x = torch.randn(rows, k, device=DEVICE).to(torch.bfloat16)
    want = reference.mxfp8_linear(x, codes, scales)
    bound = _bound(x, reference.mxfp8_weight_dequantize(codes, scales), want)
    split = gemv.GemvConfig(*config)
    for _ in range(3):
        got = gemv.mxfp8_gemv(x, weight, block_scales, config=split)
        assert ((got.double() - want).abs() <= bound).all()
    torch.accelerator.synchronize()
    assert not gemv._counters(torch.device(DEVICE).index or 0).any()


def test_expanded_scales_and_graph():
    """The row-expanded scales pack to the same layout; a captured launch
    replays on new inputs."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import gemv

    n, k, rows = 1792, 5120, 6
    codes, scales = _weight(n, k, 112, 122, seed=4)
    weight, block_scales = gemv.pack_weight(codes, scales)
    weight2, block_scales2 = gemv.pack_weight(
        codes.view(torch.float8_e4m3fn), scales.repeat_interleave(32, 0)
    )
    assert torch.equal(weight, weight2) and torch.equal(block_scales, block_scales2)

    x = torch.randn(rows, k, device=DEVICE).to(torch.bfloat16)
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=DEVICE)
    gemv.mxfp8_gemv(x, weight, block_scales, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemv.mxfp8_gemv(x, weight, block_scales, out=out)
    for _ in range(2):
        x.copy_(torch.randn(rows, k, device=DEVICE))
        out.zero_()
        graph.replay()
        assert torch.equal(out, gemv.mxfp8_gemv(x, weight, block_scales))
