# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 mHC boundary kernels against their PyTorch references and the
stock TileLang kernels.

Error bounds:

* the post-mixed streams and the layer input: one BF16 ulp (relative 2^-7)
  plus an absolute FP32-accumulation term, which matters only where the
  post-mix or the collapse cancels;
* the stream mean: bitwise the BF16 mean of the kernel's own streams (the
  FP32 sum of four BF16 values is exact at these magnitudes); against
  the reference, one BF16 ulp of the mean plus the mean of the streams'
  differences (a stream one ulp off near a rounding tie shifts the mean by a
  quarter of that ulp, which cancellation in the mean does not scale down);
* the coefficients: 4e-6 absolute against the reference evaluated on the
  kernel's own streams (FP32 sums of up to 20480 products, in another
  order) and, for the four streams on SM80, with the projection rounded to
  the two BF16 terms the tensor-core kernel multiplies by;
* against the stock kernels: the streams and layer input as above, and the
  coefficients within twice the effect of rounding the projection input to
  BF16 -- the stock fused kernel of up to 32 rows projects the unrounded
  FP32 streams, these kernels (like the stock path above 32 rows and the
  PyTorch reference) the stored BF16 streams -- plus the effect of the
  two-term projection.

Also covered: the ticket counters across repeated launches and the
side-stream boundary inside a CUDA graph.
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
HIDDEN = 5120
EPS = 1e-20
SINKHORN = (1e-6, 1e-6, 2.0, 20)


def _inputs(rows: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """A boundary's operands at DeepSeek-V4.1 scale."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=DEVICE, generator=g)

    def rand(*shape):
        return torch.rand(*shape, device=DEVICE, generator=g)

    return {
        "x": randn(rows, HIDDEN).bfloat16(),
        "residual": (randn(rows, 4, HIDDEN) * 2).bfloat16(),
        "post": rand(rows, 4) * 2,
        "comb": torch.softmax(randn(rows, 4, 4), -1),
        "pre": rand(rows, 4) + 1e-6,
        "weight": (rand(HIDDEN) + 0.5).bfloat16(),
        "fn": randn(24, 4 * HIDDEN) * 0.02,
        "scale": rand(3) + 0.5,
        "base": randn(24) * 0.5,
    }


def _projection_terms(fn: torch.Tensor) -> torch.Tensor:
    """The projection as the tensor-core kernel multiplies by it: each FP32
    weight as the sum of `hi = bf16(w)` and `lo = bf16(w - hi)`."""
    hi = fn.bfloat16().float()
    return hi + (fn - hi).bfloat16().float()


def _assert_bf16_close(got: torch.Tensor, want: torch.Tensor, atol: float):
    torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=atol)


@pytest.mark.parametrize("rows", [1, 6, 48, 192])
@pytest.mark.parametrize("mode", ["post", "post_aux", "collapse", "first"])
def test_post_collapse(rows, mode):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import mhc

    t = _inputs(rows)
    if mode.startswith("post"):
        args = (t["x"], t["residual"], t["post"], t["comb"], t["pre"])
    else:
        args = (
            None,
            t["residual"],
            None,
            None,
            t["pre"] if mode == "collapse" else None,
        )
    aux = mode == "post_aux"
    got = mhc.mhc_post_collapse(*args, t["weight"], EPS, aux)
    want = reference.mhc_post_collapse(*args, t["weight"], EPS, aux)
    _assert_bf16_close(got[0], want[0], 1e-4)
    _assert_bf16_close(got[1], want[1], 2e-3)
    assert got[2].shape == want[2].shape
    if aux:
        assert torch.equal(got[2], got[0].double().mean(1).bfloat16())
        mean, want_mean = got[2].double(), want[2].double()
        stream_error = (got[0].double() - want[0].double()).abs().mean(1)
        bound = stream_error + torch.maximum(mean.abs(), want_mean.abs()) * 2**-7
        assert ((mean - want_mean).abs() <= bound + 1e-6).all()


@pytest.mark.parametrize("rows", [1, 6, 17, 48, 192])
@pytest.mark.parametrize("mode", ["post", "streams", "broadcast"])
def test_coefficients(rows, mode):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import mhc

    t = _inputs(rows)
    fn = t["fn"]
    if mode == "post":
        args = (t["x"], t["residual"], t["post"], t["comb"])
        streams = mhc.mhc_post_collapse(*args, t["pre"], t["weight"], EPS)[0]
    elif mode == "streams":
        args = (None, t["residual"], None, None)
        streams = t["residual"]
    else:
        args = (None, t["x"], None, None)
        streams = t["x"]
        fn = fn[:, :HIDDEN].contiguous()
    tail = (fn, t["scale"], t["base"], EPS, *SINKHORN)
    got = mhc.mhc_coefficients(*args, *tail)
    # SM80's tensor-core kernel (four streams) multiplies by the two terms.
    mma = mode != "broadcast" and mhc._uses_mma(mhc.HC, HIDDEN)
    terms = _projection_terms(fn) if mma else fn
    want = reference.mhc_coefficients(None, streams, None, None, terms, *tail[1:])
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=4e-6)
    # The last CTA resets its ticket counter, so launches repeat bitwise.
    for _ in range(3):
        again = mhc.mhc_coefficients(*args, *tail)
        for g, a in zip(got, again):
            assert torch.equal(g, a)


@pytest.mark.parametrize("rows", [6, 48])
def test_boundary_in_cuda_graph(rows):
    """The boundary with its coefficients on a side stream, eagerly and
    captured and replayed, equals the single-stream boundary."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import mhc

    t = _inputs(rows)
    args = (
        t["x"],
        t["residual"],
        t["post"].unsqueeze(-1),
        t["comb"],
        t["fn"],
        t["scale"],
        t["base"],
        EPS,
        *SINKHORN,
        t["pre"],
        t["weight"],
        EPS,
        True,
    )
    want = mhc.mhc_post_pre(*args)
    side = torch.cuda.Stream()
    capture = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    # Launches of the coefficients share the ticket counters, so the side
    # stream starts after the boundary above, as the model's boundaries do.
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        warm = mhc.mhc_post_pre(*args, stream=side)
        capture.wait_stream(side)
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph, stream=capture):
            got = mhc.mhc_post_pre(*args, stream=side)
            capture.wait_stream(side)
    for g, w in zip(warm, want):
        assert torch.equal(g, w)
    for _ in range(2):
        graph.replay()
        torch.accelerator.synchronize()
        for g, w in zip(got, want):
            assert torch.equal(g, w)


@pytest.mark.parametrize("rows", [6, 48])
def test_matches_stock_kernels(rows):
    tilelang = pytest.importorskip("vllm.model_executor.kernels.mhc.tilelang")
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import mhc

    t = _inputs(rows)
    args = (
        t["x"],
        t["residual"],
        t["post"].unsqueeze(-1),
        t["comb"],
        t["fn"],
        t["scale"],
        t["base"],
        EPS,
        *SINKHORN,
    )
    stock = tilelang.mhc_fused_post_pre_delayed_tilelang(
        *args, pre_mix=t["pre"], norm_weight=t["weight"], norm_eps=EPS
    )
    ours = mhc.mhc_post_pre(*args, t["pre"], t["weight"], EPS)
    _assert_bf16_close(ours[0], stock[0], 1e-4)
    _assert_bf16_close(ours[3], stock[3], 2e-3)

    # The effect of rounding the projection input to BF16.
    tail = (t["fn"], t["scale"], t["base"], EPS, *SINKHORN)
    rounded = reference.mhc_post(t["x"], t["residual"], t["post"], t["comb"])
    exact = torch.einsum("nkj,nkh->njh", t["comb"].double(), t["residual"].double())
    exact = exact + t["post"].double()[:, :, None] * t["x"].double()[:, None, :]
    from_rounded = reference.mhc_coefficients(None, rounded, None, None, *tail)
    from_exact = reference.mhc_coefficients(None, exact, None, None, *tail)
    from_terms = reference.mhc_coefficients(
        None, rounded, None, None, _projection_terms(t["fn"]), *tail[1:]
    )
    # Post-, comb- and pre-mix, in the reference's order.
    for i, index in enumerate((1, 2, 4)):
        bound = (
            2 * (from_rounded[i] - from_exact[i]).abs().max().item()
            + (from_terms[i] - from_rounded[i]).abs().max().item()
            + 1e-5
        )
        error = (ours[index].view(rows, -1) - stock[index].view(rows, -1)).abs()
        assert error.max().item() <= bound
