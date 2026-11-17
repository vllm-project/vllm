# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 split-k mHC projection of prefill boundaries against an FP64
reference, slice by slice, and the boundaries built on it against the
reference coefficients and the stock TileLang path.

Error bounds:

* a partial mix: 2^-14 of its slice's sum of `|x| |w|` (the BF16
  `hi + lo` split keeps 16 bits of each weight; the FP32 sums of up to
  1280 products add a few ulp);
* a partial sum of squares: 1e-5 relative (exact squares, FP32 sums);
* the coefficients: 2e-5 absolute against the reference evaluated on the
  same streams; the layer input and the stream mean: one BF16 ulp of the
  stock path's, whose epilogue they share.
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


def _assert_bf16_close(got: torch.Tensor, want: torch.Tensor, atol: float):
    torch.testing.assert_close(got.float(), want.float(), rtol=2**-7, atol=atol)


@pytest.mark.parametrize("rows", [257, 1000, 4096])
@pytest.mark.parametrize("width", [HIDDEN, 4 * HIDDEN])
def test_partial_sums(rows, width):
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.mhc_projection import (  # noqa: E501
        SPLITS,
        mhc_projection,
    )

    g = torch.Generator(device=DEVICE).manual_seed(rows)
    x = (torch.randn(rows, width, device=DEVICE, generator=g) * 2).bfloat16()
    fn = torch.randn(24, width, device=DEVICE, generator=g) * 0.02
    mixes, sqrsum = mhc_projection(x, fn)
    xs = x.double().view(rows, SPLITS, -1)
    ws = fn.double().view(24, SPLITS, -1)
    want = torch.einsum("nsk,jsk->snj", xs, ws)
    scale = torch.einsum("nsk,jsk->snj", xs.abs(), ws.abs())
    assert ((mixes.double() - want).abs() <= scale * 2**-14).all()
    torch.testing.assert_close(
        sqrsum.double(), xs.square().sum(-1).T, rtol=1e-5, atol=0
    )


def _boundary(rows: int) -> dict[str, torch.Tensor]:
    """A boundary's operands at DeepSeek-V4.1 scale."""
    g = torch.Generator(device=DEVICE).manual_seed(rows)

    def randn(*shape):
        return torch.randn(*shape, device=DEVICE, generator=g)

    def rand(*shape):
        return torch.rand(*shape, device=DEVICE, generator=g)

    return {
        "x": randn(rows, HIDDEN).bfloat16(),
        "residual": (randn(rows, 4, HIDDEN) * 2).bfloat16(),
        "post": rand(rows, 4, 1) * 2,
        "comb": torch.softmax(randn(rows, 4, 4), -1),
        "pre": rand(rows, 4) + 1e-6,
        "weight": (rand(HIDDEN) + 0.5).bfloat16(),
        "fn": randn(24, 4 * HIDDEN) * 0.02,
        "scale": rand(3) + 0.5,
        "base": randn(24) * 0.5,
    }


@pytest.mark.parametrize("rows", [300, 2048])
def test_post_pre_boundary(rows):
    tilelang = pytest.importorskip("vllm.model_executor.kernels.mhc.tilelang")
    from vllm.models.deepseek_v41.nvidia.ops import mhc as ops

    t = _boundary(rows)
    tail = (t["fn"], t["scale"], t["base"], EPS, *SINKHORN)
    stock = tilelang.mhc_fused_post_pre_delayed_tilelang(
        t["x"],
        t["residual"],
        t["post"],
        t["comb"],
        *tail,
        pre_mix=t["pre"],
        norm_weight=t["weight"],
        norm_eps=EPS,
        capture_aux=True,
    )
    streams = tilelang.mhc_post_tilelang(t["x"], t["residual"], t["post"], t["comb"])
    assert torch.equal(streams, stock[0])
    post, comb, layer_input, pre, aux = ops._mhc_pre_split(
        streams, *tail, t["pre"], None, t["weight"], EPS, True
    )
    want = reference.mhc_coefficients(None, streams, None, None, *tail)
    for got, w in zip((post.view(rows, 4), comb, pre), want):
        torch.testing.assert_close(got, w.view(got.shape), rtol=0, atol=2e-5)
    _assert_bf16_close(layer_input, stock[3], 2e-3)
    _assert_bf16_close(aux, stock[5], 1e-4)


@pytest.mark.parametrize("rows", [300])
def test_first_layer_boundary(rows):
    tilelang = pytest.importorskip("vllm.model_executor.kernels.mhc.tilelang")
    from vllm.models.deepseek_v41.nvidia.ops import mhc as ops

    t = _boundary(rows)
    fn = t["fn"][:, :HIDDEN].contiguous()
    streams = t["x"].unsqueeze(1).expand(-1, 4, -1).contiguous()
    tail = (fn, t["scale"], t["base"], EPS, *SINKHORN)
    stock = tilelang.mhc_pre_delayed_tilelang(
        streams, *tail, x=t["x"], norm_weight=t["weight"], norm_eps=EPS
    )
    post, comb, layer_input, pre, aux = ops._mhc_pre_split(
        streams, *tail, None, t["x"], t["weight"], EPS
    )
    assert aux.shape == (0, HIDDEN)
    want = reference.mhc_coefficients(None, t["x"], None, None, *tail)
    for got, w in zip((post.view(rows, 4), comb, pre), want):
        torch.testing.assert_close(got, w.view(got.shape), rtol=0, atol=2e-5)
    _assert_bf16_close(layer_input, stock[2], 2e-3)
