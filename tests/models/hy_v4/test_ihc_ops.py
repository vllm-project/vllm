# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity of the Triton iHC ops (pre / post / head) against the eager HY V4
layers, plus custom-op and warmup checks."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.hy_v4.nvidia import hc as hc_mod
from vllm.models.hy_v4.nvidia import triton_ihc
from vllm.models.hy_v4.nvidia.hc import (
    HYV4HCHeadLayer,
    HYV4HCPostLayer,
    HYV4HCPreLayer,
)
from vllm.models.hy_v4.nvidia.triton_ihc import (
    triton_ihc_head,
    triton_ihc_post,
    triton_ihc_pre,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="iHC Triton kernels require CUDA and Triton",
)


@pytest.fixture(autouse=True)
def _vllm_env(dist_init, default_vllm_config):
    """ReplicatedLinear needs a TP group and a current VllmConfig."""
    yield


HC = 4
NORM_EPS = 1e-5
HC_EPS = 1e-6
MAGNITUDE = 2.0
DEVICE = "cuda"

TOKENS = [1, 2, 7, 16, 65, 300, 1024]
HIDDEN = [512, 4096, 6144]
DTYPES = [torch.bfloat16, torch.float16]


def _config(hidden_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        hc_mult=HC,
        hc_magnitude=MAGNITUDE,
        hc_eps=HC_EPS,
        rms_norm_eps=NORM_EPS,
        enable_ihc=True,
    )


def _randomize_gates(scale: torch.Tensor, base: torch.Tensor) -> None:
    # Init values (0.01 scale) leave every gate near its bias; perturb so the
    # normalization + projection path actually affects the output.
    with torch.no_grad():
        scale.fill_(0.5)
        base.add_(torch.randn_like(base) * 0.3)


def _make_pre(hidden: int) -> HYV4HCPreLayer:
    layer = HYV4HCPreLayer(
        _config(hidden), hidden, HC, MAGNITUDE, 6e-3, 0.0, HC_EPS, NORM_EPS
    ).to(DEVICE)
    layer.hpc_op = None
    with torch.no_grad():
        layer.hc_fn.weight.normal_(std=6e-3)
    _randomize_gates(layer.hc_scale, layer.hc_base)
    return layer


def _make_head(hidden: int) -> HYV4HCHeadLayer:
    layer = HYV4HCHeadLayer(_config(hidden), hidden, HC, HC_EPS).to(DEVICE)
    layer.hpc_op = None
    with torch.no_grad():
        layer.hc_head_fn.weight.normal_(std=6e-3)
    _randomize_gates(layer.hc_head_scale, layer.hc_head_base)
    return layer


def _eager(monkeypatch: pytest.MonkeyPatch, layer, *args):
    """Run the layer on its eager path (Triton dispatch disabled)."""
    with monkeypatch.context() as m:
        m.setattr(hc_mod, "triton_ihc_supported", lambda x: False)
        return layer(*args)


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("dtype", DTYPES)
def test_ihc_pre_matches_eager(
    tokens: int, hidden: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    layer = _make_pre(hidden)
    x = torch.randn(tokens, HC, hidden, dtype=dtype, device=DEVICE)

    ref_y, ref_post = _eager(monkeypatch, layer, x)
    y, post = triton_ihc_pre(
        x,
        layer.hc_fn.weight,
        layer.hc_scale,
        layer.hc_base,
        MAGNITUDE,
        HC_EPS,
        NORM_EPS,
    )
    torch.testing.assert_close(post, ref_post, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y, ref_y, atol=2e-2, rtol=2e-2)
    # The layer itself dispatches to the Triton op on this platform.
    y2, post2 = layer(x)
    torch.testing.assert_close(y2, y)
    torch.testing.assert_close(post2, post)


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("dtype", DTYPES)
def test_ihc_head_matches_eager(
    tokens: int, hidden: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    layer = _make_head(hidden)
    x = torch.randn(tokens, HC, hidden, dtype=dtype, device=DEVICE)

    ref = _eager(monkeypatch, layer, x)
    y = triton_ihc_head(
        x,
        layer.hc_head_fn.weight,
        layer.hc_head_scale,
        layer.hc_head_base,
        HC_EPS,
        NORM_EPS,
    )
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(layer(x), y)


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("dtype", DTYPES)
def test_ihc_post_matches_eager(
    tokens: int, hidden: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    layer = HYV4HCPostLayer(_config(hidden))
    layer.hpc_op = None
    x = torch.randn(tokens, hidden, dtype=dtype, device=DEVICE)
    residual = torch.randn(tokens, HC, hidden, dtype=dtype, device=DEVICE)
    post = (
        torch.rand(tokens, HC, dtype=torch.float32, device=DEVICE) * MAGNITUDE + HC_EPS
    )

    ref = _eager(monkeypatch, layer, x, residual, post)
    y = triton_ihc_post(x, residual, post)
    torch.testing.assert_close(y, ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(layer(x, residual, post), y)


def test_ihc_pre_strided_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """A channel-strided view (e.g. a slice of a wider buffer) is handled."""
    torch.manual_seed(0)
    hidden = 1024
    layer = _make_pre(hidden)
    wide = torch.randn(9, HC, 2 * hidden, dtype=torch.bfloat16, device=DEVICE)
    x = wide[:, :, :hidden]
    ref_y, ref_post = _eager(monkeypatch, layer, x)
    y, post = layer(x)
    torch.testing.assert_close(post, ref_post, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y, ref_y, atol=2e-2, rtol=2e-2)


def test_ihc_gate_math_sanity() -> None:
    """With zero projection weight the gates reduce to sigmoid(base) + eps."""
    hidden = 256
    x = torch.randn(3, HC, hidden, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.zeros(2 * HC, HC * hidden, dtype=torch.float32, device=DEVICE)
    scale = torch.full((2,), 0.5, dtype=torch.float32, device=DEVICE)
    base = torch.cat(
        [
            torch.full((HC,), -math.log(HC - 1.0), device=DEVICE),
            torch.zeros(HC, device=DEVICE),
        ]
    )
    y, post = triton_ihc_pre(x, weight, scale, base, MAGNITUDE, HC_EPS, NORM_EPS)
    pre_gate = torch.sigmoid(base[:HC]) + HC_EPS
    expected_y = (pre_gate[None, :, None] * x.float()).sum(1).to(x.dtype)
    torch.testing.assert_close(y, expected_y, atol=2e-2, rtol=2e-2)
    expected_post = (MAGNITUDE * torch.sigmoid(base[HC:]) + HC_EPS).expand(3, HC)
    torch.testing.assert_close(post, expected_post, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("tokens", [1, 5, 64])
def test_ihc_opcheck(tokens: int) -> None:
    hidden = 512
    x = torch.randn(tokens, HC, hidden, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(2 * HC, HC * hidden, dtype=torch.float32, device=DEVICE) * 6e-3
    scale = torch.full((2,), 0.5, dtype=torch.float32, device=DEVICE)
    base = torch.randn(2 * HC, dtype=torch.float32, device=DEVICE)
    torch.library.opcheck(
        torch.ops.vllm.hy_v4_ihc_pre, (x, w, scale, base, MAGNITUDE, HC_EPS, NORM_EPS)
    )
    torch.library.opcheck(
        torch.ops.vllm.hy_v4_ihc_head,
        (x, w[:HC].contiguous(), scale[:1], base[:HC].contiguous(), HC_EPS, NORM_EPS),
    )
    residual = torch.randn(tokens, HC, hidden, dtype=torch.bfloat16, device=DEVICE)
    post = torch.rand(tokens, HC, dtype=torch.float32, device=DEVICE)
    torch.library.opcheck(torch.ops.vllm.hy_v4_ihc_post, (x[:, 0], residual, post))


def _run_all_ops(tokens: int, hidden: int) -> None:
    x = torch.randn(tokens, HC, hidden, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(2 * HC, HC * hidden, dtype=torch.float32, device=DEVICE) * 6e-3
    scale = torch.full((2,), 0.5, dtype=torch.float32, device=DEVICE)
    base = torch.zeros(2 * HC, dtype=torch.float32, device=DEVICE)
    y, post = triton_ihc_pre(x, w, scale, base, MAGNITUDE, HC_EPS, NORM_EPS)
    w_head, base_head = w[:HC].contiguous(), base[:HC].contiguous()
    triton_ihc_head(x, w_head, scale[:1], base_head, HC_EPS, NORM_EPS)
    triton_ihc_post(y, x, post)


def test_ihc_warmup_covers_compile_keys() -> None:
    """One launch per op at startup compiles every Triton variant the model
    will use: the token count is not part of the compile key (what
    ``hy_v4_ihc_warmup`` relies on)."""
    hidden = 512
    device_index = torch.accelerator.current_device_index()
    kernels = (
        triton_ihc._ihc_pre_stage1,
        triton_ihc._ihc_pre_stage2,
        triton_ihc._ihc_post_kernel,
    )
    if not all(hasattr(k, "device_caches") for k in kernels):
        pytest.skip("triton JITFunction.device_caches not available")

    def compiled_variants() -> int:
        return sum(
            len(k.device_caches[device_index][0])
            for k in kernels
            if device_index in k.device_caches
        )

    _run_all_ops(1, hidden)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed > 0
    for tokens in (2, 3, 16, 17, 64, 300, 1024, 2048):
        _run_all_ops(tokens, hidden)
    torch.accelerator.synchronize()
    assert compiled_variants() == warmed
