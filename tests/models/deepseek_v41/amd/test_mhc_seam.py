# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 delayed mHC seam of the DeepSeek-V4.1 AMD decoder."""

import pytest
import torch

from vllm.model_executor.kernels.mhc.torch import (
    mhc_post_torch,
    mhc_pre_delayed_torch,
)
from vllm.model_executor.layers.mhc import MHCPreDelayedOp
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm required", allow_module_level=True)

from vllm.models.deepseek_v41.amd.ops import mhc_seam  # noqa: E402

DEVICE = current_platform.device_type
HC, HIDDEN = 4, 5120


def test_fake_residual_is_contiguous():
    """The compile fake returns a contiguous residual, as AITER does.

    AITER copies `residual` contiguous before the store. `empty_like` would
    keep a permuted residual's strides, and torch.compile rejects that fake.
    """
    residual = torch.empty(5, 9, 4, dtype=torch.bfloat16).permute(0, 2, 1)
    assert not residual.is_contiguous()
    t = residual
    residual_out, post, comb, layer_input, pre = mhc_seam._seam_fake(
        t, t, t, t, 1e-2, 1e-3, 1e-4, 2.0, 20, None, t, t, t
    )
    assert residual_out.is_contiguous()
    assert residual_out.shape == (5, 4, 9) and residual_out.dtype == torch.bfloat16
    assert post.shape == (5, 4, 1) and post.dtype == torch.float32
    assert comb.shape == (5, 4, 4) and comb.dtype == torch.float32
    assert layer_input.shape == (5, 9) and layer_input.dtype == torch.bfloat16
    assert pre.shape == (5, 4) and pre.dtype == torch.float32


def _inputs(num_tokens, strided):
    shape = (num_tokens, HC, HIDDEN)
    if strided:
        residual = torch.empty(
            num_tokens, HC, HIDDEN + 8, dtype=torch.bfloat16, device=DEVICE
        )[..., :HIDDEN]
        residual.copy_(torch.randn(shape, dtype=torch.bfloat16, device=DEVICE))
        assert not residual.is_contiguous()
    else:
        residual = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    mix_hc = 2 * HC + HC * HC
    fn = torch.randn(mix_hc, HC * HIDDEN, dtype=torch.float32, device=DEVICE) * 1e-4
    scale = torch.randn(3, dtype=torch.float32, device=DEVICE) * 0.1
    base = torch.randn(mix_hc, dtype=torch.float32, device=DEVICE) * 0.1
    return residual, fn, scale, base


@pytest.mark.skipif(
    not mhc_seam.supported(HIDDEN, HC), reason="AITER gfx942 mHC seam required"
)
@pytest.mark.parametrize("num_tokens,strided", [(63, False), (64, False), (64, True)])
@pytest.mark.parametrize("carried", [False, True])
def test_seam_dispatch(num_tokens, strided, carried, default_vllm_config, monkeypatch):
    """The seam runs from 64 tokens and matches post then pre.

    63 stays on `MHCPreDelayedOp`. A strided residual is the case AITER
    copies contiguous before the kernel. The three eps values differ, so a
    swapped argument fails.
    """
    set_random_seed(0)
    residual, fn, scale, base = _inputs(num_tokens, strided)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    post = torch.rand(num_tokens, HC, 1, dtype=torch.float32, device=DEVICE) + 0.5
    comb = torch.rand(num_tokens, HC, HC, dtype=torch.float32, device=DEVICE) + 0.2
    pre_mix = (
        torch.rand(num_tokens, HC, dtype=torch.float32, device=DEVICE) + 0.5
        if carried
        else None
    )
    pre_args = (fn, scale, base, 1e-2, 1e-3, 1e-4, 2.0, 20)
    fused = torch.ops.vllm.mhc_fused_post_pre_delayed_gfx942
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return fused(*args, **kwargs)

    monkeypatch.setattr(torch.ops.vllm, "mhc_fused_post_pre_delayed_gfx942", spy)
    actual = mhc_seam.mhc_pre_delayed_seam(
        MHCPreDelayedOp(),
        residual,
        *pre_args,
        pre_mix=pre_mix,
        sublayer_out=x,
        post_layer_mix=post,
        comb_res_mix=comb,
    )

    assert len(calls) == (num_tokens >= mhc_seam.MIN_TOKENS)
    if strided:
        assert actual[0].is_contiguous()
    next_residual = mhc_post_torch(x, residual, post, comb)
    torch.testing.assert_close(actual[0], next_residual, atol=2e-2, rtol=8e-3)
    # The pre reference runs on the kernel's own bf16 residual: a one-ulp
    # post difference otherwise shows up in the collapsed layer input.
    expected = mhc_pre_delayed_torch(actual[0], *pre_args, pre_mix=pre_mix)
    for got, want in zip(actual[1:], expected, strict=True):
        torch.testing.assert_close(got, want.reshape(got.shape), atol=2e-2, rtol=8e-3)


@pytest.mark.parametrize("case", ["no_post", "layer0", "norm"])
def test_other_seams_keep_pre_delayed(case, monkeypatch):
    """Seams without a post block, layer 0 and fused-norm seams never call
    the AITER seam, whatever the token count."""

    def no_seam(*args, **kwargs):
        raise AssertionError("AITER seam called")

    monkeypatch.setattr(torch.ops.vllm, "mhc_fused_post_pre_delayed_gfx942", no_seam)
    seen = {}

    def pre_delayed(*args, **kwargs):
        seen.update(kwargs)
        return "pre_delayed"

    n = 4 * mhc_seam.MIN_TOKENS
    residual = torch.empty(n, HC, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    t = torch.empty(1, device=DEVICE)
    kwargs = dict(sublayer_out=t, post_layer_mix=t, comb_res_mix=t)
    if case == "no_post":
        kwargs = {}
    elif case == "layer0":
        kwargs["x"] = t
    else:
        kwargs["norm_weight"] = t
    out = mhc_seam.mhc_pre_delayed_seam(
        pre_delayed, residual, t, t, t, 1e-6, 1e-6, 1e-6, 2.0, 20, **kwargs
    )
    assert out == "pre_delayed"
    for name, value in kwargs.items():
        assert seen[name] is value
