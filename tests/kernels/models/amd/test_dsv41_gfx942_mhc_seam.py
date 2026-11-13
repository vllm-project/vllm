# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 delayed mHC seam of the DeepSeek-V4.1 AMD decoder."""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.kernels.mhc.torch import (
    mhc_post_torch,
    mhc_pre_delayed_torch,
)
from vllm.model_executor.layers.mhc import MHCPreDelayedOp
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_rocm():
    pytest.skip("ROCm required", allow_module_level=True)

from vllm.models.deepseek_v41.amd import mhc as amd_mhc  # noqa: E402

DEVICE = current_platform.device_type
HC, HIDDEN = 4, 5120

requires_seam = pytest.mark.skipif(
    not amd_mhc.supports_mhc_seam(HIDDEN, HC), reason="AITER gfx942 mHC seam required"
)


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


def _seam_args(num_tokens, strided, carried):
    residual, fn, scale, base = _inputs(num_tokens, strided)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    post = torch.rand(num_tokens, HC, 1, dtype=torch.float32, device=DEVICE) + 0.5
    comb = torch.rand(num_tokens, HC, HC, dtype=torch.float32, device=DEVICE) + 0.2
    pre_mix = (
        torch.rand(num_tokens, HC, dtype=torch.float32, device=DEVICE) + 0.5
        if carried
        else None
    )
    return residual, fn, scale, base, x, post, comb, pre_mix


@requires_seam
@pytest.mark.parametrize("carried", [False, True])
def test_seam_op(carried):
    """The registered op and its fake agree, with a strided residual."""
    set_random_seed(0)
    residual, fn, scale, base, x, post, comb, pre_mix = _seam_args(64, True, carried)
    opcheck(
        torch.ops.vllm.deepseek_v41_mhc_fused_post_pre_delayed_aiter.default,
        (residual, fn, scale, base, 1e-2, 1e-3, 1e-4, 2.0, 20, pre_mix, x, post, comb),
    )


@requires_seam
@pytest.mark.parametrize("num_tokens,strided", [(63, False), (64, False), (64, True)])
@pytest.mark.parametrize("carried", [False, True])
def test_seam_dispatch(num_tokens, strided, carried, default_vllm_config, monkeypatch):
    """The seam runs from 64 tokens and matches post then pre.

    63 stays on `MHCPreDelayedOp`. A strided residual is the case AITER
    copies contiguous before the kernel. The three eps values differ, so a
    swapped argument fails.
    """
    set_random_seed(0)
    residual, fn, scale, base, x, post, comb, pre_mix = _seam_args(
        num_tokens, strided, carried
    )
    pre_args = (fn, scale, base, 1e-2, 1e-3, 1e-4, 2.0, 20)
    fused = torch.ops.vllm.deepseek_v41_mhc_fused_post_pre_delayed_aiter
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return fused(*args, **kwargs)

    monkeypatch.setattr(
        torch.ops.vllm, "deepseek_v41_mhc_fused_post_pre_delayed_aiter", spy
    )
    actual = amd_mhc.mhc_pre_delayed_seam(
        MHCPreDelayedOp(),
        residual,
        *pre_args,
        pre_mix=pre_mix,
        sublayer_out=x,
        post_layer_mix=post,
        comb_res_mix=comb,
    )

    assert len(calls) == (num_tokens >= amd_mhc.MHC_SEAM_MIN_TOKENS)
    if strided:
        assert actual[0].is_contiguous()
    next_residual = mhc_post_torch(x, residual, post, comb)
    torch.testing.assert_close(actual[0], next_residual, atol=2e-2, rtol=8e-3)
    # The pre reference runs on the kernel's own bf16 residual: a one-ulp
    # post difference otherwise shows up in the collapsed layer input.
    expected = mhc_pre_delayed_torch(actual[0], *pre_args, pre_mix=pre_mix)
    for got, want in zip(actual[1:], expected, strict=True):
        torch.testing.assert_close(got, want.reshape(got.shape), atol=2e-2, rtol=8e-3)


def test_no_post_seam_keeps_pre_delayed(monkeypatch):
    """A seam without a post block (Engram) never calls the AITER seam,
    whatever the token count."""

    def no_seam(*args, **kwargs):
        raise AssertionError("AITER seam called")

    monkeypatch.setattr(
        torch.ops.vllm, "deepseek_v41_mhc_fused_post_pre_delayed_aiter", no_seam
    )
    seen = {}

    def pre_delayed(*args, **kwargs):
        seen.update(kwargs)
        return "pre_delayed"

    n = 4 * amd_mhc.MHC_SEAM_MIN_TOKENS
    residual = torch.empty(n, HC, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    t = torch.empty(1, device=DEVICE)
    out = amd_mhc.mhc_pre_delayed_seam(
        pre_delayed, residual, t, t, t, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=t
    )
    assert out == "pre_delayed"
    assert seen["pre_mix"] is t and seen["sublayer_out"] is None
