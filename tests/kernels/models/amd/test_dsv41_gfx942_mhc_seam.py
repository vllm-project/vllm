# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx942 delayed mHC seam of the DeepSeek-V4.1 AMD decoder."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

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
from vllm.platforms.rocm import get_cdna_version  # noqa: E402

DEVICE = current_platform.device_type
HC, HIDDEN = 4, 5120

requires_cdna3 = pytest.mark.skipif(get_cdna_version() != 3, reason="CDNA 3 only")


@requires_cdna3
def test_seam_supported():
    """On CDNA 3 the seam tests run instead of skipping."""
    assert amd_mhc.supports_mhc_seam(HIDDEN, HC)


def _seam_args(num_tokens):
    """A seam as the decoder sends it: contiguous residual, carried pre gate."""
    residual = torch.randn(num_tokens, HC, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    mix_hc = 2 * HC + HC * HC
    fn = torch.randn(mix_hc, HC * HIDDEN, dtype=torch.float32, device=DEVICE) * 1e-4
    scale = torch.randn(3, dtype=torch.float32, device=DEVICE) * 0.1
    base = torch.randn(mix_hc, dtype=torch.float32, device=DEVICE) * 0.1
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    post = torch.rand(num_tokens, HC, 1, dtype=torch.float32, device=DEVICE) + 0.5
    comb = torch.rand(num_tokens, HC, HC, dtype=torch.float32, device=DEVICE) + 0.2
    pre_mix = torch.rand(num_tokens, HC, dtype=torch.float32, device=DEVICE) + 0.5
    return residual, fn, scale, base, x, post, comb, pre_mix


def _seam_spy(monkeypatch):
    """Count the calls `mhc_pre_delayed_seam` makes to AITER's seam."""
    from aiter.ops.mhc import mhc_fused_post_pre_delayed as fused

    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return fused(*args, **kwargs)

    monkeypatch.setattr("aiter.ops.mhc.mhc_fused_post_pre_delayed", spy)
    return fused, calls


@requires_cdna3
@pytest.mark.parametrize("num_tokens", [63, 64])
def test_seam_dispatch(num_tokens, default_vllm_config, monkeypatch):
    """The seam runs from 64 tokens and matches post then pre.

    63 stays on `MHCPreDelayedOp`. The three eps values differ, so a swapped
    argument fails.
    """
    set_random_seed(0)
    residual, fn, scale, base, x, post, comb, pre_mix = _seam_args(num_tokens)
    pre_args = (fn, scale, base, 1e-2, 1e-3, 1e-4, 2.0, 20)
    _, calls = _seam_spy(monkeypatch)
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
    next_residual = mhc_post_torch(x, residual, post, comb)
    torch.testing.assert_close(actual[0], next_residual, atol=2e-2, rtol=8e-3)
    # The pre reference runs on the kernel's own bf16 residual: a one-ulp
    # post difference otherwise shows up in the collapsed layer input.
    expected = mhc_pre_delayed_torch(actual[0], *pre_args, pre_mix=pre_mix)
    for got, want in zip(actual[1:], expected, strict=True):
        torch.testing.assert_close(got, want.reshape(got.shape), atol=2e-2, rtol=8e-3)


@requires_cdna3
def test_write_kv_takes_seam(default_vllm_config, monkeypatch):
    """The replay KV write runs the seam from 64 tokens and hands its layer
    input to `forward_kv`."""
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer

    set_random_seed(0)
    num_tokens = amd_mhc.MHC_SEAM_MIN_TOKENS
    residual, fn, scale, base, x, post, comb, pre_mix = _seam_args(num_tokens)
    fused, calls = _seam_spy(monkeypatch)
    written: dict[str, torch.Tensor] = {}
    attn_norm = nn.Identity()
    attn_norm.variance_epsilon = 1e-6
    layer = SimpleNamespace(
        engram=None,
        use_sequence_parallel=False,
        fuse_seam_norm=False,
        mhc_seam=partial(amd_mhc.mhc_pre_delayed_seam, MHCPreDelayedOp()),
        mhc_pre_delayed=None,
        hc_attn_fn=fn,
        hc_attn_fn_packed=None,
        hc_attn_scale=scale,
        hc_attn_base=base,
        rms_norm_eps=1e-2,
        hc_eps=1e-3,
        hc_post_alpha=2.0,
        hc_sinkhorn_iters=20,
        attn_norm=attn_norm,
        attn=SimpleNamespace(forward_kv=lambda positions, h: written.update(x=h)),
    )
    positions = torch.arange(num_tokens, device=DEVICE)
    DeepseekV4DecoderLayer.write_kv(
        layer, x, positions, None, pre_mix, post, comb, residual
    )

    assert calls == [1]
    expected = fused(
        residual, fn, scale, base, 1e-2, 1e-3, 1e-3, 2.0, 20, pre_mix, x, post, comb
    )
    torch.testing.assert_close(written["x"], expected[3])


@requires_cdna3
def test_seam_graph_replay(default_vllm_config):
    """After one eager call, as vLLM makes before capture, the seam captures
    in a CUDA graph and its replay on new inputs matches the eager call."""
    set_random_seed(0)
    num_tokens = amd_mhc.MHC_SEAM_MIN_TOKENS
    static = _seam_args(num_tokens)
    residual, fn, scale, base, x, post, comb, pre_mix = static
    pre_delayed = MHCPreDelayedOp()

    def seam():
        return amd_mhc.mhc_pre_delayed_seam(
            pre_delayed,
            residual,
            fn,
            scale,
            base,
            1e-2,
            1e-3,
            1e-4,
            2.0,
            20,
            pre_mix=pre_mix,
            sublayer_out=x,
            post_layer_mix=post,
            comb_res_mix=comb,
        )

    seam()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = seam()

    for buf, new in zip(static, _seam_args(num_tokens), strict=True):
        buf.copy_(new)
    graph.replay()
    torch.accelerator.synchronize()
    for got, want in zip(captured, seam(), strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("seam", ["no_post", "packed_fn"])
def test_seam_keeps_pre_delayed(seam, monkeypatch):
    """A seam without a post block (Engram) or with a packed `fn` (Mega-MHC)
    never calls the AITER seam, whatever the token count, and `pre_delayed`
    gets its arguments."""

    def no_seam(*args, **kwargs):
        raise AssertionError("AITER seam called")

    monkeypatch.setattr("aiter.ops.mhc.mhc_fused_post_pre_delayed", no_seam)
    seen = {}

    def pre_delayed(*args, **kwargs):
        seen.update(kwargs)
        return "pre_delayed"

    n = 4 * amd_mhc.MHC_SEAM_MIN_TOKENS
    residual = torch.empty(n, HC, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    t = torch.empty(1, device=DEVICE)
    kwargs = (
        {}
        if seam == "no_post"
        else dict(sublayer_out=t, post_layer_mix=t, comb_res_mix=t, fn_packed=t)
    )
    out = amd_mhc.mhc_pre_delayed_seam(
        pre_delayed, residual, t, t, t, 1e-6, 1e-6, 1e-6, 2.0, 20, pre_mix=t, **kwargs
    )
    assert out == "pre_delayed"
    assert seen["pre_mix"] is t
    for name in ("sublayer_out", "post_layer_mix", "comb_res_mix", "fn_packed"):
        assert seen[name] is kwargs.get(name)
