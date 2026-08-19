# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM75 (Turing) MHC: force the torch/Triton fallback with FP16 compute."""

import pytest
import torch

from vllm.model_executor.layers import mhc as mhc_mod
from vllm.model_executor.layers.mhc import (
    HCHeadOp,
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.platforms import current_platform

_capability = current_platform.get_device_capability()

pytestmark = pytest.mark.skipif(
    _capability is None or (_capability.major, _capability.minor) != (7, 5),
    reason="SM75 only",
)

_HC_MULT = 4
_HIDDEN = 128


def _turing_dtypes() -> tuple[torch.dtype, torch.dtype]:
    assert mhc_mod._is_turing()
    return torch.float16, mhc_mod._mhc_fallback_dtype()


def test_mhc_torch_fallback_fp16():
    import importlib

    mhc_torch = importlib.import_module("vllm.model_executor.kernels.mhc.torch")

    # TileLang MHC kernels are sm90+; SM75 must use the torch fallback.
    assert not mhc_mod.HAS_TILELANG_MHC
    assert hasattr(mhc_torch, "mhc_pre_torch")
    assert hasattr(mhc_torch, "mhc_post_torch")
    compute_dtype, fallback_dtype = _turing_dtypes()
    assert compute_dtype == torch.float16
    assert fallback_dtype == torch.float16


def test_mhc_pre_post_ops_fp16(default_vllm_config):
    compute_dtype, _ = _turing_dtypes()
    hc_mult, hidden = _HC_MULT, _HIDDEN
    num_tokens = 3
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual = (torch.randn(num_tokens, hc_mult, hidden, device="cuda") * 0.1).to(
        compute_dtype
    )
    fn = torch.randn(hc_mult3, hc_mult * hidden, device="cuda")
    hc_scale = torch.randn(3, device="cuda")
    hc_base = torch.randn(hc_mult3, device="cuda")
    rms_eps = 1e-6
    hc_eps = 1e-2
    hc_post_mult_value = 2.0
    sinkhorn_repeat = 3

    pre = MHCPreOp()
    post_mix, comb_mix, layer_input = pre(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    assert layer_input.dtype == compute_dtype
    assert post_mix.shape == (num_tokens, hc_mult, 1)
    assert comb_mix.shape == (num_tokens, hc_mult, hc_mult)
    assert torch.isfinite(layer_input).all()

    x = (torch.randn(num_tokens, hidden, device="cuda") * 0.1).to(compute_dtype)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device="cuda")
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device="cuda")
    out = MHCPostOp()(x, residual, post_layer_mix, comb_res_mix)
    assert out.dtype == compute_dtype
    assert out.shape == (num_tokens, hc_mult, hidden)
    assert torch.isfinite(out).all()


def test_mhc_fused_post_pre_fp16(default_vllm_config):
    compute_dtype, _ = _turing_dtypes()
    hc_mult, hidden = _HC_MULT, _HIDDEN
    num_tokens = 3
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    x = (torch.randn(num_tokens, hidden, device="cuda") * 0.1).to(compute_dtype)
    residual = (torch.randn(num_tokens, hc_mult, hidden, device="cuda") * 0.1).to(
        compute_dtype
    )
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device="cuda")
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device="cuda")
    fn = torch.randn(hc_mult3, hc_mult * hidden, device="cuda")
    hc_scale = torch.randn(3, device="cuda")
    hc_base = torch.randn(hc_mult3, device="cuda")
    hc_eps = 1e-2

    residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = MHCFusedPostPreOp()(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        hc_eps,
        hc_eps,
        2.0,
        3,
    )
    assert layer_input_cur.dtype == compute_dtype
    assert residual_cur.dtype == compute_dtype
    assert torch.isfinite(layer_input_cur).all()
    assert torch.isfinite(residual_cur).all()


def test_hc_head_fp16(default_vllm_config):
    compute_dtype, fallback_dtype = _turing_dtypes()
    hc_mult, hidden = _HC_MULT, _HIDDEN
    num_tokens = 3

    hidden_states = (torch.randn(num_tokens, hc_mult, hidden, device="cuda") * 0.1).to(
        compute_dtype
    )
    hc_fn = torch.randn(hc_mult, hc_mult * hidden, device="cuda")
    hc_scale = torch.randn(hc_mult, device="cuda")
    hc_base = torch.randn(hc_mult, device="cuda")

    out = HCHeadOp()(
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-2,
    )
    assert out.dtype == fallback_dtype
    assert out.shape == (num_tokens, hidden)
    assert torch.isfinite(out).all()
