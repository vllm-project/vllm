# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MHCFusedPostPreOp on FlashInfer's fused mHC kernel (flashinfer.mega_mhc).

The GPU tests run only where FlashInfer ships the hidden-4096 program for the
device. The fused op is checked against a float64 reference with the TileLang
op as the yardstick, for one and several launches, under CUDA graph replay and
after an in-place weight update.
"""

import os

import pytest
import torch

import vllm.model_executor.kernels.mhc  # noqa: F401  (registers the custom ops)
import vllm.model_executor.kernels.mhc.flashinfer_mega as fm
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.worker.workspace import lock_workspace

H, HC = fm.HIDDEN_SIZE, fm.HC_MULT
RMS_EPS, NORM_EPS, HC_EPS, POST_MULT, REPEAT = 1e-5, 1e-5, 1e-6, 2.0, 20
_PTXAS_ENV = "TRITON_PTXAS_BLACKWELL_PATH"


def _mega_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    # Importing FlashInfer points Triton at the system ptxas
    # (TRITON_PTXAS_BLACKWELL_PATH); keep that out of the other mHC tests that
    # share this process. The first call imports it; later calls are cached.
    saved = os.environ.get(_PTXAS_ENV)
    try:
        return fm.mega_mhc_supported(H, HC)
    finally:
        if saved is None:
            os.environ.pop(_PTXAS_ENV, None)
        else:
            os.environ[_PTXAS_ENV] = saved


requires_mega = pytest.mark.skipif(
    not _mega_supported(),
    reason="FlashInfer mega mHC (with its kernel sources) is unavailable here",
)


def _weights(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # vLLM keeps the projection in FP32, upcast from the BF16 checkpoint.
    fn = torch.randn(HC * (2 + HC), HC * H, generator=g, device="cuda") * 1e-2
    fn = fn.bfloat16().float()
    scale = torch.rand(3, generator=g, device="cuda") + 0.5
    base = torch.randn(HC * (2 + HC), generator=g, device="cuda") * 0.1
    norm = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).bfloat16()
    return fn, scale, base, norm


def _inputs(num_tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(num_tokens, HC, H, generator=g, device="cuda").bfloat16()
    x = (torch.randn(num_tokens, H, generator=g, device="cuda") * 0.5).bfloat16()
    post = torch.rand(num_tokens, HC, 1, generator=g, device="cuda") * POST_MULT
    comb = torch.softmax(
        torch.randn(num_tokens, HC, HC, generator=g, device="cuda"), dim=-1
    )
    return x, residual, post, comb


def _mega(inputs, weights, fn_split):
    fn, scale, base, norm = weights
    return torch.ops.vllm.mhc_fused_post_pre_flashinfer_mega(
        *inputs, fn, fn_split, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT,
        REPEAT, norm, NORM_EPS,
    )  # fmt: skip


def _tilelang(inputs, weights):
    fn, scale, base, norm = weights
    return torch.ops.vllm.mhc_fused_post_pre_tilelang(
        *inputs, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, REPEAT, 1,
        1, norm, NORM_EPS,
    )  # fmt: skip


def _reference(inputs, weights):
    """Float64 post block, next pre block and RMSNorm."""
    x, residual, post, comb = (t.double() for t in inputs)
    fn, scale, base, norm = (t.double() for t in weights)
    num_tokens = residual.shape[0]
    res = torch.einsum("tij,tih->tjh", comb, residual) + post * x.unsqueeze(1)
    res_bf = res.to(torch.bfloat16).double()
    z = res_bf.reshape(num_tokens, HC * H)
    mixes = (z @ fn.t()) * torch.rsqrt(z.square().mean(-1, keepdim=True) + RMS_EPS)
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post_new = torch.sigmoid(mixes[:, HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    cm = mixes[:, 2 * HC :].view(-1, HC, HC) * scale[2] + base[2 * HC :].view(HC, HC)
    cm = torch.softmax(cm, dim=-1) + HC_EPS
    cm = cm / (cm.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(REPEAT - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + HC_EPS)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + HC_EPS)
    layer_input = (pre.unsqueeze(-1) * res_bf).sum(dim=1)
    rms = torch.rsqrt(layer_input.square().mean(-1, keepdim=True) + NORM_EPS)
    return res, (post_new * POST_MULT).unsqueeze(-1), cm, layer_input * rms * norm


def _errors(out, ref):
    def rel_l2(a, b):
        return float((a.double() - b).norm() / b.norm().clamp_min(1e-30))

    return {
        "residual": rel_l2(out[0], ref[0]),
        "post": float((out[1].double() - ref[1]).abs().max()),
        "comb": float((out[2].double() - ref[2]).abs().max()),
        "y": rel_l2(out[3], ref[3]),
    }


def _assert_close(out, inputs, weights):
    assert all(bool(torch.isfinite(t).all()) for t in out)
    ref = _reference(inputs, weights)
    err, err_tl = _errors(out, ref), _errors(_tilelang(inputs, weights), ref)
    floors = {"residual": 1e-3, "post": 1e-4, "comb": 1e-4, "y": 1e-3}
    for key, floor in floors.items():
        assert err[key] <= 2 * err_tl[key] + floor, (key, err, err_tl)


def _num_sms() -> int:
    return num_compute_units()


@requires_mega
@pytest.mark.parametrize("num_tokens", [1, 5, 8, 33, 256, 4096])
def test_matches_reference(workspace_init, num_tokens):
    weights, inputs = _weights(num_tokens), _inputs(num_tokens, num_tokens + 1)
    _assert_close(_mega(inputs, weights, fm.split_mhc_fn(weights[0])), inputs, weights)


@requires_mega
def test_matches_reference_over_several_launches(workspace_init):
    # One launch serves at most 64 tokens per SM; the last launch gets 3.
    weights, inputs = _weights(11), _inputs(64 * _num_sms() + 3, 12)
    _assert_close(_mega(inputs, weights, fm.split_mhc_fn(weights[0])), inputs, weights)


@requires_mega
def test_in_place_weight_update_reaches_the_split(workspace_init):
    weights, inputs = _weights(41), _inputs(64, 42)
    fn_split = fm.split_mhc_fn(weights[0])
    _mega(inputs, weights, fn_split)
    # Weight updates write fn in place; the loader rewrites its split in place.
    weights[0].copy_(_weights(43)[0])
    fm.split_mhc_fn(weights[0], out=fn_split)
    _assert_close(_mega(inputs, weights, fn_split), inputs, weights)


@requires_mega
@pytest.mark.parametrize("num_tokens", [5, 64, 512])
def test_cuda_graph_replay_matches_eager(workspace_init, num_tokens):
    weights, inputs = _weights(21), _inputs(num_tokens, 22)
    fn_split = fm.split_mhc_fn(weights[0])
    _mega(inputs, weights, fn_split)  # eager first: allocates the counters
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _mega(inputs, weights, fn_split)
    lock_workspace()
    for dst, src in zip(inputs, _inputs(num_tokens, 23), strict=True):
        dst.copy_(src)
    eager = [t.clone() for t in _mega(inputs, weights, fn_split)]
    graph.replay()
    torch.accelerator.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))


@requires_mega
def test_warmup_keys_cover_every_selected_program(workspace_init):
    from flashinfer.mega_mhc import prepare_mega_mhc

    max_tokens = 64 * _num_sms()
    keys = fm.MEGA_MHC_KERNEL.get_warmup_keys(max_num_batched_tokens=max_tokens)
    weights = _weights(51)
    # Remainders of the chunk loop reach 1-4 tokens too.
    for num_tokens in (1, 3, 5, 384, 385, 832, 833, 3392, 3393, max_tokens):
        x, residual, post, comb = _inputs(num_tokens, num_tokens)
        plan = prepare_mega_mhc(
            x=x, residual=residual, post_mix=post, comb_res_mix=comb,
            shifted_prev_mix=None, fn=weights[0], mix_scales=weights[1],
            mix_bases=weights[2], rmsnorm_weight=weights[3], sf_layout="bf16",
        )  # fmt: skip
        assert any(key.program == plan.program for key in keys), num_tokens


def test_other_shapes_are_unsupported():
    assert not fm.mega_mhc_supported(5120, HC)
    assert not fm.mega_mhc_supported(H, 2)
