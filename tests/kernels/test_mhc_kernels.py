# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools

import pytest
import torch

import vllm.model_executor.kernels.mhc  # noqa: F401
from vllm.model_executor.kernels.mhc.tilelang import (
    _hc_prenorm_gemm,
    _select_hc_prenorm_gemm_backend,
    _tilelang_hc_prenorm_gemm,
    _torch_hc_prenorm_gemm,
)
from vllm.model_executor.layers.mhc import HAS_TILELANG_MHC
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type
requires_cutedsl_mhc = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability(100)
    or not has_cutedsl(),
    reason="CuTeDSL MHC requires CUDA SM100 and cutlass",
)


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre reference kernel from tilelang repo: https://github.com/tile-ai/tilelang/blob/d135bd1cd2d2eee74fbb41dd0a0831a427194c86/examples/deepseek_mhc/example_mhc_pre.py#L303"""
    hc_mult = residual.shape[-2]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )

    hc_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ],
    )
    mixes = mixes * hc_scale + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

    res_mix = sinkhorn_normalize_ref(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )

    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post reference kernel from tilelang repo: https://github.com/tile-ai/tilelang/blob/d135bd1cd2d2eee74fbb41dd0a0831a427194c86/examples/deepseek_mhc/example_mhc_post.py#L68"""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def hc_head_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    residual_flat = residual.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * residual.float(), dim=-2).bfloat16()


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_tilelang(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = 2 * hc_mult + hc_mult2
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(hc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1

    hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0

    ref = mhc_pre_ref(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )
    out = torch.ops.vllm.mhc_pre_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )

    for actual, expected in zip(out, ref, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize(
    ("num_tokens", "hidden_size"),
    [
        (1, 1280),
        (512, 1280),
        (2048, 1280),
        (1, 4096),
        (64, 4096),
        (512, 4096),
        (2048, 4096),
        (1, 7168),
        (64, 7168),
        (512, 7168),
        (2048, 7168),
    ],
)
def test_hc_prenorm_gemm_tilelang(num_tokens, hidden_size):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    hc_mult = 4
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    x = torch.randn((num_tokens, hc_mult * hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult3, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    out_ref = torch.empty((1, num_tokens, hc_mult3), dtype=torch.float32)
    sqrsum_ref = torch.empty((1, num_tokens), dtype=torch.float32)
    out = torch.empty_like(out_ref)
    sqrsum = torch.empty_like(sqrsum_ref)

    _torch_hc_prenorm_gemm(x, fn, out_ref, sqrsum_ref)
    _tilelang_hc_prenorm_gemm(x, fn, out, sqrsum, hidden_size, hc_mult)

    torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(sqrsum, sqrsum_ref, atol=8.0, rtol=5e-4)


@requires_cutedsl_mhc
@pytest.mark.parametrize(
    ("num_tokens", "k", "n_splits"),
    [
        *itertools.product(
            [13, 137],
            [5120, 7168, 7680, 16384, 28672],
            [1, 4, 16],
        ),
        pytest.param(137, 16384, 49, id="non-power-split"),
        pytest.param(128, 16384, 64, id="max-split"),
    ],
)
@torch.inference_mode()
def test_hc_prenorm_gemm_cutedsl(num_tokens, k, n_splits):
    set_random_seed(0)

    hc_mult = 4
    hc_mult3 = 24
    hidden_size = k // hc_mult

    x = torch.randn((num_tokens, k), dtype=torch.bfloat16, device=DEVICE)
    fn = torch.randn((hc_mult3, k), dtype=torch.float32, device=DEVICE) * 1e-4

    out_ref = torch.empty((1, num_tokens, hc_mult3), dtype=torch.float32, device=DEVICE)
    sqrsum_ref = torch.empty((1, num_tokens), dtype=torch.float32, device=DEVICE)
    out = torch.full(
        (n_splits, num_tokens, hc_mult3),
        float("nan"),
        dtype=torch.float32,
        device=DEVICE,
    )
    sqrsum = torch.full(
        (n_splits, num_tokens),
        float("nan"),
        dtype=torch.float32,
        device=DEVICE,
    )

    _torch_hc_prenorm_gemm(x, fn, out_ref, sqrsum_ref)
    _hc_prenorm_gemm(
        x,
        fn,
        out,
        sqrsum,
        hidden_size,
        hc_mult,
        n_splits=n_splits,
    )

    torch.testing.assert_close(out.sum(0), out_ref[0], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(sqrsum.sum(0), sqrsum_ref[0], atol=8.0, rtol=5e-4)


@requires_cutedsl_mhc
def test_can_use_hc_prenorm_gemm_cutedsl_split_bounds():
    from vllm.model_executor.kernels.mhc.cutedsl import can_use_hc_prenorm_gemm

    k = 16384
    x = torch.empty((1, k), dtype=torch.bfloat16, device=DEVICE)
    fn = torch.empty((24, k), dtype=torch.float32, device=DEVICE)

    assert can_use_hc_prenorm_gemm(x, fn, 64)
    assert not can_use_hc_prenorm_gemm(x, fn, 0)
    assert not can_use_hc_prenorm_gemm(x, fn, 65)
    assert _select_hc_prenorm_gemm_backend(x, fn, 49) == (True, False, 49)


@requires_cutedsl_mhc
def test_warmup_hc_prenorm_gemm_cutedsl(monkeypatch):
    from vllm.model_executor.kernels.mhc import cutedsl, tilelang_kernels

    compile_calls = []
    monkeypatch.setattr(cutedsl, "_compile", lambda k, n: compile_calls.append((k, n)))
    monkeypatch.setattr(
        tilelang_kernels,
        "compute_num_split",
        lambda _block_k, _k, grid: 2 if grid < 3 else 1,
    )

    cutedsl.warmup_hc_prenorm_gemm(16384, 192)

    assert compile_calls == [(16384, 2), (16384, 1)]


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_post_tilelang(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    out = torch.ops.vllm.mhc_post_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )

    torch.testing.assert_close(out, ref, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_fused_post_pre(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16)
    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    post_layer_mix = torch.randn((num_tokens, hc_mult, 1), dtype=torch.float32)
    comb_res_mix = torch.randn((num_tokens, hc_mult, hc_mult), dtype=torch.float32)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float)
        * 1e-4
        * (1 + torch.arange(hc_mult).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float) * 0.1

    hc_sinkhorn_eps = hc_pre_eps = rms_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0

    def run_ref():
        residual_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
        post_mix_ref, res_mix_ref, layer_input_ref = mhc_pre_ref(
            residual_ref,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_alpha,
            sinkhorn_repeat,
        )
        return residual_ref, post_mix_ref, res_mix_ref, layer_input_ref

    residual_ref, post_mix_ref, res_mix_ref, layer_input_ref = run_ref()

    residual, post_mix, res_mix, x = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
    )

    torch.testing.assert_close(residual, residual_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(post_mix, post_mix_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(res_mix, res_mix_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(x, layer_input_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head_triton(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((1,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32) * 0.1
    rms_eps = hc_eps = 1e-6

    out = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16)
    out.fill_(float("nan"))

    result = torch.ops.vllm.hc_head_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )

    assert result is None
    assert not torch.isnan(out).any()

    out_ref = hc_head_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_eps)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not HAS_TILELANG_MHC,
    reason="TileLang MHC support required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head_tilelang(num_tokens, hidden_size, hc_mult):
    torch.set_default_device(DEVICE)
    set_random_seed(0)

    residual = torch.randn((num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16)
    fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32) * 1e-4
    hc_scale = torch.randn((1,), dtype=torch.float32) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32) * 0.1
    rms_eps = hc_eps = 1e-6

    out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
    )

    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()

    out_ref = hc_head_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_eps)
    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=1e-2)
