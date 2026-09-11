# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.model_executor.kernels.mhc  # noqa: F401
import vllm.model_executor.layers.mhc as mhc_layers
from vllm.model_executor.kernels.mhc.tilelang import (
    _tilelang_hc_prenorm_gemm,
    _torch_hc_prenorm_gemm,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.mhc import (
    HAS_AITER_MHC,
    HAS_AITER_MHC_FUSED,
    HAS_AITER_MHC_FUSED_NORM,
    HAS_AITER_MHC_PRE_NORM,
    HAS_TILELANG_MHC,
    MHCFusedPostPreOp,
    MHCPreOp,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4Model,
)
from vllm.models.deepseek_v4_1.nvidia.model import (
    DeepseekV4DecoderLayer as DeepseekV41DecoderLayer,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DEVICE = current_platform.device_type


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    "num_tokens,hc_mult,hidden_size",
    [
        (0, 4, 5120),
        (1, 4, 5120),
        (33, 4, 5137),
        (128, 4, 5120),
        (1024, 4, 5120),
        (2, 1, 513),
        (7, 8, 5137),
        (65537, 4, 65),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
def test_hc_collapse_preserves_weighted_residual_sum(
    num_tokens, hc_mult, hidden_size, strided
):
    """Handle tile tails and strided streams without changing FP32 mixing."""
    set_random_seed(0)
    x = torch.randn(
        num_tokens,
        hc_mult,
        hidden_size * (2 if strided else 1),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    pre = torch.rand(num_tokens, hc_mult, device=DEVICE)
    if strided:
        x = x[..., ::2]
        pre = pre.t().contiguous().t()
    if num_tokens:
        pre[0].zero_()
        pre[0, -1] = 1
    if num_tokens > 1 and hc_mult == 4:
        pre[1].fill_(1)
        x[1, 1] = -x[1, 0]
        x[1, 3] = -x[1, 2]
    expected = (pre.unsqueeze(-1) * x.float()).sum(dim=1).to(x.dtype)
    actual = hc_collapse_triton(x, pre)
    assert actual.is_contiguous()
    # BF16 rounding only: ptxas <13.1 contracts the FP32 mix into FFMA despite
    # enable_fp_fusion=False, so compare within a ULP and pin the exact cases below.
    torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1e-2)
    if num_tokens:
        torch.testing.assert_close(actual[0], x[0, -1], atol=0, rtol=0)
    if num_tokens > 1 and hc_mult == 4:
        torch.testing.assert_close(
            actual[1], torch.zeros_like(actual[1]), atol=0, rtol=0
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_hc_collapse_custom_op_supports_compile():
    x = torch.randn(2, 4, 5120, dtype=torch.bfloat16, device=DEVICE)
    pre = torch.rand(2, 4, device=DEVICE)
    torch.library.opcheck(torch.ops.vllm.hc_collapse_triton.default, (x, pre))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_v41_dspark_head_collapses_with_last_ffn_mix(num_tokens, monkeypatch):
    """Match reference DSparkBlock.forward_head's hc_pre before its RMSNorm."""
    from vllm.models.deepseek_v4_1.nvidia import dspark

    set_random_seed(0)
    hidden_size, hc_mult = 5120, 4
    streams = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=DEVICE
    )
    mixes = [torch.rand(num_tokens, hc_mult, device=DEVICE) for _ in range(2)]
    carried_mixes = []

    def make_layer(mix):
        def forward(hidden, positions, ids, pre, post, res, residual):
            carried_mixes.append(pre)
            return hidden, None, None, None, mix

        return forward

    monkeypatch.setattr(dspark, "mhc_post_tilelang", lambda *args: streams)
    draft = SimpleNamespace(
        use_sequence_parallel=False,
        hc_mult=hc_mult,
        layers=[make_layer(mix) for mix in mixes],
    )
    actual = dspark.DSparkDeepseekV4Model.forward(
        draft,
        input_ids=torch.zeros(num_tokens, dtype=torch.long, device=DEVICE),
        positions=torch.arange(num_tokens, device=DEVICE),
        inputs_embeds=torch.zeros(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=DEVICE
        ),
    )
    expected = (streams.float() * mixes[-1].unsqueeze(-1)).sum(dim=1)
    assert carried_mixes[0] is None
    assert carried_mixes[1] is mixes[0]
    torch.testing.assert_close(
        actual, expected.to(streams.dtype), atol=1.6e-2, rtol=1e-2
    )


@pytest.mark.skipif(not HAS_TILELANG_MHC, reason="TileLang MHC support required")
@pytest.mark.parametrize(
    "num_tokens,sinkhorn_repeat,norm_eps",
    [
        (0, 20, 1e-6),
        (1, 1, 1e-6),
        (1, 20, 1e-6),
        (33, 20, 1e-20),
        (128, 20, 1e-6),
        (1024, 20, 1e-6),
    ],
)
@pytest.mark.parametrize("entry", ["broadcast", "identity", "carried"])
@pytest.mark.parametrize("use_deep_gemm", [False, True])
@pytest.mark.parametrize("fused_norm", [False, True])
def test_deepseek_v41_mhc_pre_delayed(
    num_tokens,
    sinkhorn_repeat,
    norm_eps,
    entry,
    use_deep_gemm,
    fused_norm,
    monkeypatch,
):
    """Collapse with the carried pre-mix and save the new one for the next sublayer."""
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if use_deep_gemm and not is_deep_gemm_supported():
        pytest.skip("DeepGEMM not available")
    monkeypatch.setattr(
        "vllm.utils.deep_gemm.is_deep_gemm_supported", lambda: use_deep_gemm
    )
    set_random_seed(0)
    hc_mult, hidden_size = 4, 5120
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=DEVICE
    )
    if num_tokens > 1:
        residual[0].zero_()
    fn = torch.randn(24, hc_mult, hidden_size, device=DEVICE) * 0.02
    x = None
    pre_mix = None
    if entry == "broadcast":
        x = residual[:, 0].contiguous()
        residual = x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()
        fn = fn.sum(1)
    else:
        fn = fn.flatten(1)
        if entry == "carried":
            pre_mix = torch.rand(num_tokens, hc_mult, device=DEVICE)
            if num_tokens > 1:
                # A constant collapse just above 1 rounds to BF16 1 before RMSNorm.
                residual[1].fill_(1)
                pre_mix[1].zero_()
                pre_mix[1, 0] = 1.003
    scale = torch.tensor([0.5, 0.25, 1.0], device=DEVICE)
    base = torch.randn(24, device=DEVICE)
    args = (residual, fn, scale, base, 1e-20, 1e-6, 1e-6, 2.0, sinkhorn_repeat)
    expected = mhc_pre_delayed_torch(*args, pre_mix=pre_mix, x=x)
    norm_kwargs = {}
    if fused_norm:
        weight = torch.empty(hidden_size, dtype=torch.bfloat16, device=DEVICE)
        weight.uniform_(0.5, 1.5)
        norm_kwargs = dict(norm_weight=weight, norm_eps=norm_eps)
        # Compare to the original collapse + CUDA RMSNorm, including BF16 rounding.
        unfused = mhc_pre_delayed_tilelang(*args, pre_mix=pre_mix, x=x)
        normalized = torch.empty_like(unfused[2])
        if num_tokens:
            torch.ops._C.rms_norm(normalized, unfused[2], weight, norm_eps)
        expected = (*unfused[:2], normalized, unfused[3])
    actual = mhc_pre_delayed_tilelang(*args, pre_mix=pre_mix, x=x, **norm_kwargs)
    if fused_norm and num_tokens:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mhc_pre_delayed_tilelang(
                *args, pre_mix=pre_mix, x=x, **norm_kwargs
            )
        graph.replay()
        for eager, replayed in zip(actual, captured, strict=True):
            torch.testing.assert_close(eager, replayed, atol=0, rtol=0)
    if fused_norm and entry == "carried" and num_tokens > 1:
        torch.testing.assert_close(actual[2][1], expected[2][1], atol=0, rtol=0)
    # DeepGEMM uses TF32 for the projection; the TileLang fallback uses FP32.
    atol, rtol = (1e-3, 1e-3) if use_deep_gemm else (1e-5, 1e-4)
    for i in (0, 1, 3):
        torch.testing.assert_close(actual[i], expected[i], atol=atol, rtol=rtol)
    if pre_mix is None and not fused_norm:
        torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual[2], expected[2], atol=1.6e-2, rtol=1e-2)


@pytest.mark.skipif(not HAS_TILELANG_MHC, reason="TileLang MHC support required")
@pytest.mark.parametrize("entry", ["broadcast", "pipeline", "residual", "engram"])
def test_deepseek_v41_decoder_mixes_match_torch(
    entry, monkeypatch, default_vllm_config
):
    """Preserve carried pre-mixes and Engram ordering at every decoder entry."""
    set_random_seed(0)
    decoder = DeepseekV41DecoderLayer.__new__(DeepseekV41DecoderLayer)
    nn.Module.__init__(decoder)
    decoder.hc_mult = 4
    decoder.hc_sinkhorn_iters = 20
    decoder.hc_eps = 1e-6
    decoder.rms_norm_eps = 1e-20
    decoder.hc_post_alpha = 2.0
    decoder.use_sequence_parallel = False
    decoder.engram = None
    from vllm.model_executor.layers.layernorm import RMSNorm

    decoder.attn_norm = decoder.ffn_norm = RMSNorm(5120, 1e-6).to(
        device=DEVICE, dtype=torch.bfloat16
    )
    decoder.attn = lambda positions, x, _: x * 0.5
    decoder.ffn = lambda x, input_ids: x * 0.25
    with torch.device(DEVICE):
        decoder.hc_attn_fn = torch.randn(24, 20480) * 0.02
        decoder.hc_ffn_fn = torch.randn(24, 20480) * 0.02
        decoder.hc_attn_fn_broadcast = decoder.hc_attn_fn.view(24, 4, 5120).sum(1)
        decoder.hc_attn_scale = decoder.hc_ffn_scale = torch.ones(3)
        decoder.hc_attn_base = torch.randn(24)
        decoder.hc_ffn_base = torch.randn(24)
        x = torch.randn(3, 5120, dtype=torch.bfloat16)
        positions = torch.arange(3)
        kwargs = {}
        if entry != "broadcast":
            kwargs["pre_mix"] = torch.rand(3, 4)
        if entry == "pipeline":
            x = torch.randn(3, 4, 5120, dtype=torch.bfloat16)
        if entry in ("residual", "engram"):
            kwargs.update(
                residual=torch.randn(3, 4, 5120, dtype=torch.bfloat16),
                post_mix=torch.rand(3, 4, 1),
                res_mix=torch.rand(3, 4, 4),
            )
        if entry == "engram":

            class FakeEngram(nn.Module):
                layer_hash_index = 0

                def forward(self, residual, hashes, mask):
                    return residual + 0.125

            decoder.engram = FakeEngram()
            kwargs["engram_hashes"] = torch.zeros(3, 1, 1, dtype=torch.int32)

    actual = decoder(x, positions, None, **kwargs)

    def reference(*args, norm_weight, norm_eps, **kwargs):
        post, res, collapsed, pre = mhc_pre_delayed_torch(*args, **kwargs)
        return post, res, decoder.attn_norm(collapsed), pre

    monkeypatch.setattr(
        "vllm.models.deepseek_v4_1.nvidia.model.mhc_pre_delayed_tilelang",
        reference,
    )
    expected = decoder(x, positions, None, **kwargs)
    for result, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(not HAS_TILELANG_MHC, reason="TileLang MHC support required")
@pytest.mark.parametrize("carried", [False, True])
def test_mhc_pre_delayed_custom_op_supports_compile(carried):
    set_random_seed(0)
    x = torch.randn(2, 4, 5120, dtype=torch.bfloat16, device=DEVICE)
    fn = torch.randn(24, 20480, device=DEVICE) * 0.02
    pre_mix = torch.rand(2, 4, device=DEVICE) if carried else None
    scale = torch.ones(3, device=DEVICE)
    base = torch.zeros(24, device=DEVICE)
    torch.library.opcheck(
        torch.ops.vllm.mhc_pre_delayed_tilelang.default,
        (x, fn, scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20, pre_mix),
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
    torch.testing.assert_close(sqrsum, sqrsum_ref, atol=1e-2, rtol=1e-6)


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


def _rocm_mhc_inputs(num_tokens=2, hidden_size=256, hc_mult=4):
    residual = torch.randn(
        (num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16, device=DEVICE
    )
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    fn = (
        torch.randn(
            (hc_mult3, hc_mult * hidden_size), dtype=torch.float32, device=DEVICE
        )
        * 1e-4
    )
    hc_scale = torch.randn((3,), dtype=torch.float32, device=DEVICE) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float32, device=DEVICE) * 0.1
    norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=DEVICE)
    return residual, fn, hc_scale, hc_base, norm_weight


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_mhc_pre_rocm_fallback_applies_norm(monkeypatch):
    set_random_seed(0)
    residual, fn, hc_scale, hc_base, norm_weight = _rocm_mhc_inputs()
    rms_eps = hc_pre_eps = hc_sinkhorn_eps = norm_eps = 1e-6
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
    expected_layer_input = F.rms_norm(
        ref[2], (ref[2].shape[-1],), norm_weight, norm_eps
    )
    monkeypatch.setattr(mhc_layers, "HAS_AITER_MHC", True)
    monkeypatch.setattr(mhc_layers, "HAS_AITER_MHC_PRE_NORM", False)
    monkeypatch.setattr(mhc_layers, "HAS_TILELANG_MHC", False)

    out = object.__new__(MHCPreOp).forward_hip(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    torch.testing.assert_close(out[0], ref[0])
    torch.testing.assert_close(out[1], ref[1])
    torch.testing.assert_close(out[2], expected_layer_input)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_mhc_fused_rocm_fallback_applies_norm(monkeypatch):
    set_random_seed(0)
    residual, fn, hc_scale, hc_base, norm_weight = _rocm_mhc_inputs()
    x = torch.randn((2, 256), dtype=torch.bfloat16, device=DEVICE)
    post_layer_mix = torch.randn((2, 4, 1), dtype=torch.float32, device=DEVICE)
    comb_res_mix = torch.randn((2, 4, 4), dtype=torch.float32, device=DEVICE)
    rms_eps = hc_pre_eps = hc_sinkhorn_eps = norm_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0
    residual_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    pre_ref = mhc_pre_ref(
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
    expected_layer_input = F.rms_norm(
        pre_ref[2], (pre_ref[2].shape[-1],), norm_weight, norm_eps
    )
    monkeypatch.setattr(mhc_layers, "HAS_AITER_MHC_FUSED", False)
    monkeypatch.setattr(mhc_layers, "HAS_TILELANG_MHC", False)

    out = object.__new__(MHCFusedPostPreOp).forward_hip(
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
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    torch.testing.assert_close(out[0], residual_ref)
    torch.testing.assert_close(out[1], pre_ref[0])
    torch.testing.assert_close(out[2], pre_ref[1])
    torch.testing.assert_close(out[3], expected_layer_input)


@pytest.mark.skipif(
    not (current_platform.is_rocm() and HAS_AITER_MHC and HAS_AITER_MHC_PRE_NORM),
    reason="AITER mHC with fused RMSNorm required",
)
def test_mhc_pre_rocm_aiter_fuses_norm():
    set_random_seed(0)
    residual, fn, hc_scale, hc_base, norm_weight = _rocm_mhc_inputs(
        num_tokens=2, hidden_size=7168
    )
    rms_eps = hc_pre_eps = hc_sinkhorn_eps = norm_eps = 1e-6
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
    expected_layer_input = F.rms_norm(
        ref[2], (ref[2].shape[-1],), norm_weight, norm_eps
    )

    out = object.__new__(MHCPreOp).forward_hip(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_alpha,
        sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    torch.testing.assert_close(out[0], ref[0], atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(out[1], ref[1], atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(out[2], expected_layer_input, atol=5e-2, rtol=1e-2)


@pytest.mark.skipif(
    not (
        current_platform.is_rocm() and HAS_AITER_MHC_FUSED and HAS_AITER_MHC_FUSED_NORM
    ),
    reason="AITER fused mHC with RMSNorm required",
)
def test_mhc_fused_rocm_aiter_fuses_norm():
    set_random_seed(0)
    residual, fn, hc_scale, hc_base, norm_weight = _rocm_mhc_inputs(
        num_tokens=2, hidden_size=7168
    )
    x = torch.randn((2, 7168), dtype=torch.bfloat16, device=DEVICE)
    post_layer_mix = torch.randn((2, 4, 1), dtype=torch.float32, device=DEVICE)
    comb_res_mix = torch.randn((2, 4, 4), dtype=torch.float32, device=DEVICE)
    rms_eps = hc_pre_eps = hc_sinkhorn_eps = norm_eps = 1e-6
    sinkhorn_repeat = 20
    hc_post_alpha = 1.0
    residual_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    pre_ref = mhc_pre_ref(
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
    expected_layer_input = F.rms_norm(
        pre_ref[2], (pre_ref[2].shape[-1],), norm_weight, norm_eps
    )

    out = object.__new__(MHCFusedPostPreOp).forward_hip(
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
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )

    torch.testing.assert_close(out[0], residual_ref, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(out[1], pre_ref[0], atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(out[2], pre_ref[1], atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(out[3], expected_layer_input, atol=5e-2, rtol=1e-2)


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


def _make_mhc_decoder_layer(hc_mult: int, hidden_size: int) -> DeepseekV4DecoderLayer:
    layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.hc_mult = hc_mult
    layer.hidden_size = hidden_size
    mix_hc = (2 + hc_mult) * hc_mult
    layer.hc_attn_fn = nn.Parameter(
        torch.randn(mix_hc, hc_mult * hidden_size, dtype=torch.float32),
        requires_grad=False,
    )
    layer.hc_attn_fn_broadcast = None
    return layer


def _patch_first_rank_pp_group(monkeypatch):
    monkeypatch.setattr(
        "vllm.models.deepseek_v4.nvidia.model.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )


def test_deepseek_v4_mhc_broadcast_finalize_sums_hc_streams(monkeypatch):
    """First finalize (at the end of load_weights) allocates
    hc_attn_fn_broadcast as hc_attn_fn summed over hc streams."""
    _patch_first_rank_pp_group(monkeypatch)
    layer = _make_mhc_decoder_layer(hc_mult=2, hidden_size=8)
    model = SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])

    DeepseekV4Model.finalize_mhc_broadcast_weights(model)

    assert layer.hc_attn_fn_broadcast is not None
    expected = layer.hc_attn_fn.detach().view(-1, 2, 8).sum(dim=1)
    assert torch.equal(layer.hc_attn_fn_broadcast, expected)


def test_deepseek_v4_mhc_broadcast_refit_refreshes_in_place(monkeypatch):
    """Re-finalizing after a weight refit must copy into the existing
    broadcast tensor so its address stays stable for captured CUDA graphs,
    while picking up the new hc_attn_fn values."""
    _patch_first_rank_pp_group(monkeypatch)
    layer = _make_mhc_decoder_layer(hc_mult=2, hidden_size=8)
    model = SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])

    DeepseekV4Model.finalize_mhc_broadcast_weights(model)
    buffer = layer.hc_attn_fn_broadcast

    layer.hc_attn_fn.add_(1.0)
    DeepseekV4Model.finalize_mhc_broadcast_weights(model)

    assert layer.hc_attn_fn_broadcast is buffer
    expected = layer.hc_attn_fn.detach().view(-1, 2, 8).sum(dim=1)
    assert torch.equal(layer.hc_attn_fn_broadcast, expected)
