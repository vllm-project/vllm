# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 DeepGEMM FP8 activation chain.

Covers the pieces that let activations stay MXFP8 between DeepGEMM kernels:
the 2D DeepGEMM MXFP8 linear kernel (plain and pre-quantized inputs), Mega
mHC's FP8 outputs (GEMM layout and Mega-MoE symmetric-buffer layout), the
wo_a einsum's FP8 output, and the skip-quant MoE input staging.
"""

import pytest
import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8DynamicDeepGemm,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="DeepGEMM MXFP8 chain is SM100 only",
)


def _unpack_ue8m0(packed: torch.Tensor, num_groups: int) -> torch.Tensor:
    """[T, ceil(G/4)] int32 packed (4 UE8M0 per word) -> [T, G] float32 scales."""
    words = packed.to(torch.int64)
    exps = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], dim=-1)
    exps = exps.reshape(packed.shape[0], -1)[:, :num_groups]
    return torch.exp2(exps.float() - 127.0)


def _dequant_packed(x_fp8: torch.Tensor, sf_packed: torch.Tensor) -> torch.Tensor:
    tokens, hidden = x_fp8.shape
    scales = _unpack_ue8m0(sf_packed, hidden // 32)
    return (x_fp8.float().view(tokens, hidden // 32, 32) * scales[:, :, None]).view(
        tokens, hidden
    )


def _quantize_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference MXFP8 (UE8M0 ceil, e4m3) dequantized back to fp32."""
    tokens, hidden = x.shape
    groups = x.float().view(tokens, hidden // 32, 32)
    amax = groups.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (groups / scale).to(torch.float8_e4m3fn).float() * scale
    return q.view(tokens, hidden)


@pytest.mark.parametrize("tokens", [1, 4, 7, 64, 300])
@pytest.mark.parametrize("n_k", [(1792, 5120), (5120, 2048)])
def test_deepgemm_mxfp8_linear_matches_flashinfer(tokens: int, n_k: tuple[int, int]):
    """DeepGEMM 2D MXFP8 GEMM agrees with the FlashInfer CuTe-DSL kernel."""
    from vllm.model_executor.kernels.linear.mxfp8.deep_gemm import (
        DeepGemmMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    for cls in (DeepGemmMxfp8LinearKernel, FlashInferCutedslMxfp8LinearKernel):
        ok, reason = cls.is_supported()
        if not ok:
            pytest.skip(reason)
    torch.manual_seed(0)
    N, K = n_k
    w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    wq, ws = mxfp8_e4m3_quantize(w_bf16)  # [N, K] e4m3, [N, K/32] uint8
    x = torch.randn(tokens, K, device="cuda", dtype=torch.bfloat16)

    def make_layer(cls):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(wq.clone(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(ws.clone(), requires_grad=False)
        kernel = cls(Mxfp8LinearLayerConfig())
        kernel.process_weights_after_loading(layer)
        return layer, kernel

    ref_layer, ref_kernel = make_layer(FlashInferCutedslMxfp8LinearKernel)
    dg_layer, dg_kernel = make_layer(DeepGemmMxfp8LinearKernel)
    ref = ref_kernel.apply_weights(ref_layer, x)
    out = dg_kernel.apply_weights(dg_layer, x)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-1)

    # Pre-quantized input in DeepGEMM's packed layout bypasses the quantize.
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )

    xq, xs = per_token_group_quant_fp8_packed_for_deepgemm(x, 32, use_ue8m0=True)
    qa = QuantizedActivation(xq, xs, x.dtype, x.shape, kMxfp8DynamicDeepGemm)
    out_q = dg_kernel.apply_weights(dg_layer, qa)
    torch.testing.assert_close(out_q, out, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 3, 16, 129])
def test_mega_mhc_fp8_gemm_output(tokens: int):
    """Mega mHC's y_fp8/y_gemm_sf dequantizes to its own y_bf16 (MXFP8 rounding)."""
    from vllm.models.deepseek_v41.nvidia.ops.mega_mhc import (
        is_mega_mhc_supported,
        mhc_shifted_post_pre_deep_gemm,
    )

    hidden, hc_mult = 5120, 4
    if not is_mega_mhc_supported(hidden, hc_mult):
        pytest.skip("Mega mHC unsupported")
    torch.manual_seed(0)
    dev = "cuda"
    mix_hc = (2 + hc_mult) * hc_mult
    x = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16)
    residual = torch.randn(tokens, hc_mult, hidden, device=dev, dtype=torch.bfloat16)
    prev_mix = torch.rand(tokens, hc_mult, device=dev, dtype=torch.float32)
    post_mix = torch.rand(tokens, hc_mult, 1, device=dev, dtype=torch.float32)
    comb = torch.rand(tokens, hc_mult, hc_mult, device=dev, dtype=torch.float32)
    fn = torch.randn(mix_hc, hc_mult * hidden, device=dev) * 0.02
    mix_scales = torch.tensor([0.5, 0.25, 1.0], device=dev)
    mix_bases = torch.randn(mix_hc, device=dev)
    w = torch.empty(hidden, device=dev, dtype=torch.bfloat16).uniform_(0.5, 1.5)
    args = (
        x,
        residual,
        prev_mix,
        post_mix,
        comb,
        fn,
        mix_scales,
        mix_bases,
        1e-20,
        1e-6,
        2.0,
        1e-6,
        20,
        w,
        1e-6,
    )
    *ref, none = mhc_shifted_post_pre_deep_gemm(*args)
    assert none == (None, None)
    *out, fp8 = mhc_shifted_post_pre_deep_gemm(*args, fp8_gemm_output=True)
    y_q = fp8.gemm_input
    assert isinstance(y_q, QuantizedActivation) and fp8.moe_staged is None
    assert y_q.quant_key == kMxfp8DynamicDeepGemm
    for a, b in zip(ref, out):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    y_bf16 = out[3]
    deq = _dequant_packed(y_q.data, y_q.scale)
    torch.testing.assert_close(deq, _quantize_ref(y_bf16), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 5, 64])
def test_fp8_einsum_fp8_output(tokens: int):
    """fp8_einsum's (z, sf) output equals quantizing its bf16 output."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8_packed_for_deepgemm,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import alloc_fp8_einsum_output
    from vllm.utils.deep_gemm import (
        fp8_einsum,
        get_tma_aligned_size,
        is_deep_gemm_supported,
    )

    if not is_deep_gemm_supported():
        pytest.skip("DeepGEMM unsupported")
    torch.manual_seed(0)
    groups, r, d = 2, 4096, 1024  # bhr,hdr->bhd; r = heads_per_group * head_dim
    dev = "cuda"
    a = torch.randn(tokens, groups, r, device=dev, dtype=torch.bfloat16)
    b = torch.randn(groups * d, r, device=dev, dtype=torch.bfloat16)
    # A in the mega-attention output layout: [T, groups, r/128] int32 packed
    # scales, MN-major per group (strides (1, r/128 * aligned, aligned)).
    aligned = get_tma_aligned_size(tokens, torch.int32.itemsize)
    aq = torch.empty(tokens, groups, r, device=dev, dtype=torch.float8_e4m3fn)
    asf = torch.empty((groups, r // 128, aligned), device=dev, dtype=torch.int32)
    asf = asf.permute(2, 0, 1)[:tokens]
    for g in range(groups):
        q, sf = per_token_group_quant_fp8_packed_for_deepgemm(
            a[:, g].contiguous(), 32, use_ue8m0=True
        )
        aq[:, g] = q
        asf[:, g] = sf
    bq, bs = mxfp8_e4m3_quantize(b)
    bq, bsf = deepgemm_post_process_fp8_weight_block(
        wq=bq,
        ws=bs,
        quant_block_shape=(1, 32),
        use_e8m0=False,
        is_bmm=True,
        bmm_batch_size=groups,
    )
    z_bf16 = torch.empty(tokens, groups, d, device=dev, dtype=torch.bfloat16)
    fp8_einsum("bhr,hdr->bhd", (aq, asf), (bq, bsf), z_bf16, recipe=(1, 1, 32))
    z_fp8, z_sf = alloc_fp8_einsum_output(tokens, groups, d, torch.device(dev))
    fp8_einsum("bhr,hdr->bhd", (aq, asf), (bq, bsf), (z_fp8, z_sf), recipe=(1, 1, 32))
    deq = _dequant_packed(z_fp8.flatten(1), z_sf)
    torch.testing.assert_close(
        deq, _quantize_ref(z_bf16.flatten(1)), rtol=1e-2, atol=1e-2
    )


def test_stage_megamoe_routing():
    """Routing-only staging leaves the FP8 buffers untouched and repacks the top-k."""
    from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
        prepare_megamoe_inputs,
        stage_megamoe_routing,
    )

    torch.manual_seed(0)
    dev, tokens, hidden, top_k = "cuda", 9, 5120, 6
    h = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, 384, (tokens, top_k), device=dev, dtype=torch.int32)
    wts = torch.rand(tokens, top_k, device=dev, dtype=torch.float32)
    x = torch.zeros(tokens, hidden, device=dev, dtype=torch.float8_e4m3fn)
    x_sf = torch.zeros(tokens, hidden // 128, device=dev, dtype=torch.int32)
    idx_out = torch.empty(tokens, top_k, device=dev, dtype=torch.int64)
    w_out = torch.empty(tokens, top_k, device=dev, dtype=torch.float32)
    stage_megamoe_routing(wts, ids, idx_out, w_out)
    assert x.float().abs().sum() == 0 and x_sf.abs().sum() == 0
    torch.testing.assert_close(idx_out, ids.to(torch.int64))
    torch.testing.assert_close(w_out, wts)
    # And the full path still quantizes.
    prepare_megamoe_inputs(h, wts, ids, x, x_sf, idx_out, w_out)
    torch.testing.assert_close(
        _dequant_packed(x, x_sf), _quantize_ref(h), rtol=0, atol=0
    )


@pytest.mark.parametrize("tokens,shared_block_m", [(1, 128), (7, 128), (200, 64)])
def test_mega_mhc_fp8_moe_output(tokens: int, shared_block_m: int):
    """Mega mHC's MoE-layout FP8 output equals prepare_megamoe_inputs' quantization."""
    from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
        MegaMoeFp8Target,
        prepare_megamoe_inputs,
    )
    from vllm.models.deepseek_v41.nvidia.ops.mega_mhc import (
        is_mega_mhc_supported,
        mhc_shifted_post_pre_deep_gemm,
    )

    hidden, hc_mult, top_k, max_tokens = 5120, 4, 6, 256
    if not is_mega_mhc_supported(hidden, hc_mult):
        pytest.skip("Mega mHC unsupported")
    torch.manual_seed(0)
    dev = "cuda"
    mix_hc = (2 + hc_mult) * hc_mult
    x = torch.randn(tokens, hidden, device=dev, dtype=torch.bfloat16)
    residual = torch.randn(tokens, hc_mult, hidden, device=dev, dtype=torch.bfloat16)
    prev_mix = torch.rand(tokens, hc_mult, device=dev, dtype=torch.float32)
    post_mix = torch.rand(tokens, hc_mult, 1, device=dev, dtype=torch.float32)
    comb = torch.rand(tokens, hc_mult, hc_mult, device=dev, dtype=torch.float32)
    fn = torch.randn(mix_hc, hc_mult * hidden, device=dev) * 0.02
    mix_scales = torch.tensor([0.5, 0.25, 1.0], device=dev)
    mix_bases = torch.randn(mix_hc, device=dev)
    w = torch.empty(hidden, device=dev, dtype=torch.bfloat16).uniform_(0.5, 1.5)
    args = (
        x,
        residual,
        prev_mix,
        post_mix,
        comb,
        fn,
        mix_scales,
        mix_bases,
        1e-20,
        1e-6,
        2.0,
        1e-6,
        20,
        w,
        1e-6,
    )
    # Symmetric-buffer-shaped target: routed x / x_sf row-major, shared sf
    # MN-major with the DeepGEMM shared-expert row count.
    aligned_block_m = -(-shared_block_m // 128) * 128
    shared_rows = -(-max_tokens // shared_block_m) * aligned_block_m
    mk = lambda: MegaMoeFp8Target(  # noqa: E731
        x=torch.zeros(max_tokens, hidden, device=dev, dtype=torch.float8_e4m3fn),
        x_sf=torch.zeros(max_tokens, hidden // 128, device=dev, dtype=torch.int32),
        shared_sf=torch.zeros(
            (hidden // 128, shared_rows), device=dev, dtype=torch.int32
        ).t(),
        shared_block_m=shared_block_m,
    )
    target = mk()
    *out, fp8 = mhc_shifted_post_pre_deep_gemm(*args, moe_target=target)
    assert fp8.moe_staged is target and fp8.gemm_input is None
    y_bf16 = out[3]
    # Reference: the staging kernel's own quantization of the same y_bf16.
    ref = mk()
    ids = torch.randint(0, 384, (tokens, top_k), device=dev, dtype=torch.int32)
    wts = torch.rand(tokens, top_k, device=dev, dtype=torch.float32)
    prepare_megamoe_inputs(
        y_bf16,
        wts,
        ids,
        ref.x[:tokens],
        ref.x_sf[:tokens],
        torch.empty(tokens, top_k, device=dev, dtype=torch.int64),
        torch.empty(tokens, top_k, device=dev, dtype=torch.float32),
        shared_x_sf=ref.shared_sf,
        shared_block_m=shared_block_m,
    )
    torch.testing.assert_close(
        target.x[:tokens].float(), ref.x[:tokens].float(), rtol=0, atol=0
    )
    torch.testing.assert_close(target.x_sf[:tokens], ref.x_sf[:tokens], rtol=0, atol=0)
    torch.testing.assert_close(target.shared_sf, ref.shared_sf, rtol=0, atol=0)
