# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric tests for the MuseSpark1x BF16/NVFP4 MoE on TRT-LLM and MegaMoE."""

import functools

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.muse_spark_utils import (
    DIVISION_EPS,
    assert_cudagraph_replays,
    make_runner,
    nvfp4_quant_config,
    run_moe,
    swiglu_moe_ref,
)
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.models.muse_spark.nvidia.mega_moe import MuseSparkMegaMoEExperts
from vllm.models.muse_spark.nvidia.muse_moe_runner import MuseSparkTrtLlmNvFp4Experts
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

if not (
    current_platform.has_device_capability(100) and has_flashinfer_trtllm_fused_moe()
):
    pytest.skip(
        "Requires the FlashInfer TRT-LLM fused MoE (SM100+)",
        allow_module_level=True,
    )

# The smallest shapes the TRTLLM-Gen batched GEMM accepts.
E, N, K, TOPK, M = 32, 1024, 2048, 8, 64
FP4_SF_MAX = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX


def build_moe(norm_cls, ckpt: dict[str, torch.Tensor], quant_config=None):
    # Non-zero so selecting on biased scores and normalizing unbiased ones is
    # distinguishable from using one set for both.
    expert_bias = (torch.rand(E, device="cuda", dtype=torch.float32) - 0.5) * 0.4
    norm = norm_cls(
        hidden_size=K,
        eps=1e-6,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    with torch.no_grad():
        norm.weight.copy_(
            torch.randn(K, device="cuda", dtype=torch.float32) * 0.1 - 0.2
        )
    router, runner, vllm_config = make_runner(
        norm,
        expert_bias,
        ckpt,
        quant_config,
        hidden_size=K,
        intermediate_size=N,
        top_k=TOPK,
    )
    assert router.routing_method_type == RoutingMethodType.MiniMax2
    routed = runner.routed_experts
    assert routed.quant_method.is_monolithic
    return router, routed, norm, vllm_config, runner


def _inputs(seed: int):
    torch.manual_seed(seed)
    return (
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 10,
        torch.randn(M, E, device="cuda", dtype=torch.float32),
    )


def _bf16_checkpoint(seed: int = 0):
    torch.manual_seed(seed)
    return {
        "w13_weight": torch.randn(E, 2 * N, K, device="cuda", dtype=torch.bfloat16)
        / 15,
        "w2_weight": torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) / 15,
    }


def _quant_side(w: torch.Tensor, global_scale: torch.Tensor):
    """Quantize [rows, cols] to (packed fp4 uint8, linear fp8 block scales)."""
    q, sf = ops.scaled_fp4_quant(
        w.to(torch.bfloat16),
        global_scale.to(torch.float32).reshape(1),
        is_sf_swizzled_layout=False,
    )
    assert q.shape == (w.shape[0], w.shape[1] // 2)
    assert sf.shape == (w.shape[0], w.shape[1] // 16)
    return q, sf


def _dequant(q, sf, gs):
    return dequantize_nvfp4_to_dtype(
        q,
        sf,
        gs,
        dtype=torch.float32,
        device="cuda",
        block_size=16,
        is_sf_128x4_layout=False,
        is_sf_linear_layout=True,
    )


def _nvfp4_checkpoint(seed: int = 0):
    """A checkpoint-layout NVFP4 MoE layer with distinct w1/w3 scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randn(E, 2 * N, K, device="cuda", dtype=torch.bfloat16) / 15
    w2 = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) / 15

    # modelopt convention: dequant scale s = amax / (448 * 6), quantize with 1/s.
    w13_s2 = torch.empty(E, 2, device="cuda", dtype=torch.float32)
    w13_q = torch.empty(E, 2 * N, K // 2, device="cuda", dtype=torch.uint8)
    w13_sf = torch.empty(E, 2 * N, K // 16, device="cuda", dtype=torch.uint8)
    for e in range(E):
        for col, rows in ((0, w13[e, :N]), (1, w13[e, N:])):
            jitter = 1.0 + 0.4 * torch.rand((), generator=g, device="cuda")
            s = rows.abs().max() / FP4_SF_MAX * jitter
            w13_s2[e, col] = s
            q, sf = _quant_side(rows.float(), 1.0 / s)
            w13_q[e, col * N : (col + 1) * N] = q
            w13_sf[e, col * N : (col + 1) * N] = sf.view(torch.uint8)

    w2_s2 = torch.empty(E, device="cuda", dtype=torch.float32)
    w2_q = torch.empty(E, K, N // 2, device="cuda", dtype=torch.uint8)
    w2_sf = torch.empty(E, K, N // 16, device="cuda", dtype=torch.uint8)
    for e in range(E):
        s = w2[e].abs().max() / FP4_SF_MAX
        w2_s2[e] = s
        q, sf = _quant_side(w2[e].float(), 1.0 / s)
        w2_q[e] = q
        w2_sf[e] = sf.view(torch.uint8)

    return {
        "w13_weight": w13_q,
        "w13_weight_scale": w13_sf.view(torch.float8_e4m3fn),
        "w13_weight_scale_2": w13_s2,
        "w13_input_scale": torch.rand(E, 2, generator=g, device="cuda") + 0.5,
        "w2_weight": w2_q,
        "w2_weight_scale": w2_sf.view(torch.float8_e4m3fn),
        "w2_weight_scale_2": w2_s2,
        "w2_input_scale": torch.rand(E, generator=g, device="cuda") + 0.5,
    }


def _dequant_nvfp4_weights(ckpt: dict[str, torch.Tensor]):
    w13 = torch.empty(E, 2 * N, K, device="cuda", dtype=torch.float32)
    w2 = torch.empty(E, K, N, device="cuda", dtype=torch.float32)
    for e in range(E):
        for col, sl in ((0, slice(0, N)), (1, slice(N, 2 * N))):
            w13[e, sl] = _dequant(
                ckpt["w13_weight"][e, sl],
                ckpt["w13_weight_scale"][e, sl],
                1.0 / ckpt["w13_weight_scale_2"][e, col],
            )
        w2[e] = _dequant(
            ckpt["w2_weight"][e],
            ckpt["w2_weight_scale"][e],
            1.0 / ckpt["w2_weight_scale_2"][e],
        )
    return w13, w2


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("tokens", [0, 17])
@torch.inference_mode()
def test_megamoe_bf16_nvfp4_shared_buffer(
    tokens, default_vllm_config, dist_init, workspace_init, monkeypatch
):
    """Alternating formats must preserve BF16 results, also in CUDA graphs."""
    pytest.importorskip("deep_gemm")
    config = VllmConfig(parallel_config=ParallelConfig())
    monkeypatch.setattr(
        "vllm.models.muse_spark.nvidia.mega_moe.get_ep_group", get_tp_group
    )
    torch.manual_seed(42)
    experts, hidden, intermediate, topk = 8, 4096, 1024, 4
    norm = MuseSpark1xRMSNormWithInputScale(
        hidden_size=hidden,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    norm.weight.copy_(torch.linspace(-0.5, 0.5, hidden, device="cuda"))
    layers = []
    with set_current_vllm_config(config):
        for nvfp4 in (False, True):
            layer = MuseSparkMegaMoEExperts(
                config,
                num_experts=experts,
                num_local_experts=experts,
                experts_start_idx=0,
                top_k=topk,
                hidden_size=hidden,
                intermediate_size=intermediate,
                post_expert_norm=norm,
                num_max_tokens_per_rank=128,
                is_nvfp4=nvfp4,
                prefix=f"test.{nvfp4}",
            ).cuda()
            if nvfp4:
                # Zero packed weights and unit scales give an exact zero result.
                for name, parameter in layer.named_parameters():
                    if "scale" in name:
                        parameter.fill_(1)
            else:
                w1 = torch.randn_like(layer.w13_weight) * 0.1
                w2 = torch.randn_like(layer.w2_weight) * 1e-5
                assert set(
                    layer.load_weights([("w13_weight", w1), ("w2_weight", w2)])
                ) == {"w13_weight", "w2_weight"}
            layer.finalize_weights()
            layers.append(layer)
        bf16, nvfp4 = layers
        bf16_buffer, fp4_buffer = [layer.get_symm_buffer() for layer in layers]
        assert bf16_buffer.buffer.data_ptr() == fp4_buffer.buffer.data_ptr()
        assert bf16_buffer.handle is fp4_buffer.handle
        x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
        weights, ids = torch.randn(tokens, experts, device="cuda").topk(topk)
        weights = weights.softmax(-1)
        ids = ids.int()
        ids[::3, -1] = -1
        expected = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(experts):
            token, slot = (ids == expert).nonzero(as_tuple=True)
            gate, up = (x[token].float() @ w1[expert].float().T).bfloat16().chunk(2, -1)
            act = (F.silu(gate.float()) * up.float()).bfloat16()
            rows = (act.float() @ w2[expert].float().T).bfloat16()
            scaled = rows.float() * bf16._norm_gamma
            result = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + 1e-8)
            expected.index_add_(0, token, result * weights[token, slot, None])
        for _ in range(2):
            torch.testing.assert_close(nvfp4(x, weights, ids), torch.zeros_like(x))
            actual = bf16(x, weights, ids)
            torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fp4_actual = nvfp4(x, weights, ids)
            actual = bf16(x, weights, ids)
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(fp4_actual, torch.zeros_like(x))
            torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.02)


# At hidden 4096 the block size switches at 512K and 2M elements: 128/129 and
# 512 reach all three.
@pytest.mark.parametrize("num_tokens", [1, 128, 129, 512])
def test_megamoe_staging_preserves_quantization_and_padding(num_tokens):
    from vllm.models.muse_spark.nvidia.ops.prepare_megamoe_nvfp4 import (
        prepare_megamoe_nvfp4_inputs,
    )

    hidden_size = 4096
    grid = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.bfloat16,
    )
    scales = (
        2.0 ** (torch.arange(num_tokens * hidden_size // 16, device="cuda") % 3 - 1)
    ).view(num_tokens, hidden_size // 16)
    hidden = (scales[:, :, None] * grid).reshape(num_tokens, hidden_size).bfloat16()
    ids = torch.randint(256, (num_tokens, 8), device="cuda", dtype=torch.int32)
    weights = torch.rand(num_tokens, 8, device="cuda")
    padding = torch.arange(num_tokens, device="cuda") % 3 == 0
    expected_packed = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE],
        device="cuda",
        dtype=torch.uint8,
    ).repeat(num_tokens, hidden_size // 16)
    expected_sf = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    packed = torch.empty_like(expected_packed)
    sf = torch.empty_like(expected_sf, dtype=torch.uint8)
    staged_ids = torch.empty_like(ids, dtype=torch.int64)
    staged_weights = torch.empty_like(weights)

    prepare_megamoe_nvfp4_inputs(
        hidden, weights, ids, packed, sf, staged_ids, staged_weights, padding
    )

    torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
    torch.testing.assert_close(
        sf.view(torch.uint8), expected_sf.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(staged_ids, ids.long().masked_fill(padding[:, None], -1))
    torch.testing.assert_close(staged_weights, weights.masked_fill(padding[:, None], 0))


@torch.inference_mode()
def test_muse_spark_bf16_moe_numeric(dist_init, workspace_init):
    """The norm sits between GEMM2 and the reduce, matching the fp32 reference."""
    ckpt = _bf16_checkpoint()
    router, _, norm, vllm_config, runner = build_moe(MuseSpark1xRMSNorm, ckpt)

    x, logits = _inputs(1)
    out = run_moe(runner, vllm_config, x, logits)
    assert out.shape == (M, K) and out.dtype == torch.bfloat16

    topk_weights, topk_ids = router._compute_routing(x, logits, None)
    w13, w2 = ckpt["w13_weight"], ckpt["w2_weight"]
    with set_current_vllm_config(vllm_config):
        ref = swiglu_moe_ref(x, w13, w2, topk_weights, topk_ids, norm.forward_native)
        unnormed = swiglu_moe_ref(x, w13, w2, topk_weights, topk_ids, lambda y: y)
        post_reduce = norm.forward_native(unnormed.to(torch.bfloat16)).float()
    # Only bf16 GEMM/accumulation-order error separates the two.
    assert _rel(out, ref) < 2e-2
    assert _rel(out, unnormed) > 1e-1
    assert _rel(out, post_reduce) > 1e-1


@torch.inference_mode()
def test_muse_spark_nvfp4_moe_numeric(dist_init, workspace_init):
    torch.manual_seed(0)
    ckpt = _nvfp4_checkpoint()
    router, routed, norm, vllm_config, runner = build_moe(
        MuseSpark1xRMSNormWithInputScale, ckpt, nvfp4_quant_config()
    )

    # Gate scalar from the w1 column, g1_scale_c from the w3 column.
    a13 = ckpt["w13_input_scale"].max()
    a2 = ckpt["w2_input_scale"].max()
    experts = routed.quant_method.moe_kernel.fused_experts
    assert isinstance(experts, MuseSparkTrtLlmNvFp4Experts)
    qc = experts.quant_config
    torch.testing.assert_close(qc.g1_alphas, a13 * ckpt["w13_weight_scale_2"][:, 0])
    torch.testing.assert_close(
        experts.g1_scale_c, a13 * ckpt["w13_weight_scale_2"][:, 1] / a2
    )
    torch.testing.assert_close(qc.g2_alphas, a2 * ckpt["w2_weight_scale_2"])
    torch.testing.assert_close(qc.a1_gscale, 1.0 / a13.reshape(1))

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 10
    logits = torch.randn(M, E, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = router._compute_routing(x, logits, None)
    assert topk_weights.dtype == torch.float32

    out = run_moe(runner, vllm_config, x, logits)
    assert out.shape == (M, K) and out.dtype == torch.bfloat16

    w13, w2 = _dequant_nvfp4_weights(ckpt)
    xq, xsf = _quant_side(x.float(), 1.0 / a13)
    x_dq = _dequant(xq, xsf, 1.0 / a13)

    def quant_gemm2_input(inter: torch.Tensor) -> torch.Tensor:
        iq, isf = _quant_side(inter, 1.0 / a2)
        return _dequant(iq, isf, 1.0 / a2)

    norm_fn = functools.partial(norm.forward_native, out_dtype=torch.float32)
    with set_current_vllm_config(vllm_config):
        ref_full = swiglu_moe_ref(x_dq, w13, w2, topk_weights, topk_ids, norm_fn)
        ref_q2 = swiglu_moe_ref(
            x_dq, w13, w2, topk_weights, topk_ids, norm_fn, quant_gemm2_input
        )
    # The kernel's internal GEMM2-input NVFP4 quantizer is only approximated by
    # the reference. Measured: rel_q2 ~2.3e-2, rel_full ~1.0e-1.
    assert _rel(out, ref_q2) < 5e-2
    assert _rel(out, ref_full) < 1.5e-1


@torch.inference_mode()
def test_muse_spark_bf16_moe_kernel_router_matches_reference(dist_init, workspace_init):
    """The fused router must select on biased scores but weight by unbiased ones."""
    ckpt = _bf16_checkpoint(seed=3)
    router, routed, _, vllm_config, runner = build_moe(MuseSpark1xRMSNorm, ckpt)

    captured: list[UnfinalizedMoEOutput] = []
    experts = routed.quant_method.moe_kernel.fused_experts
    inner = experts.post_gemm2_hook

    def _spy(unfinalized, out):
        captured.append(unfinalized)
        inner(unfinalized, out)

    experts.post_gemm2_hook = _spy
    try:
        x, logits = _inputs(3)
        run_moe(runner, vllm_config, x, logits)
    finally:
        experts.post_gemm2_hook = inner

    # Otherwise the kernel did not defer and the norm was silently skipped.
    assert len(captured) == 1
    kernel_weights = captured[0].expert_weights.float()

    scores = torch.sigmoid(logits.float())
    biased = scores + router.expert_bias
    ids = torch.topk(biased, TOPK, dim=-1).indices
    unbiased_norm = torch.gather(scores, 1, ids)
    unbiased_norm = unbiased_norm / (unbiased_norm.sum(-1, keepdim=True) + DIVISION_EPS)
    biased_norm = torch.gather(biased, 1, ids)
    biased_norm = biased_norm / (biased_norm.sum(-1, keepdim=True) + DIVISION_EPS)

    got = kernel_weights.sort(dim=-1).values
    want = unbiased_norm.sort(dim=-1).values
    wrong = biased_norm.sort(dim=-1).values
    assert (want - wrong).abs().max() > 1e-2
    assert (got - want).abs().max() < 5e-3
    assert (got - wrong).abs().max() > 5e-3


@pytest.mark.parametrize("nvfp4", [False, True])
@torch.inference_mode()
def test_muse_spark_moe_cudagraph(nvfp4, dist_init, workspace_init):
    """The deferred tail is stream-capturable."""
    torch.manual_seed(0)
    if nvfp4:
        ckpt, quant_config = _nvfp4_checkpoint(), nvfp4_quant_config()
    else:
        ckpt, quant_config = _bf16_checkpoint(), None
    _, _, _, vllm_config, runner = build_moe(MuseSpark1xRMSNorm, ckpt, quant_config)
    assert_cudagraph_replays(runner, vllm_config, *_inputs(5), *_inputs(6))
