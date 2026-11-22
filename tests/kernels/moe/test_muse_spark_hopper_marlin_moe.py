# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper coverage for MuseSpark's NVFP4 (Hopper and Marlin) and BF16 Triton experts."""

import gc
import hashlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.quantization.modelopt as modelopt_mod
import vllm.utils.gpu_sync_debug as gpu_sync_debug
from tests.kernels.moe.muse_spark_utils import (
    assert_cudagraph_replays,
    make_runner,
    nvfp4_quant_config,
    run_moe,
    swiglu_moe_ref,
)
from vllm.config import set_current_vllm_config
from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingGroupedExperts,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    finish_reload,
    initialize_layerwise_reload,
    load_planned,
    plan_loads,
    start_reload,
)
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    record_checkpoint_layout,
    track_loaded_params,
)
from vllm.models.muse_spark.common.layers import (
    MuseSparkRMSNorm,
    MuseSparkRMSNormWithInputScale,
)
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MusePostExpertFinalize,
    MuseSparkModelOptNvFp4FusedMoE,
)
from vllm.models.muse_spark.nvidia.ops.hopper_nvfp4_moe import (
    MuseSparkHopperNvFp4Experts,
    encode_hopper_nvfp4_weights,
    is_hopper_nvfp4_moe_supported,
)
from vllm.models.muse_spark.nvidia.ops.hopper_nvfp4_moe import (
    kernels as hopper_nvfp4_kernels,
)
from vllm.platforms import current_platform

_requires_sm90_marlin = pytest.mark.skipif(
    not (
        current_platform.is_device_capability_family(90) and is_fp4_marlin_supported()
    ),
    reason="Requires an SM90 GPU with Marlin",
)

E, N, K, TOPK, M = 8, 256, 512, 2, 8
HOPPER_E, HOPPER_N, HOPPER_K, HOPPER_TOPK = 32, 4096, 4096, 8

_E2M1_TABLE_CPU = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _free_cuda() -> None:
    gc.collect()
    torch.accelerator.empty_cache()


def _build(
    checkpoint: dict[str, torch.Tensor],
    quant_config=None,
    *,
    num_experts: int = E,
    hidden_size: int = K,
    intermediate_size: int = N,
    top_k: int = TOPK,
    moe_backend: str = "auto",
    post_eps: float = 1e-8,
):
    norm = MuseSparkRMSNormWithInputScale(
        hidden_size,
        eps=post_eps,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    norm.weight.copy_(torch.linspace(-0.2, 0.2, hidden_size, device="cuda"))
    expert_bias = torch.linspace(
        -0.5, 0.5, num_experts, dtype=torch.float32, device="cuda"
    )
    with torch.device("cuda"):
        router, runner, vllm_config = make_runner(
            norm,
            expert_bias,
            checkpoint,
            quant_config,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=top_k,
            moe_backend=moe_backend,
        )
    experts = runner.routed_experts.quant_method.moe_kernel.fused_experts
    return router, norm, runner, vllm_config, experts


def _build_hopper(
    checkpoint: dict[str, torch.Tensor],
    *,
    moe_backend: str = "auto",
    post_eps: float = 1e-8,
):
    return _build(
        checkpoint,
        nvfp4_quant_config(),
        num_experts=HOPPER_E,
        hidden_size=HOPPER_K,
        intermediate_size=HOPPER_N,
        top_k=HOPPER_TOPK,
        moe_backend=moe_backend,
        post_eps=post_eps,
    )


def _nvfp4_checkpoint() -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(0)

    def packed(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, dtype=torch.float32, device="cuda", generator=g)

    return {
        "w13_weight": packed(E, 2 * N, K // 2),
        "w13_weight_scale": torch.ones(
            E, 2 * N, K // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        "w13_weight_scale_2": rand(E, 2) * 0.01 + 0.01,
        "w13_input_scale": rand(E, 2) + 0.5,
        "w2_weight": packed(E, K, N // 2),
        "w2_weight_scale": torch.ones(
            E, K, N // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        "w2_weight_scale_2": rand(E) * 0.01 + 0.01,
        "w2_input_scale": rand(E) + 0.5,
    }


def _dequant(q, scale, global_scale, dtype: torch.dtype = torch.bfloat16):
    return dequantize_to_dtype(
        q, scale, global_scale, dtype, block_size=16, swizzle=False
    )


def _expected(router, norm, hidden_states, router_logits, w13, w2):
    topk_weights, topk_ids = router._compute_routing(hidden_states, router_logits, None)
    return swiglu_moe_ref(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        lambda y: norm.forward_native(y, out_dtype=torch.float32),
    )


def _rand_inputs(
    num_tokens: int,
    g: torch.Generator,
    *,
    hidden_size: int = HOPPER_K,
    num_experts: int = HOPPER_E,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = (
        torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda", generator=g
        )
        / 10
    )
    logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    return x, logits


def _run_checked(runner, vllm_config, x: torch.Tensor, logits: torch.Tensor):
    out = run_moe(runner, vllm_config, x, logits)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()
    return out


def _ref_for(ckpt, router, norm, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    topk_weights, topk_ids = router._compute_routing(x, logits, None)
    return _nvfp4_swiglu_moe_ref(ckpt, norm, x, topk_weights, topk_ids)


def _apply_experts(
    experts: MuseSparkHopperNvFp4Experts,
    routed: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    fill_value: float = 0.0,
    noncontiguous_out: bool = False,
) -> torch.Tensor:
    ws1_shape, ws2_shape, out_shape = experts.workspace_shapes(
        x.shape[0],
        HOPPER_N,
        HOPPER_K,
        HOPPER_TOPK,
        HOPPER_E,
        HOPPER_E,
        None,
        MoEActivation.SILU,
    )
    ws1 = torch.full(ws1_shape, fill_value, dtype=torch.bfloat16, device="cuda")
    ws2 = torch.full(ws2_shape, fill_value, dtype=torch.bfloat16, device="cuda")
    if noncontiguous_out:
        out = torch.full(
            out_shape[::-1], fill_value, dtype=torch.bfloat16, device="cuda"
        ).t()
        assert not out.is_contiguous()
    else:
        out = torch.full(out_shape, fill_value, dtype=torch.bfloat16, device="cuda")
    experts.apply(
        output=out,
        hidden_states=x,
        w1=routed.w13_weight,
        w2=routed.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=HOPPER_E,
        expert_map=routed.expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=ws1,
        workspace2=ws2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return out


@_requires_sm90_marlin
@pytest.mark.parametrize("down_scale", [1.0, 1e-4])
@torch.inference_mode()
def test_hopper_marlin_moe_runs_with_triton_finalize(
    dist_init, workspace_init, down_scale
) -> None:
    checkpoint = _nvfp4_checkpoint()
    checkpoint["w2_weight_scale_2"].mul_(down_scale)
    router, norm, runner, vllm_config, experts = _build(
        checkpoint, nvfp4_quant_config()
    )
    assert type(experts) is MarlinExperts
    assert experts.g1_alphas is not None and experts.g1_alphas.shape == (E,)

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
    router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    output = _run_checked(runner, vllm_config, hidden_states, router_logits)

    w13 = torch.empty(E, 2 * N, K, dtype=torch.bfloat16, device="cuda")
    for expert in range(E):
        for shard, rows in ((0, slice(0, N)), (1, slice(N, 2 * N))):
            w13[expert, rows] = _dequant(
                checkpoint["w13_weight"][expert, rows],
                checkpoint["w13_weight_scale"][expert, rows],
                checkpoint["w13_weight_scale_2"][expert, shard],
            )
    w2 = _dequant(
        checkpoint["w2_weight"],
        checkpoint["w2_weight_scale"],
        checkpoint["w2_weight_scale_2"],
    )
    expected = _expected(router, norm, hidden_states, router_logits, w13, w2)
    assert _rel_l2(output, expected) < 5e-2

    assert_cudagraph_replays(
        runner,
        vllm_config,
        hidden_states,
        router_logits,
        torch.randn_like(hidden_states),
        torch.randn_like(router_logits),
    )


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_bf16_boundary_layer_runs_with_triton_finalize(
    dist_init, workspace_init
) -> None:
    g = torch.Generator(device="cuda").manual_seed(1)
    weights = {
        "w13_weight": torch.randn(
            E, 2 * N, K, dtype=torch.bfloat16, device="cuda", generator=g
        )
        / 10,
        "w2_weight": torch.randn(
            E, K, N, dtype=torch.bfloat16, device="cuda", generator=g
        )
        / 10,
    }
    router, norm, runner, vllm_config, experts = _build(weights)
    assert type(experts) is TritonExperts

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
    router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    output = _run_checked(runner, vllm_config, hidden_states, router_logits)
    expected = _expected(
        router,
        norm,
        hidden_states,
        router_logits,
        weights["w13_weight"],
        weights["w2_weight"],
    )
    assert _rel_l2(output, expected) < 0.02


def _decode_hopper_nvfp4_packed(
    packed: torch.Tensor,
    row_perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert `encode_rows` on CPU to recover exact FP32 weights."""
    mb = packed.shape[0]
    sub = packed.view(mb, 64, 4, 2304)
    words = sub[..., :2048].contiguous().view(torch.int32).view(mb, 64, 4, 4, 8, 4, 4)
    tiled = torch.empty((mb, 64, 4, 4, 8, 4, 4, 4, 2), dtype=torch.int32)
    for pair_idx in range(4):
        for lane_idx in range(2):
            b0, b1, b2, b3 = hopper_nvfp4_kernels._FP4_BIT_POSITIONS[
                (pair_idx, lane_idx)
            ]
            tiled[..., pair_idx, lane_idx] = (
                ((words >> b0) & 1)
                | (((words >> b1) & 1) << 1)
                | (((words >> b2) & 1) << 2)
                | (((words >> b3) & 1) << 3)
            )

    codes = (
        tiled.view(mb, 64, 4, 4, 8, 4, 4, 2, 2, 2)
        .permute(0, 2, 3, 8, 4, 1, 6, 7, 5, 9)
        .contiguous()
        .view(mb * 256, 256, 16)
    )
    scale_bytes = (
        sub[..., 2048:]
        .contiguous()
        .view(mb, 64, 4, 4, 8, 4, 2)
        .permute(0, 2, 3, 6, 4, 1, 5)
        .contiguous()
        .view(mb * 256, 256)
    )
    exp = (scale_bytes >> 3).to(torch.int32) - hopper_nvfp4_kernels._E5_BIAS
    scales = (1.0 + (scale_bytes & 0x7).to(torch.float32) * 0.125) * torch.exp2(
        exp.float()
    )
    vals = (_E2M1_TABLE_CPU[codes] * scales.unsqueeze(-1)).view(mb * 256, HOPPER_K)
    return vals[torch.argsort(row_perm)] if row_perm is not None else vals


def _raw_nvfp4_dequant_cpu(wq: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Exact CPU FP32 dequantization of unscaled `[R, 2048]` FP4 + `[R, 256]` scales."""
    codes = torch.stack((wq & 0xF, wq >> 4), dim=-1).view(wq.shape[0], 256, 16)
    return (_E2M1_TABLE_CPU[codes.long()] * sf.float().unsqueeze(-1)).view(
        wq.shape[0], HOPPER_K
    )


def _sample_cpu_nvfp4_weights(num_experts: int, *, seed: int):
    """Build CPU raw NVFP4 weights covering all 254 finite E4M3FN block scales."""
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * HOPPER_N, HOPPER_K // 2),
        dtype=torch.uint8,
        generator=g,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, HOPPER_K, HOPPER_N // 2),
        dtype=torch.uint8,
        generator=g,
    )
    # Every finite FP8 E4M3FN byte: +/-0.0, all 7 +/- subnormals, and all normals.
    valid = torch.tensor(
        [b for b in range(256) if (b & 0x7F) != 0x7F], dtype=torch.uint8
    )
    w13_sf_u8 = valid[
        torch.randint(
            0, len(valid), (num_experts, 2 * HOPPER_N, HOPPER_K // 16), generator=g
        )
    ]
    w2_sf_u8 = valid[
        torch.randint(
            0, len(valid), (num_experts, HOPPER_K, HOPPER_N // 16), generator=g
        )
    ]
    # Guarantee row 0 covers all 254 finite E4M3FN values with non-zero nibbles.
    w13_sf_u8[0, 0, : len(valid)] = valid
    w2_sf_u8[0, 0, : len(valid)] = valid
    w13[0, 0, :] = 0x21
    w2[0, 0, :] = 0x43
    w13_s2 = (
        torch.rand((num_experts, 2), dtype=torch.float32, generator=g) * 0.05 + 0.01
    )
    w2_s2 = torch.rand((num_experts,), dtype=torch.float32, generator=g) * 0.05 + 0.01
    return w13, w13_sf_u8, w13_s2, w2, w2_sf_u8, w2_s2


def test_hopper_nvfp4_e5m3_encoder_roundtrip_and_subnormals() -> None:
    """CPU unit test: exact FP4+E5M3 roundtrip (including subnormals) & rejections."""
    num_experts = 1
    w13, w13_sf_u8, w13_s2, w2, w2_sf_u8, w2_s2 = _sample_cpu_nvfp4_weights(
        num_experts, seed=42
    )
    w13_sf, w2_sf = (
        w13_sf_u8.view(torch.float8_e4m3fn),
        w2_sf_u8.view(torch.float8_e4m3fn),
    )

    w13_packed, w2_packed, gs13_gate, gs13_up, gs2 = encode_hopper_nvfp4_weights(
        w13, w13_sf, w13_s2, w2, w2_sf, w2_s2
    )
    assert w13_packed.shape == (num_experts, 32, 64, 9216)
    assert w2_packed.shape == (num_experts, 16, 64, 9216)
    assert w13_packed.dtype == torch.uint8 and w2_packed.dtype == torch.uint8
    # 128.0 is the BF16 exponent bias the kernel dequantizer assumes.
    torch.testing.assert_close(gs13_gate, w13_s2[:, 0] * 128.0, rtol=0, atol=0)
    torch.testing.assert_close(gs13_up, w13_s2[:, 1] * 128.0, rtol=0, atol=0)
    torch.testing.assert_close(gs2, w2_s2 * 128.0, rtol=0, atol=0)

    perm = hopper_nvfp4_kernels.w13_row_permutation("cpu")
    w13_decoded = _decode_hopper_nvfp4_packed(w13_packed[0], row_perm=perm)
    w2_decoded = _decode_hopper_nvfp4_packed(w2_packed[0], row_perm=None)
    assert torch.equal(w13_decoded, _raw_nvfp4_dequant_cpu(w13[0], w13_sf[0]))
    assert torch.equal(w2_decoded, _raw_nvfp4_dequant_cpu(w2[0], w2_sf[0]))

    # Confirm E4M3 subnormal blocks (0x01..0x07), zero (0x00), and negative
    # subnormals (0x81..0x87) round-trip with exact values.
    row0_blocks = w13_decoded[0].view(256, 16)
    assert torch.equal(row0_blocks[0], torch.zeros(16))
    for sub_byte in range(1, 8):
        sub_val = sub_byte * (2.0**-9)
        assert row0_blocks[sub_byte, 0].item() == 0.5 * sub_val
        assert row0_blocks[sub_byte, 1].item() == 1.0 * sub_val
        neg_idx = sub_byte + 127
        assert row0_blocks[neg_idx, 0].item() == -0.5 * sub_val
        assert row0_blocks[neg_idx, 1].item() == -1.0 * sub_val

    # 480.0 (e5 = 24, m3 = 7) is the largest finite scale decode_scale_pair accepts.
    max_sf = torch.ones((num_experts, HOPPER_K, HOPPER_N // 16), dtype=torch.float32)
    max_sf[0, 0, 0] = 480.0
    _, w2_max_packed, _, _, _ = encode_hopper_nvfp4_weights(
        w13, w13_sf, w13_s2, w2, max_sf, w2_s2
    )
    assert torch.equal(
        _decode_hopper_nvfp4_packed(w2_max_packed[0], row_perm=None),
        _raw_nvfp4_dequant_cpu(w2[0], max_sf[0]),
    )

    bad_e4m3 = w13_sf_u8.clone()
    bad_e4m3[0, 0, 0] = 0x7F
    with pytest.raises(ValueError, match="Non-finite"):
        encode_hopper_nvfp4_weights(
            w13, bad_e4m3.view(torch.float8_e4m3fn), w13_s2, w2, w2_sf, w2_s2
        )

    for bad_val in (float("inf"), float("-inf"), float("nan")):
        bad_sf = torch.ones((num_experts, 2 * HOPPER_N, HOPPER_K // 16))
        bad_sf[0, 0, 0] = bad_val
        with pytest.raises(ValueError, match="Non-finite"):
            encode_hopper_nvfp4_weights(w13, bad_sf, w13_s2, w2, w2_sf, w2_s2)

    # Reject inexact mantissas and exponents decode_scale_pair overflows (>= 512).
    for inexact_val in (1.0 + 1.0 / 16.0, 2.0**-17, 2.0**9, 2.0**16):
        bad_sf = torch.ones((num_experts, HOPPER_K, HOPPER_N // 16))
        bad_sf[0, 0, 0] = inexact_val
        with pytest.raises(ValueError, match="E5M3"):
            encode_hopper_nvfp4_weights(w13, w13_sf, w13_s2, w2, bad_sf, w2_s2)

    for bad_val in (float("nan"), float("inf")):
        bad_gs13 = w13_s2.clone()
        bad_gs13[0, 1] = bad_val
        with pytest.raises(ValueError, match="Non-finite"):
            encode_hopper_nvfp4_weights(w13, w13_sf, bad_gs13, w2, w2_sf, w2_s2)


def test_hopper_nvfp4_packed_layout_is_pinned() -> None:
    """CPU unit test: pin packed bytes independently of the roundtrip decoder."""
    valid = torch.tensor(
        [b for b in range(256) if (b & 0x7F) != 0x7F], dtype=torch.uint8
    )
    w13 = (
        ((torch.arange(8192 * 2048) * 37 + 11) % 256)
        .to(torch.uint8)
        .view(1, 8192, 2048)
    )
    w2 = (
        ((torch.arange(4096 * 2048) * 53 + 7) % 256).to(torch.uint8).view(1, 4096, 2048)
    )
    w13_sf = valid[(torch.arange(8192 * 256) * 13) % 254].view(1, 8192, 256)
    w2_sf = valid[(torch.arange(4096 * 256) * 29) % 254].view(1, 4096, 256)
    w13_packed, w2_packed, _, _, _ = encode_hopper_nvfp4_weights(
        w13,
        w13_sf.view(torch.float8_e4m3fn),
        torch.tensor([[0.0125, 0.05]]),
        w2,
        w2_sf.view(torch.float8_e4m3fn),
        torch.tensor([0.025]),
    )
    assert (
        hashlib.sha256(w13_packed.numpy().tobytes()).hexdigest()
        == "84aa58028628e17ec6ff29cc09a0f324ea8a9419253471da9731d878ba67fa9f"
    )
    assert (
        hashlib.sha256(w2_packed.numpy().tobytes()).hexdigest()
        == "876f251d9c6d7ca13a066af71567f46e11eb8e65de4fcee7987141a1bc528f1f"
    )


def _make_hopper_gate_inputs(
    *,
    num_experts: int = 256,
    num_local_experts: int = HOPPER_E,
    hidden_dim: int = HOPPER_K,
    intermediate_size: int = HOPPER_N,
    top_k: int = HOPPER_TOPK,
    ep_size: int = 8,
    tp_size: int = 1,
    dp_size: int = 1,
    pcp_size: int = 1,
    sp_size: int = 1,
    all2all_backend: str = "allgather_reducescatter",
    enable_eplb: bool = False,
    is_lora_enabled: bool = False,
    moe_backend: str = "auto",
    in_dtype: torch.dtype = torch.bfloat16,
    activation: MoEActivation = MoEActivation.SILU,
    has_bias: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    norm_cls: type[torch.nn.Module] = MuseSparkRMSNormWithInputScale,
    var_hidden_size: int | None = None,
    with_finalize: bool = True,
):
    parallel_config = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=pcp_size,
        dp_size=dp_size,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=sp_size,
        use_ep=ep_size > 1,
        all2all_backend=all2all_backend,
        enable_eplb=enable_eplb,
    )
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=parallel_config,
        in_dtype=in_dtype,
        max_num_tokens=64,
        activation=activation,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.MiniMax2,
        moe_backend=moe_backend,
        is_lora_enabled=is_lora_enabled,
        has_bias=has_bias,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    if not with_finalize:
        return moe_config, None
    norm = norm_cls(
        hidden_dim,
        eps=1e-8,
        var_hidden_size=var_hidden_size,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    )
    return moe_config, MusePostExpertFinalize(norm, moe_config)


def test_hopper_nvfp4_selection_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU unit test: verify Hopper NVFP4 selection gate accept/reject rules."""
    assert MuseSparkModelOptNvFp4FusedMoE.supports_pre_processed_weights is False

    # Keep MusePostExpertFinalize CPU-only so tp_size>1 never queries dist state.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hopper_nvfp4_kernels.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        hopper_nvfp4_kernels.current_platform, "is_cuda_alike", lambda: True
    )
    monkeypatch.setattr(
        hopper_nvfp4_kernels.current_platform,
        "is_device_capability_family",
        lambda family: family == 90,
    )
    monkeypatch.setattr(hopper_nvfp4_kernels, "_current_cuda_sm_count", lambda: 132)
    monkeypatch.setattr(hopper_nvfp4_kernels.envs, "VLLM_BATCH_INVARIANT", False)

    ep8_cfg, ep8_finalize = _make_hopper_gate_inputs()
    assert is_hopper_nvfp4_moe_supported(ep8_cfg, ep8_finalize)
    single_cfg, single_finalize = _make_hopper_gate_inputs(
        num_experts=HOPPER_E, ep_size=1
    )
    assert is_hopper_nvfp4_moe_supported(single_cfg, single_finalize)

    # MIG SM-count guard: accept 20..255, reject < 20 or >= 256.
    for sm_count, expected in (
        (16, False),
        (19, False),
        (20, True),
        (132, True),
        (256, False),
    ):
        monkeypatch.setattr(
            hopper_nvfp4_kernels, "_current_cuda_sm_count", lambda n=sm_count: n
        )
        assert is_hopper_nvfp4_moe_supported(ep8_cfg, ep8_finalize) is expected
    monkeypatch.setattr(hopper_nvfp4_kernels, "_current_cuda_sm_count", lambda: 132)

    # Real MusePostExpertFinalize with pre_norm_all_reduce=True must be rejected.
    _, tp_finalize = _make_hopper_gate_inputs(
        tp_size=2, intermediate_size=2 * HOPPER_N, num_experts=HOPPER_E, ep_size=1
    )
    assert tp_finalize is not None and tp_finalize.pre_norm_all_reduce
    assert not is_hopper_nvfp4_moe_supported(ep8_cfg, tp_finalize)

    rejected_cases: list[dict[str, Any]] = [
        {"hidden_dim": 2048},
        {"intermediate_size": 2048},
        {"num_local_experts": 16},
        {"num_experts": 512},
        {"top_k": 4},
        {"moe_backend": "marlin"},
        {"in_dtype": torch.float16},
        {"activation": MoEActivation.GELU},
        {"activation": MoEActivation.SILU_NO_MUL},
        {"has_bias": True},
        {"swiglu_limit": 7.0},
        {"swiglu_alpha": 1.5},
        {"swiglu_beta": 1.0},
        {"with_finalize": False},
        {"norm_cls": MuseSparkRMSNorm},
        {"var_hidden_size": 2048},
        {"enable_eplb": True},
        {"dp_size": 2, "num_experts": HOPPER_E, "ep_size": 1},
        {"pcp_size": 2, "num_experts": HOPPER_E, "ep_size": 1},
        {"sp_size": 2, "num_experts": HOPPER_E, "ep_size": 1},
        {"all2all_backend": "passthrough"},
        {
            "tp_size": 2,
            "intermediate_size": 2 * HOPPER_N,
            "num_experts": HOPPER_E,
            "ep_size": 1,
        },
        {"is_lora_enabled": True},
    ]
    for overrides in rejected_cases:
        cfg, finalize = _make_hopper_gate_inputs(**overrides)
        assert not is_hopper_nvfp4_moe_supported(cfg, finalize), (
            f"Expected rejection for {overrides}"
        )

    monkeypatch.setattr(hopper_nvfp4_kernels.envs, "VLLM_BATCH_INVARIANT", True)
    assert not is_hopper_nvfp4_moe_supported(ep8_cfg, ep8_finalize)
    monkeypatch.setattr(hopper_nvfp4_kernels.envs, "VLLM_BATCH_INVARIANT", False)

    for is_cuda, family in ((True, 100), (False, 0)):
        monkeypatch.setattr(
            hopper_nvfp4_kernels.current_platform, "is_cuda", lambda v=is_cuda: v
        )
        monkeypatch.setattr(
            hopper_nvfp4_kernels.current_platform,
            "is_device_capability_family",
            lambda f, want=family: f == want,
        )
        assert not is_hopper_nvfp4_moe_supported(ep8_cfg, ep8_finalize)


def test_hopper_nvfp4_jit_failure_is_cached_and_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU unit test: a failing JIT build warns once, returns None, and is cached."""
    attempts: list[str] = []
    warnings: list[str] = []

    def failing_compile(name: str, source_path: str) -> Any:
        attempts.append(name)
        raise RuntimeError("nvcc not found")

    monkeypatch.setattr(hopper_nvfp4_kernels, "_extensions", None)
    monkeypatch.setattr(hopper_nvfp4_kernels, "_build_failed", False)
    monkeypatch.setattr(hopper_nvfp4_kernels, "_compile_extension", failing_compile)
    monkeypatch.setattr(
        hopper_nvfp4_kernels.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args if args else str(msg)),
    )
    assert hopper_nvfp4_kernels.build_hopper_nvfp4_moe() is None
    assert hopper_nvfp4_kernels.build_hopper_nvfp4_moe() is None
    assert len(attempts) == 1
    assert len(warnings) == 1
    assert "nvcc not found" in warnings[0]
    assert "falling back to Marlin" in warnings[0]


def test_hopper_nvfp4_packing_keeps_block_scales_in_own_storage() -> None:
    """CPU unit test: packing empties each block scale in its own storage, which
    direct reload keeps for the next update, and preserves weight data_ptr."""
    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.zeros((1, 2 * HOPPER_N, HOPPER_K // 2), dtype=torch.uint8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.zeros((1, 2 * HOPPER_N, HOPPER_K // 16), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.zeros((1, HOPPER_K, HOPPER_N // 2), dtype=torch.uint8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.zeros((1, HOPPER_K, HOPPER_N // 16), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    record_checkpoint_layout(layer)

    w13_packed = torch.full((1, 32, 64, 9216), 3, dtype=torch.uint8)
    w2_packed = torch.full((1, 16, 64, 9216), 5, dtype=torch.uint8)
    hopper_nvfp4_kernels._replace_packed_weights(layer, w13_packed, w2_packed)

    ckpt_scale_bytes = 2 * HOPPER_N * (HOPPER_K // 16) + HOPPER_K * (HOPPER_N // 16)
    for name in ("w13_weight", "w2_weight"):
        scale = getattr(layer, f"{name}_scale")
        assert scale.numel() == 0
        assert (
            scale.untyped_storage().data_ptr()
            != getattr(layer, name).untyped_storage().data_ptr()
        )
    unique_bytes = sum(
        {
            p.untyped_storage().data_ptr(): p.untyped_storage().nbytes()
            for p in layer.parameters()
        }.values()
    )
    assert unique_bytes == w13_packed.nbytes + w2_packed.nbytes + ckpt_scale_bytes

    w13_ptr = layer.w13_weight.data_ptr()
    w2_ptr = layer.w2_weight.data_ptr()
    begin_direct_reload(layer)
    assert layer.w13_weight.shape == (1, 2 * HOPPER_N, HOPPER_K // 2)
    assert layer.w13_weight_scale.shape == (1, 2 * HOPPER_N, HOPPER_K // 16)
    assert layer.w2_weight.shape == (1, HOPPER_K, HOPPER_N // 2)
    assert layer.w2_weight_scale.shape == (1, HOPPER_K, HOPPER_N // 16)
    assert not hopper_nvfp4_kernels._tensors_overlap(
        layer.w13_weight, layer.w13_weight_scale
    )
    assert not hopper_nvfp4_kernels._tensors_overlap(
        layer.w2_weight, layer.w2_weight_scale
    )

    w13_repacked = torch.full_like(w13_packed, 7)
    w2_repacked = torch.full_like(w2_packed, 9)
    hopper_nvfp4_kernels._replace_packed_weights(layer, w13_repacked, w2_repacked)
    assert layer.w13_weight.data_ptr() == w13_ptr
    assert layer.w2_weight.data_ptr() == w2_ptr
    assert torch.equal(layer.w13_weight, w13_repacked)
    assert torch.equal(layer.w2_weight, w2_repacked)
    assert layer.w13_weight_scale.numel() == 0
    assert layer.w2_weight_scale.numel() == 0


_TOY_NVFP4_EXPERT_SHAPES = {  # one expert, in ModelOpt registration order
    "w13_weight": ((1, 2 * HOPPER_N, HOPPER_K // 2), torch.uint8),
    "w2_weight": ((1, HOPPER_K, HOPPER_N // 2), torch.uint8),
    "w13_weight_scale": ((1, 2 * HOPPER_N, HOPPER_K // 16), torch.float8_e4m3fn),
    "w2_weight_scale": ((1, HOPPER_K, HOPPER_N // 16), torch.float8_e4m3fn),
    "w13_weight_scale_2": ((1, 2), torch.float32),
    "w2_weight_scale_2": ((1,), torch.float32),
}


class _ToyNvFp4Experts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name, (shape, dtype) in _TOY_NVFP4_EXPERT_SHAPES.items():
            self.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(shape, dtype=dtype), requires_grad=False
                ),
            )


class _ToyNvFp4Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _ToyNvFp4Experts()

    def load_weights(self, weights) -> set[str]:
        loaded = set()
        for name, tensor in weights:
            param = getattr(self.experts, name.removeprefix("experts."))
            param.data[0].copy_(tensor[0])  # per-expert narrow + copy_, as FusedMoE
            loaded.add(name)
        return loaded


def _toy_nvfp4_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    ckpt = {}
    for name, (shape, dtype) in _TOY_NVFP4_EXPERT_SHAPES.items():
        if dtype == torch.uint8:
            t = torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)
        elif dtype == torch.float8_e4m3fn:  # finite positive E4M3 codes
            t = torch.randint(0, 0x7F, shape, dtype=torch.uint8, generator=gen)
            t = t.view(torch.float8_e4m3fn)
        else:
            t = torch.rand(shape, generator=gen) + 0.5
        ckpt[f"experts.{name}"] = t
    return ckpt


def test_hopper_nvfp4_direct_reload_load_plan_credits_block_scales() -> None:
    """CPU unit test: a planned direct-reload update (the NCCL weight-transfer
    load plan) credits each emptied block scale to its own parameter, so the
    update passes the completeness check at finish."""
    model = _ToyNvFp4Model()
    track_loaded_params(model)  # as a direct-mode cold start does
    model.load_weights(_toy_nvfp4_checkpoint(seed=0).items())
    record_checkpoint_layout(model)
    hopper_nvfp4_kernels._replace_packed_weights(
        model.experts,
        torch.full((1, 32, 64, 9216), 3, dtype=torch.uint8),
        torch.full((1, 16, 64, 9216), 5, dtype=torch.uint8),
    )

    update = _toy_nvfp4_checkpoint(seed=1)
    start_reload(model)
    plan = plan_loads(
        model, [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in update.items()]
    )
    load_planned(model, plan, list(update.items()))
    model_config = SimpleNamespace(
        word_embeddings_untied_by_checkpoint=False,
        quantization=None,
        dtype=torch.bfloat16,
    )
    finish_reload(model, model_config, torch.device("cpu"))

    for name in ("w13_weight_scale", "w2_weight_scale"):
        assert torch.equal(
            getattr(model.experts, name).view(torch.uint8),
            update[f"experts.{name}"].view(torch.uint8),
        )


def test_hopper_nvfp4_experts_requires_muse_quant_config() -> None:
    """CPU unit test: a quant config without the split global scales is rejected."""
    moe_config, _ = _make_hopper_gate_inputs(num_experts=HOPPER_E, ep_size=1)
    plain_qcfg = nvfp4_w4a16_moe_quant_config(
        g1_alphas=torch.ones(HOPPER_E),
        g2_alphas=torch.ones(HOPPER_E),
        w1_scale=torch.empty(0),
        w2_scale=torch.empty(0),
    )
    with pytest.raises(TypeError, match="MuseSparkNvFp4MoEQuantConfig"):
        MuseSparkHopperNvFp4Experts(moe_config=moe_config, quant_config=plain_qcfg)


def _hopper_nvfp4_checkpoint(
    *, subnormal_scales: bool = False, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Build an EP8-shape (`E=32, N=4096, K=4096`) NVFP4 checkpoint on CUDA."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def packed(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    def rand_f32(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, dtype=torch.float32, device="cuda", generator=g)

    w13_weight = packed(HOPPER_E, 2 * HOPPER_N, HOPPER_K // 2)
    w2_weight = packed(HOPPER_E, HOPPER_K, HOPPER_N // 2)
    gate_up_ratio = torch.linspace(
        0.25, 4.0, HOPPER_E, dtype=torch.float32, device="cuda"
    )
    sf13_shape = (HOPPER_E, 2 * HOPPER_N, HOPPER_K // 16)
    sf2_shape = (HOPPER_E, HOPPER_K, HOPPER_N // 16)

    if subnormal_scales:
        # E4M3 subnormals 0x01..0x07 plus a zero-weight 448.0 anchor spanning the
        # full E4M3 range.
        w13_sf_u8 = torch.randint(
            1, 8, sf13_shape, dtype=torch.uint8, device="cuda", generator=g
        )
        w2_sf_u8 = torch.randint(
            1, 8, sf2_shape, dtype=torch.uint8, device="cuda", generator=g
        )
        w13_sf_u8[0, 0, 0] = 0x7E
        w13_weight[0, 0, :8] = 0
        w2_sf_u8[0, 0, 0] = 0x7E
        w2_weight[0, 0, :8] = 0
        w13_scale = w13_sf_u8.view(torch.float8_e4m3fn)
        w2_scale = w2_sf_u8.view(torch.float8_e4m3fn)
        s2_lo, s2_span = 0.5, 0.5
    else:
        normal = torch.tensor(
            [0x30, 0x38, 0x3C, 0x40], dtype=torch.uint8, device="cuda"
        )
        w13_scale = normal[
            torch.randint(0, len(normal), sf13_shape, device="cuda", generator=g)
        ].view(torch.float8_e4m3fn)
        w2_scale = normal[
            torch.randint(0, len(normal), sf2_shape, device="cuda", generator=g)
        ].view(torch.float8_e4m3fn)
        s2_lo, s2_span = 0.005, 0.005

    gate_s2 = rand_f32(HOPPER_E) * s2_span + s2_lo
    w2_s2 = rand_f32(HOPPER_E) * s2_span + s2_lo
    return {
        "w13_weight": w13_weight,
        "w13_weight_scale": w13_scale,
        "w13_weight_scale_2": torch.stack(
            (gate_s2, gate_s2 * gate_up_ratio), dim=1
        ).contiguous(),
        "w13_input_scale": rand_f32(HOPPER_E, 2) + 0.5,
        "w2_weight": w2_weight,
        "w2_weight_scale": w2_scale,
        "w2_weight_scale_2": w2_s2,
        "w2_input_scale": rand_f32(HOPPER_E) + 0.5,
    }


def _nvfp4_swiglu_moe_ref(
    checkpoint: dict[str, torch.Tensor],
    norm: Any,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Streaming per-expert FP32 SwiGLU MoE reference for NVFP4 checkpoints."""
    out = torch.zeros(
        hidden_states.shape, dtype=torch.float32, device=hidden_states.device
    )
    num_experts = checkpoint["w13_weight"].shape[0]
    inter_dim = checkpoint["w13_weight"].shape[1] // 2
    for expert in range(num_experts):
        token_idx, slot = (topk_ids == expert).nonzero(as_tuple=True)
        if token_idx.numel() == 0:
            continue
        x_e = hidden_states[token_idx].float()
        w_gate = _dequant(
            checkpoint["w13_weight"][expert, :inter_dim],
            checkpoint["w13_weight_scale"][expert, :inter_dim],
            checkpoint["w13_weight_scale_2"][expert, 0],
            torch.float32,
        )
        w_up = _dequant(
            checkpoint["w13_weight"][expert, inter_dim:],
            checkpoint["w13_weight_scale"][expert, inter_dim:],
            checkpoint["w13_weight_scale_2"][expert, 1],
            torch.float32,
        )
        w2_e = _dequant(
            checkpoint["w2_weight"][expert],
            checkpoint["w2_weight_scale"][expert],
            checkpoint["w2_weight_scale_2"][expert],
            torch.float32,
        )
        inter = F.silu(x_e @ w_gate.t()) * (x_e @ w_up.t())
        y = norm.forward_native(
            (inter @ w2_e.t()).to(torch.bfloat16), out_dtype=torch.float32
        )
        out.index_add_(0, token_idx, y * topk_weights[token_idx, slot, None].float())
    return out


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected).norm() / expected.norm()).item()


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_nvfp4_cuda_pack_experts_parity_and_errors() -> None:
    """SM90 CUDA `pack_experts` matches CPU bit-for-bit and rejects invalid scales."""
    exts = hopper_nvfp4_kernels.build_hopper_nvfp4_moe()
    assert exts is not None
    decode_ext, _ = exts
    assert hasattr(decode_ext, "pack_experts")

    cuda_inputs = cuda_out = None
    try:
        w13, w13_sf_u8, w13_s2, w2, w2_sf_u8, w2_s2 = _sample_cpu_nvfp4_weights(
            2, seed=101
        )
        cpu_inputs = (
            w13,
            w13_sf_u8.view(torch.float8_e4m3fn),
            w13_s2,
            w2,
            w2_sf_u8.view(torch.float8_e4m3fn),
            w2_s2,
        )
        cpu_out = encode_hopper_nvfp4_weights(*cpu_inputs)
        cuda_inputs = tuple(t.cuda() for t in cpu_inputs)
        cuda_out = encode_hopper_nvfp4_weights(*cuda_inputs, decode_ext=decode_ext)
        assert len(cuda_out) == len(cpu_out) == 5
        for got, expected in zip(cuda_out, cpu_out):
            assert got.is_cuda and torch.equal(got.cpu(), expected)

        w13_c, w13_sf_c, w13_s2_c, w2_c, w2_sf_c, w2_s2_c = (t[:1] for t in cuda_inputs)
        # Reject E4M3FN NaN (0x7f) on CUDA pack_experts.
        bad_e4m3 = w13_sf_u8[:1].cuda()
        bad_e4m3[0, 0, 0] = 0x7F
        with pytest.raises(ValueError):
            encode_hopper_nvfp4_weights(
                w13_c,
                bad_e4m3.view(torch.float8_e4m3fn),
                w13_s2_c,
                w2_c,
                w2_sf_c,
                w2_s2_c,
                decode_ext=decode_ext,
            )

        # Reject NaN, Inf, -Inf, inexact mantissa, and overflowed E5M3 (2**9) on CUDA.
        for bad_val in (
            float("nan"),
            float("inf"),
            float("-inf"),
            1.0 + 2.0**-7,
            2.0**9,
        ):
            bad_sf = torch.ones(
                (1, HOPPER_K, HOPPER_N // 16), dtype=torch.float32, device="cuda"
            )
            bad_sf[0, 0, 0] = bad_val
            with pytest.raises(ValueError):
                encode_hopper_nvfp4_weights(
                    w13_c,
                    w13_sf_c,
                    w13_s2_c,
                    w2_c,
                    bad_sf,
                    w2_s2_c,
                    decode_ext=decode_ext,
                )
    finally:
        del cuda_inputs, cuda_out
        _free_cuda()


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_nvfp4_moe_accuracy_and_fallback(dist_init, workspace_init) -> None:
    """SM90 accuracy vs FP32 ref & Marlin, boundaries, NaN invariance, CUDA graphs."""
    assert hopper_nvfp4_kernels.DECODE_MAX_TOKENS == 1536

    ckpt = sub_ckpt = routed = None
    router = norm = runner = vllm_config = experts = None
    marlin_runner = marlin_cfg = marlin_experts = None
    try:
        # 1. Normal scales + split w13_weight_scale_2 (up to 4x gate/up ratio).
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=0)
        router, norm, runner, vllm_config, experts = _build_hopper(ckpt)
        assert type(experts) is MuseSparkHopperNvFp4Experts
        assert experts.post_gemm2_hook is runner.post_expert_finalize

        _, _, marlin_runner, marlin_cfg, marlin_experts = _build_hopper(
            ckpt, moe_backend="marlin"
        )
        assert type(marlin_experts) is MarlinExperts
        assert marlin_experts.post_gemm2_hook is marlin_runner.post_expert_finalize

        g = torch.Generator(device="cuda").manual_seed(11)
        for num_tokens in (1, 8, 16, 17, 64, 256):
            x, logits = _rand_inputs(num_tokens, g)
            hopper_out = _run_checked(runner, vllm_config, x, logits)
            marlin_out = _run_checked(marlin_runner, marlin_cfg, x, logits)
            ref = _ref_for(ckpt, router, norm, x, logits)
            hopper_err = _rel_l2(hopper_out, ref)
            marlin_err = _rel_l2(marlin_out, ref)
            assert hopper_err < 1e-2, f"T={num_tokens}: hopper rel-L2 {hopper_err}"
            assert hopper_err <= marlin_err, (
                f"T={num_tokens}: hopper {hopper_err} > marlin {marlin_err}"
            )

        marlin_runner = marlin_cfg = marlin_experts = None
        _free_cuda()

        # 2. Decode/extended-decode (T=1023, 1024, 1536) and prefill (1537, 2048).
        for num_tokens in (1023, 1024, 1536, 1537, 2048):
            x, logits = _rand_inputs(num_tokens, g)
            hopper_out = _run_checked(runner, vllm_config, x, logits)
            assert _rel_l2(hopper_out, _ref_for(ckpt, router, norm, x, logits)) < 1e-2
            assert torch.equal(hopper_out, run_moe(runner, vllm_config, x, logits))

        # Adversarial all-8-local routes at T=1536 (12288 rows, zero overflow).
        routed = runner.routed_experts
        x_adv, _ = _rand_inputs(1536, g)
        adv_ids = (
            torch.arange(HOPPER_TOPK, dtype=torch.int32, device="cuda")
            .expand(1536, HOPPER_TOPK)
            .contiguous()
        )
        adv_weights = torch.full(
            (1536, HOPPER_TOPK), 1.0 / HOPPER_TOPK, dtype=torch.float32, device="cuda"
        )
        assert adv_ids.numel() == 1536 * HOPPER_TOPK == 12288
        out_adv = _apply_experts(
            experts, routed, x_adv, adv_weights, adv_ids, fill_value=float("nan")
        )
        out_adv_repeat = _apply_experts(experts, routed, x_adv, adv_weights, adv_ids)
        assert torch.isfinite(out_adv).all()
        assert torch.equal(out_adv, out_adv_repeat)
        ref_adv = _nvfp4_swiglu_moe_ref(ckpt, norm, x_adv, adv_weights, adv_ids)
        assert _rel_l2(out_adv, ref_adv) < 1e-2

        # 3. T=0 and all-off-rank (-1) IDs must produce exact zeros.
        empty_out = run_moe(
            runner,
            vllm_config,
            torch.empty((0, HOPPER_K), dtype=torch.bfloat16, device="cuda"),
            torch.empty((0, HOPPER_E), dtype=torch.float32, device="cuda"),
        )
        assert empty_out.shape == (0, HOPPER_K) and empty_out.dtype == torch.bfloat16

        for num_tokens in (8, 32, 1536, 1537):
            x_off, _ = _rand_inputs(num_tokens, g)
            off_ids = torch.full(
                (num_tokens, HOPPER_TOPK), -1, dtype=torch.int32, device="cuda"
            )
            off_weights = torch.rand(
                (num_tokens, HOPPER_TOPK),
                dtype=torch.float32,
                device="cuda",
                generator=g,
            )
            out_off = _apply_experts(
                experts, routed, x_off, off_weights, off_ids, fill_value=float("nan")
            )
            assert torch.equal(out_off, torch.zeros_like(out_off))

        # 4. NaN-poisoned workspace invariance + deterministic repeat across paths.
        for num_tokens in (16, 64, 1537):
            x_det, logits_det = _rand_inputs(num_tokens, g)
            topk_weights, topk_ids = router._compute_routing(x_det, logits_det, None)
            out_clean = _apply_experts(experts, routed, x_det, topk_weights, topk_ids)
            out_nan = _apply_experts(
                experts, routed, x_det, topk_weights, topk_ids, fill_value=float("nan")
            )
            assert torch.equal(out_clean, out_nan)
            assert torch.equal(
                run_moe(runner, vllm_config, x_det, logits_det),
                run_moe(runner, vllm_config, x_det, logits_det),
            )

        # 5. Bitwise CUDA-graph replay across small-T, decode, extended, and prefill.
        for num_tokens in (1, 16, 64, 1024, 1537):
            x_cg, logits_cg = _rand_inputs(num_tokens, g)
            next_x, next_logits = _rand_inputs(num_tokens, g)
            assert_cudagraph_replays(
                runner, vllm_config, x_cg, logits_cg, next_x, next_logits, exact=True
            )

        routed = router = norm = runner = vllm_config = experts = ckpt = None
        _free_cuda()

        # 6. E4M3 subnormal block scales (0x01..0x07) + split w13_weight_scale_2.
        sub_ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=True, seed=7)
        x_sub, logits_sub = _rand_inputs(64, g)
        router, norm, runner, vllm_config, experts = _build_hopper(sub_ckpt)
        assert type(experts) is MuseSparkHopperNvFp4Experts
        hopper_sub_out = _run_checked(runner, vllm_config, x_sub, logits_sub)
        ref_sub = _ref_for(sub_ckpt, router, norm, x_sub, logits_sub)
        hopper_sub_err = _rel_l2(hopper_sub_out, ref_sub)
        assert hopper_sub_err < 1e-2, f"subnormal hopper rel-L2 {hopper_sub_err}"
    finally:
        del ckpt, sub_ckpt, routed, router, norm, runner, vllm_config, experts
        del marlin_runner, marlin_cfg, marlin_experts
        _free_cuda()


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_nvfp4_moe_reload_idempotency_and_post_norm_refresh(
    dist_init, workspace_init
) -> None:
    """2nd `process_weights_after_loading` stays Hopper; `refresh_post_norm` stays
    in-place."""
    ckpt = router = norm = runner = vllm_config = experts = None
    try:
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=9)
        router, norm, runner, vllm_config, experts = _build_hopper(ckpt)
        routed = runner.routed_experts
        assert type(experts) is MuseSparkHopperNvFp4Experts
        assert routed.quant_method.supports_pre_processed_weights is False
        assert experts._post_weight is not None

        g = torch.Generator(device="cuda").manual_seed(31)
        x, logits = _rand_inputs(16, g)
        out_before = _run_checked(runner, vllm_config, x, logits).clone()
        w13_ptr = routed.w13_weight.data_ptr()
        w2_ptr = routed.w2_weight.data_ptr()
        post_ptr = experts._post_weight.data_ptr()

        # Calling process_weights_after_loading a second time on already-packed
        # Hopper weights must stay on MuseSparkHopperNvFp4Experts without
        # falling back to Marlin or reallocating weight/post-norm buffers.
        routed.quant_method.process_weights_after_loading(routed)
        experts_after = routed.quant_method.moe_kernel.fused_experts
        assert type(experts_after) is MuseSparkHopperNvFp4Experts
        assert experts_after.post_gemm2_hook is runner.post_expert_finalize
        assert routed.w13_weight.data_ptr() == w13_ptr
        assert routed.w2_weight.data_ptr() == w2_ptr
        assert experts_after._post_weight is not None
        assert experts_after._post_weight.data_ptr() == post_ptr
        assert torch.equal(run_moe(runner, vllm_config, x, logits), out_before)

        # Updating norm weights and calling refresh_post_norm must preserve
        # _post_weight.data_ptr() in place for captured CUDA graphs.
        new_weight = torch.linspace(0.3, -0.3, HOPPER_K, device="cuda")
        norm.weight.copy_(new_weight)
        experts_after.refresh_post_norm(routed)
        assert experts_after._post_weight.data_ptr() == post_ptr
        torch.testing.assert_close(
            experts_after._post_weight, new_weight + float(norm.fused_gain_center)
        )

        out_refreshed = _run_checked(runner, vllm_config, x, logits)
        assert not torch.equal(out_refreshed, out_before)
        assert _rel_l2(out_refreshed, _ref_for(ckpt, router, norm, x, logits)) < 1e-2
    finally:
        del ckpt, router, norm, runner, vllm_config, experts
        _free_cuda()


@_requires_sm90_marlin
@pytest.mark.parametrize("reason", ["jit_failure", "rank_disagreement"])
@torch.inference_mode()
def test_hopper_nvfp4_moe_forced_jit_failure_falls_back_to_marlin(
    dist_init, workspace_init, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """Forced JIT build failure or rank disagreement falls back cleanly to Marlin."""
    if reason == "jit_failure":
        monkeypatch.setattr(
            hopper_nvfp4_kernels, "build_hopper_nvfp4_moe", lambda: None
        )
    else:
        monkeypatch.setattr(
            hopper_nvfp4_kernels, "_all_hopper_nvfp4_ranks_ok", lambda *a, **kw: False
        )

    ckpt = None
    router = norm = runner = vllm_config = experts = None
    try:
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=3)

        router, norm, runner, vllm_config, experts = _build_hopper(ckpt)
        assert type(experts) is MarlinExperts
        assert experts.post_gemm2_hook is runner.post_expert_finalize
        assert runner.routed_experts.w13_weight.ndim != 4

        g = torch.Generator(device="cuda").manual_seed(19)
        x, logits = _rand_inputs(16, g)
        output = _run_checked(runner, vllm_config, x, logits)
        assert _rel_l2(output, _ref_for(ckpt, router, norm, x, logits)) < 5e-2
    finally:
        del ckpt, router, norm, runner, vllm_config, experts
        _free_cuda()


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_nvfp4_moe_humming_auto_does_not_crash(
    dist_init, workspace_init, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SM90 `moe_backend='auto'` must not crash when the oracle picks HUMMING."""
    humming_calls = 0

    def _select_humming(*args, **kwargs):
        nonlocal humming_calls
        humming_calls += 1
        return NvFp4MoeBackend.HUMMING, HummingGroupedExperts

    monkeypatch.setattr(modelopt_mod, "select_nvfp4_moe_backend", _select_humming)

    small_ckpt = ckpt = None
    router = norm = runner = vllm_config = experts = None
    try:
        # Ineligible small shape falls back cleanly to MarlinExperts.
        small_ckpt = _nvfp4_checkpoint()
        router, norm, runner, vllm_config, experts = _build(
            small_ckpt, nvfp4_quant_config()
        )
        assert type(experts) is MarlinExperts
        assert experts.post_gemm2_hook is runner.post_expert_finalize

        g = torch.Generator(device="cuda").manual_seed(23)
        x_small, logits_small = _rand_inputs(M, g, hidden_size=K, num_experts=E)
        out_small = _run_checked(runner, vllm_config, x_small, logits_small)
        ref_small = _ref_for(small_ckpt, router, norm, x_small, logits_small)
        assert _rel_l2(out_small, ref_small) < 5e-2

        router = norm = runner = vllm_config = experts = small_ckpt = None
        _free_cuda()

        # Eligible EP8 Muse Spark geometry selects MuseSparkHopperNvFp4Experts.
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=5)
        router, norm, runner, vllm_config, experts = _build_hopper(ckpt)
        assert type(experts) is MuseSparkHopperNvFp4Experts
        assert experts.post_gemm2_hook is runner.post_expert_finalize
        assert humming_calls >= 2

        x, logits = _rand_inputs(16, g)
        output = _run_checked(runner, vllm_config, x, logits)
        assert _rel_l2(output, _ref_for(ckpt, router, norm, x, logits)) < 1e-2
    finally:
        del small_ckpt, ckpt, router, norm, runner, vllm_config, experts
        _free_cuda()


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_nvfp4_moe_global_scales_and_post_eps_are_observable(
    dist_init, workspace_init
) -> None:
    """At post_eps=1.0, gs13_up, gs2, and post_eps are all directly observable."""
    ckpt = router = norm = runner = vllm_config = experts = None
    try:
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=0)
        router, norm, runner, vllm_config, experts = _build_hopper(ckpt, post_eps=1.0)
        assert type(experts) is MuseSparkHopperNvFp4Experts
        assert experts._gs13_gate is not None
        assert experts._gs13_up is not None
        assert experts._gs2 is not None
        assert experts._post_eps == 1.0

        g = torch.Generator(device="cuda").manual_seed(13)
        samples: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for num_tokens in (8, 64, 1024, 1537):
            x, logits = _rand_inputs(num_tokens, g)
            out = _run_checked(runner, vllm_config, x, logits)
            ref = _ref_for(ckpt, router, norm, x, logits)
            err = _rel_l2(out, ref)
            assert err < 1e-2, f"T={num_tokens}: rel-L2 {err}"
            samples.append((num_tokens, x, logits, ref))

        orig_up = experts._gs13_up.clone()
        orig_gs2 = experts._gs2.clone()
        orig_eps = experts._post_eps
        try:
            # Mutating gs13_up := gs13_gate must break accuracy at post_eps=1.0.
            experts._gs13_up.copy_(experts._gs13_gate)
            for num_tokens, x, logits, ref in samples:
                mut_err = _rel_l2(run_moe(runner, vllm_config, x, logits), ref)
                assert mut_err > 5e-2, (
                    f"T={num_tokens}: gs13_up mutation stayed at rel-L2 {mut_err}"
                )
            experts._gs13_up.copy_(orig_up)

            # Mutating gs2 must break accuracy at post_eps=1.0.
            experts._gs2.mul_(2.0)
            for num_tokens, x, logits, ref in samples:
                mut_err = _rel_l2(run_moe(runner, vllm_config, x, logits), ref)
                assert mut_err > 5e-2, (
                    f"T={num_tokens}: gs2 mutation stayed at rel-L2 {mut_err}"
                )
            experts._gs2.copy_(orig_gs2)

            # Mutating post_eps := 1e-8 must break accuracy against eps=1.0 ref.
            experts._post_eps = 1e-8
            for num_tokens, x, logits, ref in samples:
                mut_err = _rel_l2(run_moe(runner, vllm_config, x, logits), ref)
                assert mut_err > 5e-2, (
                    f"T={num_tokens}: post_eps mutation stayed at rel-L2 {mut_err}"
                )
        finally:
            experts._gs13_up.copy_(orig_up)
            experts._gs2.copy_(orig_gs2)
            experts._post_eps = orig_eps
    finally:
        del ckpt, router, norm, runner, vllm_config, experts
        _free_cuda()


@_requires_sm90_marlin
@pytest.mark.parametrize("ep_rank", [0, 7])
@torch.inference_mode()
def test_hopper_nvfp4_moe_ep8_expert_map(
    dist_init, workspace_init, ep_rank: int
) -> None:
    """Global IDs through an EP8 expert_map match local IDs; -1 and >=256 add 0."""
    ckpt = router = norm = runner = vllm_config = experts = None
    try:
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=0)
        router, norm, runner, vllm_config, experts = _build_hopper(ckpt)
        assert type(experts) is MuseSparkHopperNvFp4Experts
        routed = runner.routed_experts

        num_global = 8 * HOPPER_E
        lo = ep_rank * HOPPER_E
        hi = lo + HOPPER_E
        expert_map = torch.full((num_global,), -1, dtype=torch.int32, device="cuda")
        expert_map[lo:hi] = torch.arange(HOPPER_E, dtype=torch.int32, device="cuda")

        g = torch.Generator(device="cuda").manual_seed(29 + ep_rank)

        def run_with_map(
            target_experts: MuseSparkHopperNvFp4Experts,
            target_routed: torch.nn.Module,
            x: torch.Tensor,
            weights: torch.Tensor,
            ids: torch.Tensor,
            emap: torch.Tensor | None,
            global_num_experts: int,
        ) -> torch.Tensor:
            ws1_shape, ws2_shape, out_shape = target_experts.workspace_shapes(
                x.shape[0],
                HOPPER_N,
                HOPPER_K,
                HOPPER_TOPK,
                global_num_experts,
                HOPPER_E,
                None,
                MoEActivation.SILU,
            )
            out = torch.full(out_shape, float("nan"), dtype=x.dtype, device="cuda")
            target_experts.apply(
                output=out,
                hidden_states=x,
                w1=target_routed.w13_weight,
                w2=target_routed.w2_weight,
                topk_weights=weights,
                topk_ids=ids,
                activation=MoEActivation.SILU,
                global_num_experts=global_num_experts,
                expert_map=emap,
                a1q_scale=None,
                a2_scale=None,
                workspace13=torch.full(
                    ws1_shape, float("nan"), dtype=x.dtype, device="cuda"
                ),
                workspace2=torch.full(
                    ws2_shape, float("nan"), dtype=x.dtype, device="cuda"
                ),
                expert_tokens_meta=None,
                apply_router_weight_on_input=False,
            )
            return out

        for num_tokens in (8, 64, 1024, 1537):
            x, _ = _rand_inputs(num_tokens, g)
            scores = torch.rand(
                (num_tokens, num_global),
                dtype=torch.float32,
                device="cuda",
                generator=g,
            )
            global_ids = scores.topk(HOPPER_TOPK, dim=-1).indices.to(torch.int32)
            sentinel = torch.rand(
                (num_tokens, HOPPER_TOPK),
                dtype=torch.float32,
                device="cuda",
                generator=g,
            )
            global_ids = torch.where(sentinel < 0.15, -1, global_ids)
            global_ids = torch.where(
                (sentinel >= 0.15) & (sentinel < 0.25),
                torch.randint(
                    num_global,
                    num_global + 44,
                    (num_tokens, HOPPER_TOPK),
                    dtype=torch.int32,
                    device="cuda",
                    generator=g,
                ),
                global_ids,
            )
            # Token 0 has all -1 slots in a mixed batch to catch rank-7 wraparound.
            global_ids[0] = -1
            local_ids = torch.where(
                (global_ids >= lo) & (global_ids < hi),
                global_ids - lo,
                torch.full_like(global_ids, -1),
            )
            weights = torch.rand(
                (num_tokens, HOPPER_TOPK),
                dtype=torch.float32,
                device="cuda",
                generator=g,
            )

            out_global = run_with_map(
                experts, routed, x, weights, global_ids, expert_map, num_global
            )
            out_local = run_with_map(
                experts, routed, x, weights, local_ids, None, HOPPER_E
            )
            assert torch.isfinite(out_global).all()
            assert torch.equal(out_global[0], torch.zeros_like(out_global[0]))
            assert torch.equal(out_global, out_local)
            ref = _nvfp4_swiglu_moe_ref(ckpt, norm, x, weights, local_ids)
            assert _rel_l2(out_global, ref) < 1e-2
    finally:
        del ckpt, router, norm, runner, vllm_config, experts
        _free_cuda()


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_nvfp4_moe_chunked_apply_matches_per_chunk_calls(
    dist_init, workspace_init
) -> None:
    """Chunks above MAX_PREFILL_CHUNK (8192 prefill + 1 decode tail) match per-chunk
    calls, and a non-contiguous output gets the same result."""
    num_tokens = hopper_nvfp4_kernels.MAX_PREFILL_CHUNK + 1
    ckpt = router = norm = runner = vllm_config = experts = None
    try:
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=5)
        router, norm, runner, vllm_config, experts = _build_hopper(ckpt)
        assert type(experts) is MuseSparkHopperNvFp4Experts
        routed = runner.routed_experts
        g = torch.Generator(device="cuda").manual_seed(23)
        x, logits = _rand_inputs(num_tokens, g)
        topk_weights, topk_ids = router._compute_routing(x, logits, None)

        full = _apply_experts(experts, routed, x, topk_weights, topk_ids)
        step = hopper_nvfp4_kernels.MAX_PREFILL_CHUNK
        parts = torch.cat(
            [
                _apply_experts(
                    experts,
                    routed,
                    x[s : min(s + step, num_tokens)],
                    topk_weights[s : min(s + step, num_tokens)],
                    topk_ids[s : min(s + step, num_tokens)],
                )
                for s in range(0, num_tokens, step)
            ]
        )
        assert torch.equal(full, parts)
        strided = _apply_experts(
            experts, routed, x, topk_weights, topk_ids, noncontiguous_out=True
        )
        assert torch.equal(full, strided)
        ref = _nvfp4_swiglu_moe_ref(ckpt, norm, x, topk_weights, topk_ids)
        assert _rel_l2(full, ref) < 1e-2
    finally:
        del ckpt, router, norm, runner, vllm_config, experts
        _free_cuda()


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_nvfp4_moe_layerwise_reload_in_place(dist_init, workspace_init) -> None:
    """Layerwise reload updates weights and post-norm in place for captured graphs."""
    ckpt = new_ckpt = router = norm = runner = vllm_config = experts = None
    try:
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=0)
        router, norm, runner, vllm_config, experts = _build_hopper(ckpt)
        assert type(experts) is MuseSparkHopperNvFp4Experts
        routed = runner.routed_experts
        assert experts._post_weight is not None
        assert experts._gs13_gate is not None
        assert experts._gs13_up is not None
        assert experts._gs2 is not None

        g = torch.Generator(device="cuda").manual_seed(37)
        x, logits = _rand_inputs(16, g)
        with (
            set_current_vllm_config(vllm_config),
            set_forward_context(None, vllm_config),
        ):
            for _ in range(3):
                runner.forward(hidden_states=x, router_logits=logits)
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_static = runner.forward(hidden_states=x, router_logits=logits)
            graph.replay()
            out_before = out_static.clone()

        ptrs = (
            routed.w13_weight.data_ptr(),
            routed.w2_weight.data_ptr(),
            experts._post_weight.data_ptr(),
            experts._gs13_gate.data_ptr(),
            experts._gs13_up.data_ptr(),
            experts._gs2.data_ptr(),
        )

        new_ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=13)
        new_norm_weight = torch.linspace(0.25, -0.25, HOPPER_K, device="cuda")
        initialize_layerwise_reload(runner)
        loaded = list(routed.load_weights(list(new_ckpt.items())))
        assert len(loaded) == len(new_ckpt)
        norm.weight.copy_(new_norm_weight)
        finalize_layerwise_reload(runner, vllm_config.model_config)

        experts_after = routed.quant_method.moe_kernel.fused_experts
        assert experts_after is experts
        assert experts_after._post_weight is not None
        assert experts_after._gs13_gate is not None
        assert experts_after._gs13_up is not None
        assert experts_after._gs2 is not None
        assert (
            routed.w13_weight.data_ptr(),
            routed.w2_weight.data_ptr(),
            experts_after._post_weight.data_ptr(),
            experts_after._gs13_gate.data_ptr(),
            experts_after._gs13_up.data_ptr(),
            experts_after._gs2.data_ptr(),
        ) == ptrs
        torch.testing.assert_close(
            experts_after._post_weight,
            new_norm_weight + float(norm.fused_gain_center),
        )

        graph.replay()
        assert not torch.equal(out_static, out_before)
        assert _rel_l2(out_static, _ref_for(new_ckpt, router, norm, x, logits)) < 1e-2
    finally:
        del ckpt, new_ckpt, router, norm, runner, vllm_config, experts
        _free_cuda()


@_requires_sm90_marlin
@torch.inference_mode()
def test_hopper_nvfp4_moe_forward_has_no_gpu_sync(
    dist_init, workspace_init, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steady-state forwards on every Hopper regime never sync the host."""
    ckpt = router = norm = runner = vllm_config = experts = None
    try:
        ckpt = _hopper_nvfp4_checkpoint(subnormal_scales=False, seed=0)
        router, norm, runner, vllm_config, experts = _build_hopper(ckpt)
        assert type(experts) is MuseSparkHopperNvFp4Experts

        g = torch.Generator(device="cuda").manual_seed(31)
        inputs = [_rand_inputs(t, g) for t in (1, 16, 64, 1024, 1537)]
        for x, logits in inputs:
            run_moe(runner, vllm_config, x, logits)

        monkeypatch.setattr(gpu_sync_debug, "_SYNC_CHECK_MODE", "error")
        monkeypatch.setattr(gpu_sync_debug, "_sync_check_enabled", True)
        gpu_sync_debug._install_copy_checkers()
        checked_run_moe = gpu_sync_debug.with_gpu_sync_check(run_moe)
        for x, logits in inputs:
            checked_run_moe(runner, vllm_config, x, logits)
    finally:
        del ckpt, router, norm, runner, vllm_config, experts
        _free_cuda()
