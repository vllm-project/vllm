# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
from dataclasses import dataclass
from importlib.util import find_spec

import pytest
import torch
from packaging import version

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

QUARK_MXFP4_AVAILABLE = find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse("0.8.99")

TRTLLM_GEN_MXFP4_AVAILABLE = (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
)

TRTLLM_GEN_MXFP8_AVAILABLE = TRTLLM_GEN_MXFP4_AVAILABLE

HOPPER_MXFP4_BF16_AVAILABLE = (
    current_platform.is_cuda()
    and current_platform.is_device_capability(90)
    and has_flashinfer()
)

# ROCm platform and dependencies
ROCM_AVAILABLE = current_platform.is_rocm()
ROCM_TRITON_KERNELS_AVAILABLE = False
ROCM_AITER_AVAILABLE = False
ROCM_GFX950 = False

if ROCM_AVAILABLE:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950
    from vllm.utils.import_utils import has_triton_kernels

    ROCM_TRITON_KERNELS_AVAILABLE = has_triton_kernels()
    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_AVAILABLE = rocm_aiter_ops.is_enabled()

    if ROCM_AITER_AVAILABLE:
        from aiter.ops.triton.moe.quant_moe import upcast_from_mxfp
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

if TRTLLM_GEN_MXFP4_AVAILABLE:
    from flashinfer import (
        fp4_quantize,
        mxfp8_quantize,
        reorder_rows_for_gated_act_gemm,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        trtllm_fp4_block_scale_moe,
        trtllm_fp8_block_scale_moe,
    )
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave

if TRTLLM_GEN_MXFP8_AVAILABLE:
    from flashinfer.fused_moe.core import (
        Fp8QuantizationType,
        get_w2_permute_indices_with_cache,
    )


@dataclass
class ModelCase:
    model_id: str
    tp: int


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.mark.parametrize(
    "model_case",
    [
        ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4", tp=2),
        ModelCase("fxmarty/deepseek_r1_3_layers_mxfp4", tp=8),
        ModelCase("fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4", tp=1),
        ModelCase("fxmarty/Llama-3.1-70B-Instruct-2-layers-mxfp6", tp=1),
        ModelCase("fxmarty/Llama-3.1-70B-Instruct-2-layers-mxfp6", tp=4),
    ],
)
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
def test_mxfp4_loading_and_execution_moe(vllm_runner, model_case: ModelCase):
    if torch.accelerator.device_count() < model_case.tp:
        pytest.skip(
            f"This test requires >={model_case.tp} gpus, got only "
            f"{torch.accelerator.device_count()}"
        )

    # `cudagraph_capture_sizes=[16]` to reduce load time.
    with vllm_runner(
        model_case.model_id,
        tensor_parallel_size=model_case.tp,
        load_format="dummy",
        compilation_config={"cudagraph_capture_sizes": [16]},
    ) as llm:
        # Disabled as check_model is broken: https://github.com/vllm-project/vllm/pull/18465#issuecomment-3329880562
        # def check_model(model):
        #     from vllm.model_executor.layers.quantization.quark.quark import (  # noqa: E501
        #         QuarkLinearMethod)
        #     from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import QuarkOCP_MX  # noqa: E501
        #     from vllm.model_executor.layers.quantization.quark.quark_moe import (  # noqa: E501
        #         QuarkOCP_MX_MoEMethod)

        #     layer = model.model.layers[0]

        #     qkv_proj = layer.self_attn.qkv_proj

        #     assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
        #     assert isinstance(qkv_proj.scheme, QuarkOCP_MX)

        #     assert isinstance(layer.mlp.experts.quant_method,
        #                       QuarkOCP_MX_MoEMethod)

        # if model_case.model_id == "fxmarty/qwen_1.5-moe-a2.7b-mxfp4":
        #     llm.apply_model(check_model)

        output = llm.generate_greedy("Today I am in the French Alps and", max_tokens=20)
        assert output


def swiglu(x, alpha: float = 1.702, beta: float = 1.0, limit: float | None = None):
    # Note we add an extra bias of 1 to the linear layer
    # Uses chunked layout: first half is gate, second half is up
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + beta)


def swigluoai(x, alpha: float = 1.702, limit: float = 7.0):
    # OAI swiglu uses interleaved layout: gate/up alternating
    # See SwigluOAIAndMul in vllm/model_executor/layers/activation.py
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


fp4_lookup_table = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]


def mxfp4_dequantize(x, scale):
    assert x.dtype == torch.uint8
    x = x.view(torch.uint8).to(torch.int32)
    x_unpacked = torch.zeros(
        *x.shape[:-1], x.shape[-1] * 2, dtype=torch.int32, device=x.device
    )
    x_unpacked[..., 0::2].copy_(x & 0xF)
    x_unpacked[..., 1::2].copy_((x >> 4) & 0xF)

    x_float = torch.zeros(x_unpacked.shape, dtype=torch.float32, device=x.device)
    for i, val in enumerate(fp4_lookup_table):
        x_float[x_unpacked == i] = val

    scale = scale.view(torch.uint8).to(torch.int32)
    scale = (scale << 23).view(torch.float32)
    scale = scale.reshape(*x.shape[:-1], -1)
    scale = torch.stack([scale] * 32, dim=-1).reshape(*x_float.shape)

    return x_float * scale


def mxfp8_dequantize(x, scale):
    assert x.dtype == torch.float8_e4m3fn
    x_float = x.to(torch.float32)

    scale = scale.view(torch.uint8).to(torch.int32)
    scale = (scale << 23).view(torch.float32)
    scale = scale.reshape(*x.shape[:-1], -1)
    scale = torch.stack([scale] * 32, dim=-1).reshape(*x_float.shape)

    return x_float * scale


def reference_moe(
    roouting_logits,
    topk,
    num_experts,
    hidden_states,
    w13,
    bias13,
    w2,
    bias2,
    alpha,
    beta,
    limit,
    act_type,
    activation: str = "swiglu",
    use_interleaved_layout: bool = False,
):
    """
    Reference MoE implementation for accuracy testing.

    Args:
        activation: One of "swiglu", "silu", "relu2". Controls the activation
            function used after the first MLP.
        use_interleaved_layout: If True, uses interleaved gate/up layout
            (gate=x[..., ::2], up=x[..., 1::2]) as used by SWIGLUOAI.
            If False, uses chunked layout (gate, up = chunk(x, 2)) as used
            by standard swiglu/silu.
    """
    # renormalize routing
    experts = torch.topk(roouting_logits, k=topk, dim=-1, sorted=True)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices
    t = hidden_states.clone()
    # MLP #1
    mlp1_weight = w13[expert_indices, ...]
    mlp1_bias = bias13[expert_indices, ...]
    t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias

    # Apply activation
    if activation in ("swiglu", "silu"):
        if use_interleaved_layout:
            # SWIGLUOAI: interleaved gate/up layout
            t = swigluoai(t, alpha=alpha, limit=limit)
        else:
            # Standard swiglu/silu: chunked layout
            t = swiglu(t, alpha=alpha, beta=beta, limit=limit)
    elif activation == "relu2":
        # RELU2_NO_MUL: relu(x)^2
        t = torch.relu(t)
        t = t * t
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if act_type == "mxfp8":
        t_quantized, t_scale = mxfp8_quantize(
            t.to(torch.bfloat16), is_sf_swizzled_layout=False
        )
        t = mxfp8_dequantize(t_quantized, t_scale)
    # MLP #2
    mlp2_weight = w2[expert_indices, ...]
    mlp2_bias = bias2[expert_indices, ...]
    t = torch.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
    # Weighted sum of experts
    t = torch.einsum("bec,be->bc", t, expert_weights)
    assert t.shape == hidden_states.shape
    return t.to(torch.bfloat16)


def tg_mxfp4_moe(
    router_logits,
    topk,
    num_experts,
    intermediate_size,
    hidden_size,
    hidden_states,
    hidden_states_scale,
    w13_weight,
    w13_weight_scale,
    w13_bias,
    w2_weight,
    w2_weight_scale,
    w2_bias,
    act_type,
    alpha,
    beta,
    limit,
    transpose_optimized: bool = False,
) -> torch.Tensor:
    sf_block_size = 32
    assert (
        w13_weight.dim() == 3
        and w13_weight.shape[0] == num_experts
        and w13_weight.shape[1] == intermediate_size * 2
        and w13_weight.shape[2] == hidden_size // 2
    )
    assert (
        w13_weight_scale.dim() == 3
        and w13_weight_scale.shape[0] == num_experts
        and w13_weight_scale.shape[1] == intermediate_size * 2
        and w13_weight_scale.shape[2] == hidden_size // sf_block_size
    )
    assert (
        w2_weight.dim() == 3
        and w2_weight.shape[0] == num_experts
        and w2_weight.shape[1] == hidden_size
        and w2_weight.shape[2] == intermediate_size // 2
    )
    assert (
        w2_weight_scale.dim() == 3
        and w2_weight_scale.shape[1] == hidden_size
        and w2_weight_scale.shape[2] == intermediate_size // sf_block_size
    )
    assert (
        w13_bias.dim() == 2
        and w13_bias.shape[0] == num_experts
        and w13_bias.shape[1] == intermediate_size * 2
    )
    assert (
        w2_bias.dim() == 2
        and w2_bias.shape[0] == num_experts
        and w2_bias.shape[1] == hidden_size
    )

    # Swap w1 and w3 as the definition of
    # swiglu is different in the trtllm-gen
    w13_weight_scale_ = w13_weight_scale.clone()
    w13_weight_ = w13_weight.clone()
    w13_bias_ = w13_bias.clone()
    w13_weight[:, :intermediate_size, :].copy_(w13_weight_[:, intermediate_size:, :])
    w13_weight[:, intermediate_size:, :].copy_(w13_weight_[:, :intermediate_size, :])
    w13_weight_scale[:, :intermediate_size, :].copy_(
        w13_weight_scale_[:, intermediate_size:, :]
    )
    w13_weight_scale[:, intermediate_size:, :].copy_(
        w13_weight_scale_[:, :intermediate_size, :]
    )
    w13_bias[:, :intermediate_size].copy_(w13_bias_[:, intermediate_size:])
    w13_bias[:, intermediate_size:].copy_(w13_bias_[:, :intermediate_size])

    # Interleave the weights and scaling factors for activation
    w13_weight_interleaved = []
    w13_weight_scale_interleaved = []
    w13_bias_interleaved = []
    for i in range(num_experts):
        w13_weight_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_weight[i].clone())
        )
        w13_weight_scale_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_weight_scale[i].clone())
        )
        w13_bias_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_bias[i].clone().reshape(-1, 1))
        )
    w13_weight = torch.stack(w13_weight_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )
    w13_weight_scale = torch.stack(w13_weight_scale_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 32
    )
    w13_bias = torch.stack(w13_bias_interleaved).reshape(
        num_experts, 2 * intermediate_size
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_shuffled = []
    gemm2_scales_shuffled = []
    gemm1_bias_shuffled = []
    gemm2_bias_shuffled = []
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
    if transpose_optimized:
        for i in range(num_experts):
            # w13 weight shuffling
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_weight[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_shuffled.append(
                w13_weight[i]
                .view(torch.uint8)[permute_indices.to(w13_weight.device)]
                .contiguous()
            )
            # w13 scale shuffling
            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_weight_scale[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm1_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w13_weight_scale[i]
                    .view(torch.uint8)[permute_sf_indices.to(w13_weight_scale.device)]
                    .contiguous()
                )
            )
            # w13 bias shuffling
            permute_bias_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_bias[i].clone().reshape(-1, 1),
                epilogue_tile_m,
            )
            gemm1_bias_shuffled.append(
                w13_bias[i]
                .clone()
                .reshape(-1, 1)[permute_bias_indices.to(w13_bias.device)]
                .contiguous()
            )
            # w2 weight shuffling
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_weight[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_shuffled.append(
                w2_weight[i]
                .view(torch.uint8)[permute_indices.to(w2_weight.device)]
                .contiguous()
            )
            # w2 scale shuffling
            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_weight_scale[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w2_weight_scale[i]
                    .view(torch.uint8)[permute_sf_indices.to(w2_weight_scale.device)]
                    .contiguous()
                )
            )
            # w2 bias shuffling
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_bias[i].clone().reshape(-1, 1),
                epilogue_tile_m,
            )
            gemm2_bias_shuffled.append(
                w2_bias[i]
                .clone()
                .reshape(-1, 1)[permute_indices.to(w2_bias.device)]
                .contiguous()
            )

    else:
        for i in range(num_experts):
            gemm1_weights_shuffled.append(
                shuffle_matrix_a(w13_weight[i].view(torch.uint8), epilogue_tile_m)
            )
            gemm1_scales_shuffled.append(
                shuffle_matrix_sf_a(
                    w13_weight_scale[i].view(torch.uint8), epilogue_tile_m
                )
            )

            gemm2_weights_shuffled.append(
                shuffle_matrix_a(w2_weight[i].view(torch.uint8), epilogue_tile_m)
            )
            gemm2_scales_shuffled.append(
                shuffle_matrix_sf_a(
                    w2_weight_scale[i].view(torch.uint8), epilogue_tile_m
                )
            )
            gemm1_bias_shuffled.append(
                shuffle_matrix_a(w13_bias[i].reshape(-1, 1), epilogue_tile_m)
            )
            gemm2_bias_shuffled.append(
                shuffle_matrix_a(w2_bias[i].reshape(-1, 1), epilogue_tile_m)
            )

    w13_weight = torch.stack(gemm1_weights_shuffled)
    w13_weight_scale = (
        torch.stack(gemm1_scales_shuffled)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // sf_block_size)
        .view(torch.float8_e4m3fn)
    )
    w13_bias = torch.stack(gemm1_bias_shuffled).reshape(num_experts, -1)

    w2_weight = torch.stack(gemm2_weights_shuffled)
    w2_weight_scale = (
        torch.stack(gemm2_scales_shuffled)
        .reshape(num_experts, hidden_size, intermediate_size // sf_block_size)
        .view(torch.float8_e4m3fn)
    )
    w2_bias = torch.stack(gemm2_bias_shuffled).reshape(num_experts, -1)

    tg_result = trtllm_fp4_block_scale_moe(
        routing_logits=router_logits.to(torch.bfloat16),
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=w13_weight,
        gemm1_weights_scale=w13_weight_scale,
        gemm1_bias=w13_bias,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
        gemm2_weights=w2_weight,
        gemm2_weights_scale=w2_weight_scale,
        gemm2_bias=w2_bias,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=num_experts,
        top_k=topk,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,  # renormalize
        do_finalize=True,
    )[0]
    return tg_result


def check_accuracy(a, b, atol, rtol, percent):
    """Allow a mismatch percentage of 1 - percent."""
    if torch.any(torch.isnan(a)):
        raise Exception("NaN in reference output")
    if torch.any(torch.isnan(b)):
        raise Exception("NaN in actual output")
    if torch.any(torch.isinf(a)):
        raise Exception("Inf in reference output")
    if torch.any(torch.isinf(b)):
        raise Exception("Inf in actual output")
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    left = torch.abs(a - b)
    right = atol + rtol * torch.abs(b)
    count = torch.sum(left > right)
    mismatch_percent = count / a.numel()
    if mismatch_percent > 1 - percent:
        raise Exception(
            f"Mismatch percentage is {mismatch_percent:.4f} for rtol {rtol} "
            f"(threshold: {1 - percent:.4f})"
        )


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("num_tokens", [1, 128, 1024])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("alpha,beta,limit", [(1.0, 1.0, None), (1.702, 1.0, 7.0)])
@pytest.mark.parametrize("act_type", ["mxfp8", "bf16"])
@pytest.mark.parametrize("transpose_optimized", [False, True])
@pytest.mark.skipif(
    not TRTLLM_GEN_MXFP4_AVAILABLE,
    reason="nvidia gpu and compute capability sm100 is required for this test",
)
def test_trtllm_gen_mxfp4_fused_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    alpha: float,
    beta: float,
    limit: float | None,
    act_type: str,
    transpose_optimized: bool,
):
    seed = 42
    torch.manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda:0", dtype=torch.bfloat16
    )
    w13 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    bias13 = torch.randn(num_experts, intermediate_size * 2, device="cuda:0") * 10
    bias2 = torch.randn(num_experts, hidden_size, device="cuda:0") * 10
    router_logits = torch.rand(num_tokens, num_experts, dtype=torch.float32).cuda()

    w13, w13_scale = fp4_quantize(
        w13,
        torch.tensor(1.0, device="cuda:0"),
        32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,
    )
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, intermediate_size * 2, hidden_size // 32
    )
    w2, w2_scale = fp4_quantize(
        w2,
        torch.tensor(1.0, device="cuda:0"),
        32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,
    )
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 32
    )
    if act_type == "mxfp8":
        hidden_states, hidden_states_scale = mxfp8_quantize(
            hidden_states, is_sf_swizzled_layout=False
        )
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            *hidden_states.shape[:-1], -1
        )
    else:
        hidden_states_scale = None

    # reference result
    ref_result = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    w13_ref = mxfp4_dequantize(w13.clone(), w13_scale.clone())
    w2_ref = mxfp4_dequantize(w2.clone(), w2_scale.clone())
    bias13_ref = bias13
    bias2_ref = bias2
    if act_type == "mxfp8":
        hidden_states_ref = mxfp8_dequantize(hidden_states, hidden_states_scale).to(
            torch.float32
        )
    else:
        hidden_states_ref = hidden_states.to(torch.float32)
    # Process tokens in chunks of 32 to reduce memory usage
    chunk_size = 32
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_tokens)
        chunk_result = reference_moe(
            router_logits[start_idx:end_idx].to(torch.float32),
            topk,
            num_experts,
            hidden_states_ref[start_idx:end_idx],
            w13_ref,
            bias13_ref,
            w2_ref,
            bias2_ref,
            alpha,
            beta,
            limit,
            act_type,
            activation="swiglu",
            use_interleaved_layout=False,
        )
        ref_result[start_idx:end_idx].copy_(chunk_result)

    # trtllm-gen result
    if alpha is not None:
        alpha = torch.full((num_experts,), alpha, device=hidden_states.device)
    if limit is not None:
        limit = torch.full((num_experts,), limit, device=hidden_states.device)
    if beta is not None:
        beta = torch.full((num_experts,), beta, device=hidden_states.device)
    tg_result = tg_mxfp4_moe(
        router_logits=router_logits,
        topk=topk,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        w13_weight=w13,
        w13_weight_scale=w13_scale,
        w13_bias=bias13,
        w2_weight=w2,
        w2_weight_scale=w2_scale,
        w2_bias=bias2,
        act_type=act_type,
        alpha=alpha,
        beta=beta,
        limit=limit,
        transpose_optimized=transpose_optimized,
    )
    # relatively loose check since the mxfp4 quantization is less accurate
    check_accuracy(ref_result, tg_result, atol=0, rtol=0.3, percent=0.8)


def _interleave_scales_lastdim_by4(scales: torch.Tensor) -> torch.Tensor:
    """Interleave scales on the last dimension by groups of 4, matching
    the transformation in mxfp4.py's BF16 (Hopper) path."""
    s = scales.to(torch.uint8)
    s_shape = s.shape
    assert s_shape[-1] % 4 == 0
    s = s.reshape(*s_shape[:-1], s_shape[-1] // 4, 4)
    # Move the 4-group dimension before the row dimension
    permuted = s.permute(0, 2, 1, 3)
    # Merge the row dim with the 4-group dim
    return permuted.reshape(s_shape[0], s_shape[-1] // 4, s_shape[1] * 4)


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("alpha,beta,limit", [(1.0, 1.0, None), (1.702, 1.0, 7.0)])
@pytest.mark.skipif(
    not HOPPER_MXFP4_BF16_AVAILABLE,
    reason="nvidia gpu sm90 and flashinfer are required for this test",
)
def test_flashinfer_cutlass_mxfp4_fused_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    alpha: float,
    beta: float,
    limit: float | None,
):
    torch.manual_seed(42)
    device = "cuda:0"

    # Inputs
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    # Random MXFP4 weights and scales (uint8), contiguous [w1; w3]
    w13_q = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w13_scale = torch.randint(
        118,
        123,
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        device=device,
        dtype=torch.uint8,
    )

    w2_q = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2_scale = torch.randint(
        118,
        123,
        (num_experts, hidden_size, intermediate_size // 32),
        device=device,
        dtype=torch.uint8,
    )
    # Bias contiguous [b1; b3]
    bias13 = (
        torch.randn(
            num_experts, 2 * intermediate_size, device=device, dtype=torch.bfloat16
        )
        * 10
    )
    bias2 = (
        torch.randn(num_experts, hidden_size, device=device, dtype=torch.bfloat16) * 10
    )
    router_logits = torch.rand(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    w13_ref = mxfp4_dequantize(w13_q.clone(), w13_scale.clone()).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )
    w2_ref = mxfp4_dequantize(w2_q.clone(), w2_scale.clone()).reshape(
        num_experts, hidden_size, intermediate_size
    )
    ref = reference_moe(
        router_logits.to(torch.float32),
        topk,
        num_experts,
        hidden_states.to(torch.float32),
        w13_ref,
        bias13.to(torch.float32),
        w2_ref,
        bias2.to(torch.float32),
        alpha,
        beta,
        limit,
        "bf16",
        activation="swiglu",
        use_interleaved_layout=False,
    )

    from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

    # Swap halves to arrange as [w3; w1] (kernel expectation)
    w1_w, w3_w = torch.chunk(w13_q, 2, dim=1)
    w13_q_swapped = torch.cat([w3_w, w1_w], dim=1)

    b1, b3 = torch.chunk(bias13.to(torch.float32), 2, dim=-1)
    w13_b = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)

    w1_s, w3_s = torch.chunk(w13_scale, 2, dim=1)
    w13_s = torch.cat([w3_s, w1_s], dim=1)
    w13_s_inter = _interleave_scales_lastdim_by4(w13_s)
    w2_s_inter = _interleave_scales_lastdim_by4(w2_scale)

    routing_weights = torch.nn.functional.softmax(
        router_logits, dim=1, dtype=torch.float32
    )
    token_final_scales, token_selected_experts = torch.topk(
        routing_weights, topk, dim=-1
    )
    token_final_scales = token_final_scales / token_final_scales.sum(
        dim=-1, keepdim=True
    )
    token_selected_experts = token_selected_experts.to(torch.int).contiguous()

    out = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    if alpha is not None:
        alpha = torch.full((num_experts,), alpha, device=hidden_states.device)
    if beta is not None:
        beta = torch.full((num_experts,), beta, device=hidden_states.device)
    if limit is not None:
        limit = torch.full((num_experts,), limit, device=hidden_states.device)

    _ = flashinfer_cutlass_fused_moe(
        input=hidden_states,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        fc1_expert_weights=w13_q_swapped,
        fc2_expert_weights=w2_q,
        output_dtype=torch.bfloat16,
        output=out,
        quant_scales=[w13_s_inter.to(torch.uint8), w2_s_inter.to(torch.uint8)],
        fc1_expert_biases=w13_b,
        fc2_expert_biases=bias2.to(torch.bfloat16),
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        use_w4_group_scaling=True,
    )

    # Allow some mismatch due to MXFP4 quantization
    check_accuracy(ref, out, atol=0, rtol=0.3, percent=0.8)


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("alpha,beta,limit", [(1.0, 1.0, None), (1.702, 1.0, 7.0)])
@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer()
    ),
    reason="NVIDIA GPU sm100 and flashinfer are required for this test",
)
def test_flashinfer_cutlass_mxfp4_mxfp8_fused_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    alpha: float | None,
    beta: float | None,
    limit: float | None,
):
    torch.manual_seed(42)
    device = "cuda:0"

    # Inputs
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    # Float weights in w13 format [w1; w3]
    w13 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    # Bias contiguous [b1; b3]
    bias13 = (
        torch.randn(
            num_experts, 2 * intermediate_size, device=device, dtype=torch.bfloat16
        )
        * 10
    )
    bias2 = (
        torch.randn(num_experts, hidden_size, device=device, dtype=torch.bfloat16) * 10
    )
    router_logits = torch.rand(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    # Quantize weights to MXFP4 per expert (SM100 path)
    from flashinfer import mxfp4_quantize

    def quant_mxfp4_batches(a: torch.Tensor, e: int):
        qs, sfs = [], []
        for i in range(e):
            q, sf = mxfp4_quantize(a[i].cuda())
            qs.append(q)
            sfs.append(sf)
        return torch.stack(qs), torch.stack(sfs)

    def dequant_mxfp4_batches(mat_fp4: torch.Tensor, scale_tensor: torch.Tensor):
        num_batches = mat_fp4.size(0)
        scale_tensor = scale_tensor.view(num_batches, -1)
        from flashinfer import mxfp4_dequantize

        return torch.stack(
            [
                mxfp4_dequantize(mat_fp4[b, :, :], scale_tensor[b, :])
                for b in range(num_batches)
            ]
        )

    w13_q, w13_scale = quant_mxfp4_batches(w13, num_experts)
    w2_q, w2_scale = quant_mxfp4_batches(w2, num_experts)

    # Reference result using dequantized tensors and reference_moe
    w13_ref = (
        dequant_mxfp4_batches(
            w13_q.view(torch.uint8), w13_scale.view(torch.uint8).reshape(-1)
        )
        .to(torch.float32)
        .reshape(num_experts, 2 * intermediate_size, hidden_size)
        .to(device)
    )
    w2_ref = (
        dequant_mxfp4_batches(
            w2_q.view(torch.uint8), w2_scale.view(torch.uint8).reshape(-1)
        )
        .to(torch.float32)
        .reshape(num_experts, hidden_size, intermediate_size)
        .to(device)
    )

    # Quantize activations for SM100 path and dequantize for reference
    hidden_states_q, hidden_states_sf = mxfp8_quantize(hidden_states, True, 32)
    # Reference uses BF16 input but quantizes intermediate activation to MXFP8
    ref = reference_moe(
        router_logits.to(torch.float32),
        topk,
        num_experts,
        hidden_states.to(torch.float32),
        w13_ref,
        bias13.to(torch.float32),
        w2_ref,
        bias2.to(torch.float32),
        alpha,
        beta,
        limit,
        "mxfp8",
        activation="swiglu",
        use_interleaved_layout=False,
    )

    # Prepare inputs for FlashInfer CUTLASS fused MoE
    from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

    # Swap halves to arrange as [w3; w1] (kernel expectation)
    w1_w, w3_w = torch.chunk(w13_q, 2, dim=1)
    w13_q_swapped = torch.cat([w3_w, w1_w], dim=1)

    # Swap scales halves to match swapped weights
    s1, s3 = torch.chunk(w13_scale, 2, dim=1)
    w13_scale_swapped = torch.cat([s3, s1], dim=1)

    b1, b3 = torch.chunk(bias13.to(torch.float32), 2, dim=-1)
    w13_b = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)

    # Build routing for kernel
    routing_weights = torch.nn.functional.softmax(
        router_logits, dim=1, dtype=torch.float32
    )
    token_final_scales, token_selected_experts = torch.topk(
        routing_weights, topk, dim=-1
    )
    token_final_scales = token_final_scales / token_final_scales.sum(
        dim=-1, keepdim=True
    )
    token_selected_experts = token_selected_experts.to(torch.int).contiguous()

    out = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    if alpha is not None:
        alpha_t = torch.full((num_experts,), alpha, device=hidden_states.device)
    else:
        alpha_t = None
    if beta is not None:
        beta_t = torch.full((num_experts,), beta, device=hidden_states.device)
    else:
        beta_t = None
    if limit is not None:
        limit_t = torch.full((num_experts,), limit, device=hidden_states.device)
    else:
        limit_t = None

    # Quant scales for SM100 MXFP8+MXFP4 path
    fake_input_scale = torch.ones(num_experts, device=device)
    quant_scales = [
        w13_scale_swapped.view(torch.int32),
        fake_input_scale,
        w2_scale.view(torch.int32),
        fake_input_scale,
    ]

    _ = flashinfer_cutlass_fused_moe(
        input=hidden_states_q,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        fc1_expert_weights=w13_q_swapped.contiguous().view(torch.long),
        fc2_expert_weights=w2_q.contiguous().view(torch.long),
        output_dtype=torch.bfloat16,
        output=out,
        quant_scales=quant_scales,
        fc1_expert_biases=w13_b,
        fc2_expert_biases=bias2.to(torch.bfloat16),
        swiglu_alpha=alpha_t,
        swiglu_beta=beta_t,
        swiglu_limit=limit_t,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        use_mxfp8_act_scaling=True,
        input_sf=hidden_states_sf,
    )

    # Allow some mismatch due to MXFP4 quantization
    check_accuracy(ref, out, atol=0, rtol=0.3, percent=0.8)


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("is_gated", [True], ids=["gated"])
@pytest.mark.skipif(
    not TRTLLM_GEN_MXFP8_AVAILABLE,
    reason="nvidia gpu and compute capability sm100 is required for this test",
)
def test_trtllm_gen_mxfp8_block_scale_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    is_gated: bool,
):
    torch.manual_seed(42)
    device = "cuda:0"

    inter_size = intermediate_size * (2 if is_gated else 1)

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 20
    )
    w13 = (
        torch.randn(
            num_experts,
            inter_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 20
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 20
    )
    router_logits = torch.rand(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    router_logits_kernel = router_logits.to(torch.bfloat16)

    # Quantize weights to MXFP8 and normalize scales to [E, M, K//32].
    w13_q, w13_scale = mxfp8_quantize(w13, is_sf_swizzled_layout=False)
    w2_q, w2_scale = mxfp8_quantize(w2, is_sf_swizzled_layout=False)
    if w13_scale.ndim == 1:
        w13_scale = w13_scale.view(
            num_experts,
            inter_size,
            hidden_size // 32,
        )
    if w2_scale.ndim == 1:
        w2_scale = w2_scale.view(num_experts, hidden_size, intermediate_size // 32)

    # Quantize activations to MXFP8.
    hidden_states_q, hidden_states_scale = mxfp8_quantize(
        hidden_states, is_sf_swizzled_layout=False
    )
    if hidden_states_scale.ndim == 1:
        hidden_states_scale = hidden_states_scale.view(num_tokens, hidden_size // 32)

    # Reference output using dequantized tensors + MXFP8 intermediate quantization.
    w13_ref = mxfp8_dequantize(w13_q, w13_scale).to(torch.float32)
    w2_ref = mxfp8_dequantize(w2_q, w2_scale).to(torch.float32)
    hidden_states_ref = mxfp8_dequantize(hidden_states_q, hidden_states_scale).to(
        torch.float32
    )
    bias13 = torch.zeros(
        num_experts,
        intermediate_size * (2 if is_gated else 1),
        device=device,
    )
    bias2 = torch.zeros(num_experts, hidden_size, device=device)
    ref = reference_moe(
        router_logits_kernel.to(torch.float32),
        topk,
        num_experts,
        hidden_states_ref,
        w13_ref,
        bias13,
        w2_ref,
        bias2,
        alpha=1.0,
        beta=0.0,
        limit=None,
        act_type="mxfp8",
        activation="swiglu" if is_gated else "relu2",
        use_interleaved_layout=False,
    )

    # Shuffle weights/scales with the same indexed layout used by TRTLLM kernels.
    epilogue_tile_m = 128
    gemm1_weights_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_shuffled = []
    gemm2_scales_shuffled = []
    for i in range(num_experts):
        w13_rows = intermediate_size * (2 if is_gated else 1)
        w13_interleaved = w13_q[i].clone().reshape(w13_rows, -1)
        w13_scale_interleaved = w13_scale[i].clone().reshape(w13_rows, -1)
        if is_gated:
            w13_interleaved = reorder_rows_for_gated_act_gemm(w13_interleaved)
            w13_scale_interleaved = reorder_rows_for_gated_act_gemm(
                w13_scale_interleaved
            )
        gemm1_weights_shuffled.append(
            shuffle_matrix_a(w13_interleaved.view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(w13_q.dtype)
        )
        gemm2_weights_shuffled.append(
            shuffle_matrix_a(w2_q[i].view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(w2_q.dtype)
        )

        gemm1_scales_shuffled.append(
            shuffle_matrix_sf_a(
                w13_scale_interleaved.view(torch.uint8).reshape(w13_rows, -1),
                epilogue_tile_m,
            )
            .contiguous()
            .view(w13_scale.dtype)
        )
        gemm2_scales_shuffled.append(
            shuffle_matrix_sf_a(
                w2_scale[i].view(torch.uint8).reshape(hidden_size, -1), epilogue_tile_m
            )
            .contiguous()
            .view(w2_scale.dtype)
        )

    out = trtllm_fp8_block_scale_moe(
        routing_logits=router_logits_kernel,
        routing_bias=None,
        hidden_states=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=torch.stack(gemm1_weights_shuffled),
        gemm1_weights_scale=torch.stack(gemm1_scales_shuffled),
        gemm2_weights=torch.stack(gemm2_weights_shuffled),
        gemm2_weights_scale=torch.stack(gemm2_scales_shuffled),
        num_experts=num_experts,
        top_k=topk,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,  # renormalize routing
        use_shuffled_weight=True,
        weight_layout=0,  # MajorK
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
    )

    # Block-scale MXFP8 kernels are approximate; require majority close.
    check_accuracy(ref, out, atol=0.1, rtol=0.85, percent=0.8)


# -----------------------------------------------------------------------------
# ROCm Oracle-based kernel execution tests
# -----------------------------------------------------------------------------
# TODO: Further tighten the accuracy threshold.
# - More accurate ref moe to include activation quantization
# - Check aiter kernel accuracy. E.g., quant / dequant details.
ROCM_BACKEND_CONFIGS = {
    "TRITON": {
        "activation": "SWIGLUOAI",
        "rtol": 0.3,
        "percent": 0.95,
        "requires_aiter": False,
        "requires_gfx950": False,
    },
    "TRITON_UNFUSED": {
        "activation": "SWIGLUOAI",
        "rtol": 0.3,
        "percent": 0.95,
        "requires_aiter": False,
        "requires_gfx950": False,
    },
    "AITER_MXFP4_BF16": {
        "activation": "SILU",
        "rtol": 1.0,
        "percent": 0.7,
        "requires_aiter": True,
        "requires_gfx950": True,
    },
    "AITER_MXFP4_FP8": {
        "activation": "SWIGLUOAI",
        "rtol": 0.5,
        "percent": 0.9,
        "requires_aiter": True,
        "requires_gfx950": True,
    },
}


@pytest.mark.parametrize("backend_name", list(ROCM_BACKEND_CONFIGS.keys()))
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_tokens,hidden_size,intermediate_size", [(16, 256, 256)])
@pytest.mark.skipif(
    not ROCM_AVAILABLE,
    reason="ROCm is required for this test",
)
@torch.inference_mode()
def test_rocm_mxfp4_moe_oracle(
    backend_name: str,
    topk: int,
    num_experts: int,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
):
    """
    Test ROCm MXFP4 MoE using oracle functions.

    This test validates that the oracle functions work end-to-end:
    - select_mxfp4_moe_backend() selects a valid backend
    - convert_to_mxfp4_moe_kernel_format() converts weights without error
    - make_mxfp4_moe_quant_config() builds a valid quant config
    - make_mxfp4_moe_kernel() creates a kernel that runs without error
    - The kernel output is within accuracy tolerance of reference
    """
    config = ROCM_BACKEND_CONFIGS[backend_name]

    # Check platform requirements
    if not ROCM_TRITON_KERNELS_AVAILABLE:
        pytest.skip("triton_kernels required for quantization")
    if config["requires_aiter"] and not ROCM_AITER_AVAILABLE:
        pytest.skip(f"Backend {backend_name} requires AITER")
    if config["requires_gfx950"] and not ROCM_GFX950:
        pytest.skip(f"Backend {backend_name} requires GFX950")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        backend_to_kernel_cls,
        convert_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    # Initialize workspace manager (needed for modular kernels)
    init_workspace_manager(torch.accelerator.current_device_index())

    # Map string to enum
    backend = Mxfp4MoeBackend[backend_name]

    # Get experts class from oracle
    experts_cls_list = backend_to_kernel_cls(backend)
    if experts_cls_list is None or len(experts_cls_list) == 0:
        pytest.skip(f"Backend {backend_name} not available")

    # Use first experts class
    experts_cls = experts_cls_list[0]

    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda:0"

    # Create MoE config with Renormalize routing (required by monolithic kernels)
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation[config["activation"]],
        in_dtype=dtype,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
    )

    # Create float weights in checkpoint format:
    # w13: [num_experts, 2*intermediate_size, hidden_size]
    # w2: [num_experts, hidden_size, intermediate_size]
    w13_float = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device
    )
    w2_float = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )

    # dynamic_mxfp4_quant expects 2D input, so reshape 3D weights
    # w13: [E, 2*I, H] -> [E*2*I, H] -> quantize -> [E, 2*I, H//2]
    # w2: [E, H, I] -> [E*H, I] -> quantize -> [E, H, I//2]
    w13_2d = w13_float.reshape(-1, hidden_size)
    w13_quant_2d, w13_scale_2d = dynamic_mxfp4_quant(w13_2d)
    w13_quant = w13_quant_2d.reshape(num_experts, 2 * intermediate_size, -1)
    w13_scale = w13_scale_2d.reshape(num_experts, 2 * intermediate_size, -1)

    w2_2d = w2_float.reshape(-1, intermediate_size)
    w2_quant_2d, w2_scale_2d = dynamic_mxfp4_quant(w2_2d)
    w2_quant = w2_quant_2d.reshape(num_experts, hidden_size, -1)
    w2_scale = w2_scale_2d.reshape(num_experts, hidden_size, -1)

    w13_bias = torch.randn(
        num_experts, 2 * intermediate_size, dtype=dtype, device=device
    )
    w2_bias = torch.randn(num_experts, hidden_size, dtype=dtype, device=device)

    # Create static input scales for W4A8 backend (AITER_MXFP4_FP8)
    w13_input_scale: torch.Tensor | None = None
    w2_input_scale: torch.Tensor | None = None
    if backend_name == "AITER_MXFP4_FP8":
        # Static FP8 scales: one scale per expert
        w13_input_scale = torch.ones(num_experts, dtype=torch.float32, device=device)
        w2_input_scale = torch.ones(num_experts, dtype=torch.float32, device=device)

    # Create mock layer for oracle functions
    class MockLayer:
        w13_weight: torch.Tensor
        w2_weight: torch.Tensor
        w13_weight_scale: torch.Tensor
        w2_weight_scale: torch.Tensor
        w13_input_scale: torch.Tensor | None
        w2_input_scale: torch.Tensor | None

    layer = MockLayer()
    layer.w13_weight = w13_quant
    layer.w2_weight = w2_quant
    layer.w13_weight_scale = w13_scale
    layer.w2_weight_scale = w2_scale
    layer.w13_input_scale = w13_input_scale
    layer.w2_input_scale = w2_input_scale

    # Convert weights using oracle
    w13_conv, w2_conv, w13_scale_conv, w2_scale_conv, w13_bias_conv, w2_bias_conv = (
        convert_to_mxfp4_moe_kernel_format(
            mxfp4_backend=backend,
            layer=layer,  # type: ignore[arg-type]
            w13_weight=w13_quant,
            w2_weight=w2_quant,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
        )
    )

    # Build quant config using oracle
    quant_config = make_mxfp4_moe_quant_config(
        mxfp4_backend=backend,
        w1_scale=w13_scale_conv,
        w2_scale=w2_scale_conv,
        w1_bias=w13_bias_conv,
        w2_bias=w2_bias_conv,
        a1_scale=w13_input_scale,
        a2_scale=w2_input_scale,
    )

    # Select activation based on backend
    activation_name = str(config["activation"])
    activation = MoEActivation[activation_name]

    # Build kernel using oracle
    assert quant_config is not None, "Failed to create quant config"
    with set_current_vllm_config(VllmConfig()):
        kernel = make_mxfp4_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=moe_config,
            mxfp4_backend=backend,
            experts_cls=experts_cls,
            routing_tables=None,
            shared_experts=None,
        )

        # Create inputs
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        router_logits = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device=device
        )
        topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1, sorted=True)
        topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

        # Run kernel - use appropriate method based on impl type
        if kernel.is_monolithic:
            # Monolithic impl uses router_logits
            out = kernel.apply_monolithic(
                hidden_states=x,
                w1=w13_conv,
                w2=w2_conv,
                router_logits=router_logits,
                activation=activation,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )
        else:
            # Modular impl uses topk_weights and topk_ids
            out = kernel.apply(
                hidden_states=x,
                w1=w13_conv,
                w2=w2_conv,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

    # Verify output is valid (no NaN/Inf) and has expected shape
    assert out.shape == (num_tokens, hidden_size), f"Unexpected shape: {out.shape}"
    assert not torch.any(torch.isnan(out)), "Output contains NaN"
    assert not torch.any(torch.isinf(out)), "Output contains Inf"

    # Verify output has reasonable magnitude (not all zeros)
    assert out.abs().max() > 0.01, "Output is effectively zero"

    # Dequantize weights for reference computation
    w13_dq = upcast_from_mxfp(
        w13_quant.view(torch.uint8), w13_scale, torch.bfloat16, axis=-1
    )
    w2_dq = upcast_from_mxfp(
        w2_quant.view(torch.uint8), w2_scale, torch.bfloat16, axis=-1
    )

    # Determine activation type and layout
    # SWIGLUOAI uses interleaved layout (gate/up alternating)
    # SILU uses chunked layout (first half gate, second half up)
    use_interleaved = activation == MoEActivation.SWIGLUOAI
    if activation in [MoEActivation.SWIGLUOAI, MoEActivation.SILU]:
        act_name = "swiglu"
    else:
        act_name = "relu2"

    ref = reference_moe(
        router_logits,
        topk,
        num_experts,
        x.to(torch.float32),
        w13_dq.to(torch.float32),
        w13_bias.to(torch.float32),
        w2_dq.to(torch.float32),
        w2_bias.to(torch.float32),
        alpha=1.702 if activation == MoEActivation.SWIGLUOAI else 1.0,
        beta=1.0 if activation == MoEActivation.SWIGLUOAI else 0.0,
        limit=7.0 if activation == MoEActivation.SWIGLUOAI else None,
        act_type="bf16",
        activation=act_name,
        use_interleaved_layout=use_interleaved,
    )

    # Compute and print accuracy statistics
    diff = (ref.float() - out.float()).abs()
    rel_diff = diff / (ref.float().abs() + 1e-6)

    print(f"\n[{backend_name}] Accuracy statistics:")
    print(
        f"  Reference: min={ref.min():.4f}, max={ref.max():.4f}, mean={ref.mean():.4f}"
    )
    print(
        f"  Output:    min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}"
    )
    print(
        f"  Abs diff:  min={diff.min():.4f}, max={diff.max():.4f}, "
        f"mean={diff.mean():.4f}"
    )
    print(
        f"  Rel diff:  min={rel_diff.min():.4f}, max={rel_diff.max():.4f}, "
        f"mean={rel_diff.mean():.4f}"
    )

    # Check what percentage of values are within various tolerances
    for rtol in [0.1, 0.5, 1.0, 2.0]:
        within_tol = (diff <= rtol * out.float().abs()).float().mean()
        print(f"  Within rtol={rtol}: {within_tol * 100:.1f}%")

    # Check accuracy using per-backend thresholds
    check_accuracy(ref, out, atol=0.1, rtol=config["rtol"], percent=config["percent"])
