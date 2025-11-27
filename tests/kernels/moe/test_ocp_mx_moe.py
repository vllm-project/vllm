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
    current_platform.is_cuda() and current_platform.is_device_capability(100)
)

HOPPER_MXFP4_BF16_AVAILABLE = (
    current_platform.is_cuda()
    and current_platform.is_device_capability(90)
    and has_flashinfer()
)

if TRTLLM_GEN_MXFP4_AVAILABLE:
    from flashinfer import (
        fp4_quantize,
        mxfp8_quantize,
        next_positive_power_of_2,
        reorder_rows_for_gated_act_gemm,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        trtllm_fp4_block_scale_moe,
    )
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache


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
    if torch.cuda.device_count() < model_case.tp:
        pytest.skip(
            f"This test requires >={model_case.tp} gpus, got only "
            f"{torch.cuda.device_count()}"
        )

    # `cuda_graph_sizes=[16]` to reduce load time.
    with vllm_runner(
        model_case.model_id,
        tensor_parallel_size=model_case.tp,
        load_format="dummy",
        cuda_graph_sizes=[16],
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
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + beta)


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
):
    # renormalize routing
    experts = torch.topk(roouting_logits, k=topk, dim=-1, sorted=True)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices
    t = hidden_states.clone()
    # MLP #1
    mlp1_weight = w13[expert_indices, ...]
    mlp1_bias = bias13[expert_indices, ...]
    t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
    t = swiglu(t, alpha=alpha, beta=beta, limit=limit)

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


def get_tile_tokens_dim(x: torch.Tensor, top_k: int, num_experts: int):
    # Number of tokens in the input tensor.
    num_tokens = x.shape[0]
    # Factor to account for the imbalance of the experts.
    # factor equals to the
    # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
    # - 1.0 means perfect expert distribution.
    # - > 1.0 means some experts have more
    #     tokens than the perfect distribution.
    # - < 1.0 does not make sense.
    imbalance_factor = 1.3
    # Calculate the number of tokens per expert
    # assuming perfect distribution.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # Apply the imbalance factor.
    num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile
    # as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


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
        tile_tokens_dim=get_tile_tokens_dim(hidden_states, topk, num_experts),
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
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(-1)
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
        router_logits,
        topk,
        num_experts,
        intermediate_size,
        hidden_size,
        hidden_states,
        hidden_states_scale,
        w13,
        w13_scale,
        bias13,
        w2,
        w2_scale,
        bias2,
        act_type,
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
        and current_platform.is_device_capability(100)
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
