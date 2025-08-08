# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
from flashinfer import (fp4_quantize, mxfp8_quantize, next_positive_power_of_2,
                        reorder_rows_for_gated_act_gemm, shuffle_matrix_a,
                        shuffle_matrix_sf_a, trtllm_fp4_block_scale_moe)
from packaging import version

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec(
    "quark") is not None and version.parse(
        importlib.metadata.version("amd-quark")) >= version.parse('0.8.99')


@dataclass
class ModelCase:
    model_id: str
    tp: int


@pytest.mark.parametrize('model_case', [
    ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4", tp=1),
    ModelCase("fxmarty/deepseek_r1_3_layers_mxfp4", tp=8),
    ModelCase("fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4", tp=1)
])
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE,
                    reason="amd-quark>=0.9 is not available")
def test_mxfp4_loading_and_execution_moe(vllm_runner, model_case: ModelCase):
    if torch.cuda.device_count() < model_case.tp:
        pytest.skip(f"This test requires >={model_case.tp} gpus, got only "
                    f"{torch.cuda.device_count()}")

    with vllm_runner(model_case.model_id,
                     tensor_parallel_size=model_case.tp,
                     load_format="dummy") as llm:

        # TODO: llm.apply_model(check_model) currently relies on V0 internals.
        # Re-enable this later.
        # def check_model(model):
        #     layer = model.model.layers[0]

        #     qkv_proj = layer.self_attn.qkv_proj

        #     assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
        #     assert isinstance(qkv_proj.scheme, QuarkW4A4MXFP4)

        #     assert isinstance(layer.mlp.experts.quant_method,
        #                       QuarkW4A4MXFp4MoEMethod)

        # if model_case.model_id == "fxmarty/qwen_1.5-moe-a2.7b-mxfp4":
        #     llm.apply_model(check_model)

        output = llm.generate_greedy("Today I am in the French Alps and",
                                     max_tokens=20)
        assert output


def swiglu(x, alpha: float = 1.702, limit: Optional[float] = None):
    # Note we add an extra bias of 1 to the linear layer
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


fp4_lookup_table = [
    0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6
]


def mxfp4_dequantize(x, scale):
    assert x.dtype == torch.uint8
    x = x.view(torch.uint8).to(torch.int32)
    x_unpacked = torch.zeros(*x.shape[:-1],
                             x.shape[-1] * 2,
                             dtype=torch.int32,
                             device=x.device)
    x_unpacked[..., 0::2].copy_(x & 0xF)
    x_unpacked[..., 1::2].copy_((x >> 4) & 0xF)

    x_float = torch.zeros(x_unpacked.shape,
                          dtype=torch.float32,
                          device=x.device)
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
    w2,
    alpha,
    limit,
    act_type,
):
    experts = torch.topk(roouting_logits, k=topk, dim=-1, sorted=True)
    expert_weights = torch.nn.functional.softmax(experts.values,
                                                 dim=1).to(hidden_states.dtype)
    expert_indices = experts.indices
    t = hidden_states.clone()
    # MLP #1
    mlp1_weight = w13[expert_indices, ...]
    # mlp1_bias = w1_bias[topk_ids, ...]
    t = torch.einsum("beck,bk->bec", mlp1_weight, t)  # + mlp1_bias
    t = swiglu(t, alpha=alpha, limit=limit)

    if act_type == 'mxfp8':
        t_quantized, t_scale = mxfp8_quantize(t, is_sf_swizzled_layout=False)
        t = mxfp8_dequantize(t_quantized, t_scale).to(torch.bfloat16)

    # MLP #2
    mlp2_weight = w2[expert_indices, ...]
    # mlp2_bias = w2_bias[topk_ids, ...]
    t = torch.einsum("beck,bek->bec", mlp2_weight, t)  # + mlp2_bias

    # Weighted sum of experts
    t = torch.einsum("bec,be->bc", t, expert_weights)
    return t


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
    w2_weight,
    w2_weight_scale,
    act_type,
    alpha,
    limit,
) -> torch.Tensor:
    sf_block_size = 32
    assert (w13_weight.dim() == 3 and w13_weight.shape[0] == num_experts
            and w13_weight.shape[1] == intermediate_size * 2
            and w13_weight.shape[2] == hidden_size // 2)
    assert (w13_weight_scale.dim() == 3
            and w13_weight_scale.shape[0] == num_experts
            and w13_weight_scale.shape[1] == intermediate_size * 2
            and w13_weight_scale.shape[2] == hidden_size // sf_block_size)
    assert (w2_weight.dim() == 3 and w2_weight.shape[0] == num_experts
            and w2_weight.shape[1] == hidden_size
            and w2_weight.shape[2] == intermediate_size // 2)
    assert (w2_weight_scale.dim() == 3
            and w2_weight_scale.shape[1] == hidden_size
            and w2_weight_scale.shape[2] == intermediate_size // sf_block_size)

    # Swap w1 and w3 as the defenition of
    # swiglu is different in the trtllm-gen
    w13_weight_scale_ = w13_weight_scale.clone()
    w13_weight_ = w13_weight.clone()
    w13_weight[:, :intermediate_size, :].copy_(
        w13_weight_[:, intermediate_size:, :])
    w13_weight[:, intermediate_size:, :].copy_(
        w13_weight_[:, :intermediate_size, :])
    w13_weight_scale[:, :intermediate_size, :].copy_(
        w13_weight_scale_[:, intermediate_size:, :])
    w13_weight_scale[:, intermediate_size:, :].copy_(
        w13_weight_scale_[:, :intermediate_size, :])

    # Interleave the weights and scaling factors for activation
    w13_weight_interleaved = []
    w13_weight_scale_interleaved = []
    for i in range(num_experts):
        w13_weight_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_weight[i].clone()))
        w13_weight_scale_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_weight_scale[i].clone()))
    w13_weight = torch.stack(w13_weight_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2)
    w13_weight_scale = torch.stack(w13_weight_scale_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 32)

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_mxfp4_shuffled = []
    gemm1_scales_mxfp4_shuffled = []
    gemm2_weights_mxfp4_shuffled = []
    gemm2_scales_mxfp4_shuffled = []
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
    for i in range(num_experts):
        gemm1_weights_mxfp4_shuffled.append(
            shuffle_matrix_a(w13_weight[i].view(torch.uint8), epilogue_tile_m))
        gemm1_scales_mxfp4_shuffled.append(
            shuffle_matrix_sf_a(w13_weight_scale[i].view(torch.uint8),
                                epilogue_tile_m))

        gemm2_weights_mxfp4_shuffled.append(
            shuffle_matrix_a(w2_weight[i].view(torch.uint8), epilogue_tile_m))
        gemm2_scales_mxfp4_shuffled.append(
            shuffle_matrix_sf_a(w2_weight_scale[i].view(torch.uint8),
                                epilogue_tile_m))

    w13_weight = torch.stack(gemm1_weights_mxfp4_shuffled)
    w13_weight_scale = torch.stack(gemm1_scales_mxfp4_shuffled).reshape(
        num_experts, 2 * intermediate_size,
        hidden_size // sf_block_size).view(torch.float8_e4m3fn)

    w2_weight = torch.stack(gemm2_weights_mxfp4_shuffled)
    w2_weight_scale = torch.stack(gemm2_scales_mxfp4_shuffled).reshape(
        num_experts, hidden_size,
        intermediate_size // sf_block_size).view(torch.float8_e4m3fn)

    tg_result = trtllm_fp4_block_scale_moe(
        routing_logits=router_logits.to(torch.bfloat16),
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=w13_weight,
        gemm1_weights_scale=w13_weight_scale,
        gemm1_bias=None,
        gemm1_alpha=alpha,
        gemm1_beta=None,
        gemm1_clamp_limit=limit,
        gemm2_weights=w2_weight,
        gemm2_weights_scale=w2_weight_scale,
        gemm2_bias=None,
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
        do_finalize=True)[0]
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
            f"(threshold: {1-percent:.4f})")


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("num_tokens", [1, 128, 1024])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("alpha,limit", [(1.0, None), (1.702, 7.0)])
@pytest.mark.parametrize("act_type", ['mxfp8', 'bf16'])
def test_trtllm_gen_mxfp4_fused_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    alpha: Optional[float],
    limit: Optional[float],
    act_type: str,
):
    seed = 42
    torch.manual_seed(seed)
    hidden_states = torch.randn(num_tokens,
                                hidden_size,
                                device="cuda:0",
                                dtype=torch.bfloat16)
    w13 = (torch.randn(num_experts,
                       intermediate_size * 2,
                       hidden_size,
                       device="cuda:0",
                       dtype=torch.bfloat16))
    w2 = (torch.randn(num_experts,
                      hidden_size,
                      intermediate_size,
                      device="cuda:0",
                      dtype=torch.bfloat16))
    router_logits = torch.rand(num_tokens, num_experts,
                               dtype=torch.float32).cuda()

    w13, w13_scale = fp4_quantize(w13,
                                  torch.tensor(1.0, device="cuda:0"),
                                  32,
                                  sf_use_ue8m0=True,
                                  is_sf_swizzled_layout=False)
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, intermediate_size * 2, hidden_size // 32)
    w2, w2_scale = fp4_quantize(w2,
                                torch.tensor(1.0, device="cuda:0"),
                                32,
                                sf_use_ue8m0=True,
                                is_sf_swizzled_layout=False)
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 32)
    if act_type == 'mxfp8':
        hidden_states, hidden_states_scale = mxfp8_quantize(
            hidden_states, is_sf_swizzled_layout=False)
        hidden_states_scale = hidden_states_scale.view(
            torch.float8_e4m3fn).reshape(-1)
    else:
        hidden_states_scale = None

    # reference result
    ref_result = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    w13_ref = mxfp4_dequantize(w13, w13_scale).to(torch.bfloat16)
    w2_ref = mxfp4_dequantize(w2, w2_scale).to(torch.bfloat16)
    if act_type == 'mxfp8':
        hidden_states_ref = mxfp8_dequantize(
            hidden_states, hidden_states_scale).to(torch.bfloat16)
    else:
        hidden_states_ref = hidden_states
    # Process tokens in chunks of 32 to reduce memory usage
    chunk_size = 32
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_tokens)
        chunk_result = reference_moe(
            router_logits[start_idx:end_idx],
            topk,
            num_experts,
            hidden_states_ref[start_idx:end_idx],
            w13_ref,
            w2_ref,
            alpha,
            limit,
            act_type,
        )
        ref_result[start_idx:end_idx].copy_(chunk_result)

    # trtllm-gen result
    if alpha is not None:
        alpha = torch.full((num_experts, ), alpha, device=hidden_states.device)
    if limit is not None:
        limit = torch.full((num_experts, ), limit, device=hidden_states.device)
    tg_result = tg_mxfp4_moe(router_logits,
                             topk,
                             num_experts,
                             intermediate_size,
                             hidden_size,
                             hidden_states,
                             hidden_states_scale,
                             w13,
                             w13_scale,
                             w2,
                             w2_scale,
                             act_type,
                             alpha=alpha,
                             limit=limit)

    # relatively loose check since the mxfp4 quantization is less accurate
    check_accuracy(ref_result, tg_result, atol=0, rtol=0.8, percent=0.8)


if __name__ == "__main__":
    torch.set_printoptions(threshold=1000, sci_mode=False, precision=3)
    test_trtllm_gen_mxfp4_fused_moe(
        topk=4,
        num_experts=32,
        num_tokens=128,
        intermediate_size=4096,
        hidden_size=4096,
        act_type='mxfp8',
    )
