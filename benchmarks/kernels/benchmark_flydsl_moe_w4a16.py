# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 FlyDSL Project Contributors

import json
import os

import torch
from aiter.test_common import run_perftest

from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_flydsl_moe import fused_flydsl_moe
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a16_flydsl,
)
from vllm.platforms import current_platform

RoutingBuffers = tuple[
    torch.Tensor,  # sorted_token_ids
    torch.Tensor,  # sorted_weights
    torch.Tensor,  # sorted_expert_ids
    torch.Tensor,  # num_valid_ids (shape [1], i32)
    int,  # sorted_size
    int,  # blocks
]

MODEL_PARAMS_TO_TUNE = [
    # (num_experts, inter_dim, hidden_size, topk)
    (384, 256, 7168, 8),  # Kimi K2.5 TP=8
    (384, 512, 7168, 8),  # Kimi K2.5 TP=4
]

NUM_TOKENS_TO_TUNE = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]

TILE_M_SEARCH_SPACE = [16, 32, 64, 128, 256]
TILE_N_SEARCH_SPACE = [16, 32, 64, 128, 256]
TILE_K_SEARCH_SPACE = [16, 32, 64, 128, 256, 512]
TILE_N2_SEARCH_SPACE = [16, 32, 64, 128, 256]
TILE_K2_SEARCH_SPACE = [16, 32, 64, 128, 256, 512]

TILE_CONFIGS = []
for tile_m in TILE_M_SEARCH_SPACE:
    for tile_n in TILE_N_SEARCH_SPACE:
        for tile_k in TILE_K_SEARCH_SPACE:
            for tile_n2 in TILE_N2_SEARCH_SPACE:
                for tile_k2 in TILE_K2_SEARCH_SPACE:
                    TILE_CONFIGS.append(
                        {
                            "tile_m": tile_m,
                            "tile_n": tile_n,
                            "tile_k": tile_k,
                            "tile_n2": tile_n2,
                            "tile_k2": tile_k2,
                        }
                    )


def tune_flydsl_moe_w4a16(
    device: str = "cuda", num_iters: int = 100, num_warmup: int = 10
):
    packed_factor = 8
    w13_num_shards = 2
    params_dtype = torch.bfloat16
    group_size = 32
    scale_factor = 0.01

    for model_params in MODEL_PARAMS_TO_TUNE:
        num_experts = model_params[0]
        inter_dim = model_params[1]
        hidden_size = model_params[2]
        topk = model_params[3]
        print(
            f"\nTuning: num_experts={num_experts}, inter_dim={inter_dim}, "
            f"hidden_size={hidden_size}, topk={topk}...\n"
        )

        w2_scales_size = inter_dim
        num_groups_w2 = w2_scales_size // group_size
        num_groups_w13 = hidden_size // group_size

        w13_weight = torch.randint(
            0,
            255,
            (num_experts, hidden_size // packed_factor, w13_num_shards * inter_dim),
            dtype=torch.int32,
            device=device,
        )

        w2_weight = torch.randint(
            0,
            255,
            (num_experts, inter_dim // packed_factor, hidden_size),
            dtype=torch.int32,
            device=device,
        )
        w13_scale = scale_factor * torch.randn(
            num_experts,
            num_groups_w13,
            w13_num_shards * inter_dim,
            dtype=params_dtype,
            device=device,
        )
        w2_scale = scale_factor * torch.randn(
            num_experts, num_groups_w2, hidden_size, dtype=params_dtype, device=device
        )

        w13 = w13_weight
        w13 = compressed_tensors_moe_w4a16_flydsl._gptq_int32_to_flydsl_packed(w13)
        w13 = w13.view(-1).contiguous()

        w2 = w2_weight
        w2 = compressed_tensors_moe_w4a16_flydsl._gptq_int32_to_flydsl_packed(w2)
        w2 = w2.view(-1).contiguous()

        w13_scale_flydsl = w13_scale
        w2_scale_flydsl = w2_scale

        if group_size > 0 and w13_scale.dim() == 3 and w13_scale.shape[1] > 1:
            E, G, N = w13_scale.shape
            w13_scale_flydsl = (
                w13_scale_flydsl.view(E, G // 2, 2, N)
                .permute(0, 1, 3, 2)
                .contiguous()
                .view(-1)
                .contiguous()
            )
        elif w13_scale.dim() == 3 and w13_scale.shape[1] == 1:
            w13_scale_flydsl = w13_scale_flydsl.squeeze(1)

        if group_size > 0 and w2_scale.dim() == 3 and w2_scale.shape[1] > 1:
            E, G, N = w2_scale.shape
            w2_scale_flydsl = (
                w2_scale_flydsl.view(E, G // 2, 2, N)
                .permute(0, 1, 3, 2)
                .contiguous()
                .view(-1)
                .contiguous()
            )
        elif w2_scale.dim() == 3 and w2_scale.shape[1] == 1:
            w2_scale_flydsl = w2_scale_flydsl.squeeze(1)

        w13_scale_flydsl = w13_scale_flydsl.contiguous()
        w2_scale_flydsl = w2_scale_flydsl.contiguous()

        w13.is_shuffled = True
        w2.is_shuffled = True

        w13_weight_scale = w13_scale.transpose(1, 2).contiguous()
        w2_weight_scale = w2_scale.transpose(1, 2).contiguous()
        w13_weight_packed = w13_weight.transpose(1, 2).contiguous().view(torch.uint8)
        w2_weight_packed = w2_weight.transpose(1, 2).contiguous().view(torch.uint8)

        moe_quant_config = int4_w4a16_moe_quant_config(
            w1_scale=w13_weight_scale,
            w2_scale=w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, group_size],
        )

        tuned_config = {}

        for num_tokens in NUM_TOKENS_TO_TUNE:
            score = torch.rand(
                (num_tokens, num_experts), device=device, dtype=torch.float32
            )
            topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
            topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
            x = torch.randn(
                (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
            )
            us_best = float("inf")
            for tile_config in TILE_CONFIGS:
                try:
                    tile_m = tile_config["tile_m"]
                    tile_n = tile_config["tile_n"]
                    tile_k = tile_config["tile_k"]
                    tile_n2 = tile_config["tile_n2"]
                    tile_k2 = tile_config["tile_k2"]

                    model_dim = x.shape[1]
                    assert model_dim % 64 == 0
                    assert model_dim % tile_k == 0
                    assert inter_dim % tile_n == 0
                    assert model_dim % tile_n2 == 0
                    assert inter_dim % tile_k2 == 0
                    assert ((tile_m * tile_k2) % 256) == 0
                    bytes_per_thread_x = (tile_m * tile_k2) // 256
                    assert (bytes_per_thread_x % 4) == 0

                    out, _us = run_perftest(
                        fused_flydsl_moe,
                        x,
                        w13,
                        w2,
                        num_experts,
                        inter_dim,
                        topk_weights,
                        topk_ids,
                        num_iters=num_iters,
                        num_warmup=num_warmup,
                        w1_scale=w13_scale_flydsl,
                        w2_scale=w2_scale_flydsl,
                        topk=topk_weights.shape[-1],
                        group_size=group_size,
                        doweight_stage1=False,
                        scale_is_bf16=True,
                        config=tile_config,
                    )
                    torch.accelerator.synchronize()
                except Exception:
                    torch.accelerator.synchronize()
                    continue
                else:
                    us = _us.item()
                    if us < us_best:
                        out_ref = fused_experts(
                            x,
                            w13_weight_packed,
                            w2_weight_packed,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                            activation=MoEActivation.SILU,
                            apply_router_weight_on_input=False,
                            global_num_experts=num_experts,
                            expert_map=None,
                            quant_config=moe_quant_config,
                        )
                        try:
                            assert torch.allclose(out, out_ref, atol=0.5, rtol=0.1)
                        except Exception:
                            continue
                        else:
                            print(
                                f"For [num_tokens={num_tokens}, num_experts={num_experts}, "  # noqa: E501
                                f"inter_dim={inter_dim}] found new best "  # noqa: E501
                                f"config={tile_config}, us={us:0.3f}"
                            )
                            us_best = us
                            tuned_config[str(num_tokens)] = tile_config
            device_name = current_platform.get_device_name().replace(" ", "_")
            tuned_config_file_name = (
                f"E={num_experts},N={inter_dim},device_name={device_name},"
                f"dtype=int4_w4a16,backend=flydsl.json"
            )
            tuner_dir_path = os.path.dirname(os.path.realpath(__file__))
            store_path = os.path.join(tuner_dir_path, tuned_config_file_name)
            with open(store_path, "w") as f:
                json.dump(tuned_config, f, indent=4)
            print(
                f"\nTuned config for num_tokens={num_tokens} was stored at {store_path}\n"  # noqa: E501
            )


if __name__ == "__main__":
    tune_flydsl_moe_w4a16(device="cuda")
