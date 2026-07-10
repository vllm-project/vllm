# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE Triton kernels."""

import functools
import json
import os

import flydsl.compiler as flyc
import torch
from aiter.fused_moe import moe_sorting as aiter_moe_sorting
from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
    compile_moe_gemm1,
    compile_moe_gemm2,
)

from vllm.logger import init_logger
from vllm.utils.platform_utils import get_device_name_as_file_name
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_FLYDSL_MOE_GEMM1_CACHE: dict = {}
_FLYDSL_MOE_GEMM2_CACHE: dict = {}

_FLYDSL_MOE_DEFAULT_CONFIG = {
    1: {"tile_m": 16, "tile_n": 64, "tile_k": 512, "tile_n2": 256, "tile_k2": 256},
    2: {"tile_m": 16, "tile_n": 64, "tile_k": 512, "tile_n2": 256, "tile_k2": 128},
    4: {"tile_m": 16, "tile_n": 64, "tile_k": 512, "tile_n2": 256, "tile_k2": 128},
    8: {"tile_m": 16, "tile_n": 64, "tile_k": 512, "tile_n2": 256, "tile_k2": 256},
    16: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 256},
    24: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 256, "tile_k2": 256},
    32: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 256, "tile_k2": 256},
    48: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 256, "tile_k2": 256},
    64: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    128: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 256, "tile_k2": 256},
    256: {"tile_m": 16, "tile_n": 128, "tile_k": 128, "tile_n2": 256, "tile_k2": 256},
    512: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 256, "tile_k2": 256},
    1024: {"tile_m": 32, "tile_n": 64, "tile_k": 128, "tile_n2": 256, "tile_k2": 256},
    2048: {"tile_m": 64, "tile_n": 64, "tile_k": 64, "tile_n2": 256, "tile_k2": 64},
    4096: {"tile_m": 32, "tile_n": 64, "tile_k": 128, "tile_n2": 256, "tile_k2": 256},
    8192: {"tile_m": 64, "tile_n": 64, "tile_k": 64, "tile_n2": 256, "tile_k2": 64},
}


def moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    block_m: int,
):
    topk_ids_i32 = topk_ids.to(torch.int32)
    topk_w_f32 = topk_weights.to(torch.float32)
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, _moe_buf = (
        aiter_moe_sorting(
            topk_ids_i32,
            topk_w_f32,
            num_experts,
            model_dim,
            torch.float16,
            block_m,
        )
    )
    if num_valid_ids.numel() > 1:
        num_valid_ids = num_valid_ids[:1].contiguous()
    return sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids


def build_routing_buffers(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    model_dim: int,
    tile_m: int,
):
    res = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        model_dim=model_dim,
        block_m=tile_m,
    )
    if res is None:
        raise RuntimeError(
            "aiter moe_sorting failed/unavailable; cannot build routing buffers."
        )
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = res

    sorted_token_ids = sorted_token_ids.contiguous()
    sorted_weights = sorted_weights.contiguous()
    sorted_expert_ids = sorted_expert_ids.contiguous()
    sorted_size = int(sorted_token_ids.numel())
    blocks = int(sorted_expert_ids.numel())
    return (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    )


@functools.lru_cache
def try_get_optimal_config(num_experts, inter_dim):
    device_name = get_device_name_as_file_name()
    json_file_name = (
        f"E={num_experts},N={inter_dim},device_name={device_name},"
        "dtype=int4_w4a16,backend=flydsl.json"
    )
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info_once(
                "Using tuned FlyDSL MoE config from %s",
                config_file_path,
                scope="global",
            )
            tuned_config = json.load(f)
            return {int(key): val for key, val in tuned_config.items()}

    logger.warning_once(
        "Using default FlyDSL MoE config. Performance might be sub-optimal! "
        "Config file not found at %s",
        config_file_path,
        scope="local",
    )
    return _FLYDSL_MOE_DEFAULT_CONFIG


def fused_flydsl_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    inter_dim: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    topk: int = 8,
    group_size: int = 32,
    doweight_stage1: bool = False,
    in_dtype: str = "int4_bf16",
    out_dtype: str = "bf16",
    scale_is_bf16: bool = True,
    tile_m: int | None = None,
    tile_n: int | None = None,
    tile_k: int | None = None,
    tile_n2: int | None = None,
    tile_k2: int | None = None,
) -> torch.Tensor:
    device = hidden_states.device
    tokens = hidden_states.shape[0]
    model_dim = hidden_states.shape[1]

    tuned_config = {}
    if tile_m and tile_n and tile_k and tile_n2 and tile_k2:
        tuned_config["tile_m"] = tile_m
        tuned_config["tile_n"] = tile_n
        tuned_config["tile_k"] = tile_k
        tuned_config["tile_n2"] = tile_n2
        tuned_config["tile_k2"] = tile_k2
    else:
        tuned_config = try_get_optimal_config(num_experts, inter_dim)
        tuned_config = tuned_config[
            min(tuned_config.keys(), key=lambda x: abs(x - tokens))
        ]
    out_torch_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    tile_m = tuned_config["tile_m"]
    tile_n = tuned_config["tile_n"]
    tile_k = tuned_config["tile_k"]
    tile_n2 = tuned_config["tile_n2"]
    tile_k2 = tuned_config["tile_k2"]

    routing = build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_experts=num_experts,
        model_dim=model_dim,
        tile_m=tile_m,
    )
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    ) = routing

    scale_x_1d = torch.empty((0,), device=device, dtype=torch.float32)
    sorted_weights_1d = sorted_weights.view(-1).contiguous()
    out_stage1 = torch.empty(
        (tokens, topk, inter_dim), device=device, dtype=out_torch_dtype
    )

    stream = torch.cuda.current_stream()

    key1 = (
        model_dim,
        inter_dim,
        num_experts,
        topk,
        in_dtype,
        out_dtype,
        group_size,
        tile_m,
        tile_n,
        tile_k,
        bool(doweight_stage1),
        False,
    )

    compiled_exe1 = _FLYDSL_MOE_GEMM1_CACHE.get(key1)
    if compiled_exe1 is None:
        exe1 = compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=num_experts,
            topk=topk,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            group_size=group_size,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=bool(doweight_stage1),
            use_cshuffle_epilog=False,
            scale_is_bf16=scale_is_bf16,
        )
        compiled_exe1 = flyc.compile(
            exe1,
            out_stage1,
            hidden_states,
            w1,
            scale_x_1d,
            w1_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights_1d,
            num_valid_ids,
            tokens,
            inter_dim,
            model_dim,
            int(blocks),
            stream,
        )
        _FLYDSL_MOE_GEMM1_CACHE[key1] = compiled_exe1

    compiled_exe1(
        out_stage1,
        hidden_states,
        w1,
        scale_x_1d,
        w1_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
        num_valid_ids,
        tokens,
        inter_dim,
        model_dim,
        int(blocks),
        stream,
    )

    a2_1d = out_stage1.view(-1).contiguous()
    a2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32)
    out_stage2 = torch.empty((tokens, model_dim), device=device, dtype=out_torch_dtype)
    doweight_stage2 = not bool(doweight_stage1)

    key2 = (
        model_dim,
        inter_dim,
        num_experts,
        topk,
        in_dtype,
        out_dtype,
        group_size,
        tile_m,
        tile_n2,
        tile_k2,
        bool(doweight_stage2),
    )

    compiled_exe2 = _FLYDSL_MOE_GEMM2_CACHE.get(key2)
    if compiled_exe2 is None:
        exe2 = compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=num_experts,
            topk=topk,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            group_size=group_size,
            tile_m=tile_m,
            tile_n=tile_n2,
            tile_k=tile_k2,
            doweight_stage2=bool(doweight_stage2),
            scale_is_bf16=scale_is_bf16,
        )
        compiled_exe2 = flyc.compile(
            exe2,
            out_stage2,
            a2_1d,
            w2,
            a2_scale_1d,
            w2_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights_1d,
            num_valid_ids,
            tokens,
            model_dim,
            inter_dim,
            int(blocks),
            stream,
        )
        _FLYDSL_MOE_GEMM2_CACHE[key2] = compiled_exe2

    out_stage2.zero_()
    compiled_exe2(
        out_stage2,
        a2_1d,
        w2,
        a2_scale_1d,
        w2_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
        num_valid_ids,
        tokens,
        model_dim,
        inter_dim,
        int(blocks),
        stream,
    )
    return out_stage2


def fused_flydsl_moe_impl_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    inter_dim: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    topk: int = 8,
    group_size: int = 32,
    doweight_stage1: bool = False,
    in_dtype: str = "int4_bf16",
    out_dtype: str = "bf16",
    scale_is_bf16: bool = True,
    tile_m: int | None = None,
    tile_n: int | None = None,
    tile_k: int | None = None,
    tile_n2: int | None = None,
    tile_k2: int | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="fused_flydsl_moe_impl",
    op_func=fused_flydsl_moe_impl,
    fake_impl=fused_flydsl_moe_impl_fake,
)


def fused_flydsl_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    inter_dim: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    topk: int = 8,
    group_size: int = 32,
    doweight_stage1: bool = False,
    in_dtype: str = "int4_bf16",
    out_dtype: str = "bf16",
    scale_is_bf16: bool = True,
    config: dict | None = None,
) -> torch.Tensor:
    tile_m = None
    tile_n = None
    tile_k = None
    tile_n2 = None
    tile_k2 = None
    if config is not None:
        tile_m = config.get("tile_m")
        tile_n = config.get("tile_n")
        tile_k = config.get("tile_k")
        tile_n2 = config.get("tile_n2")
        tile_k2 = config.get("tile_k2")
    return torch.ops.vllm.fused_flydsl_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        num_experts=num_experts,
        inter_dim=inter_dim,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk=topk,
        group_size=group_size,
        doweight_stage1=doweight_stage1,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scale_is_bf16=scale_is_bf16,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        tile_n2=tile_n2,
        tile_k2=tile_k2,
    )
