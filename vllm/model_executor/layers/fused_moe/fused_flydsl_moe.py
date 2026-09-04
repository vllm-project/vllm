# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE Triton kernels."""

import functools
import json
import os

import torch
from aiter.fused_moe import moe_sorting as aiter_moe_sorting
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
    flydsl_a16w4_gemm1,
    flydsl_a16w4_gemm2,
)

from vllm.logger import init_logger
from vllm.utils.platform_utils import get_device_name_as_file_name
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Valid tiles for the aiter a16w-mix int4 kernels (aiter#4646): stage1 requires
# tile_n in {64,128} and tile_k in {128,256} (k_wave=1); stage2 requires tile_n2==128
# and tile_k2 in {128,256}. The pre-#4646 defaults (tile_k=512/64, tile_n2=256) are no
# longer registered and make the port emit an empty-fragment kernel.
_FLYDSL_MOE_DEFAULT_CONFIG = {
    1: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    2: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    4: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    8: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    16: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    24: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    32: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    48: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    64: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    128: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    256: {"tile_m": 16, "tile_n": 128, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    512: {"tile_m": 16, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    1024: {"tile_m": 32, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    2048: {"tile_m": 64, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    4096: {"tile_m": 32, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
    8192: {"tile_m": 64, "tile_n": 64, "tile_k": 128, "tile_n2": 128, "tile_k2": 128},
}

# aiter#4646's a16w-mix int4 kernels only register a fixed tile set. Tiles from before
# the bump (default config or tuned JSONs) carry tile_k=512/64 and tile_n2=256, which
# emit an empty-fragment kernel (IndexError at emit / wrong output). Snap any resolved
# tiles to the nearest registered value so a stale tuned config can't crash the kernel.
_FLYDSL_INT4_TILE_M = (16, 32, 64, 128)
_FLYDSL_INT4_TILE_N1 = (64, 128)
_FLYDSL_INT4_TILE_K = (128, 256)


def _snap_flydsl_int4_tiles(tile_m, tile_n, tile_k, tile_n2, tile_k2):
    def _nearest(v, allowed):
        return min(allowed, key=lambda a: (abs(a - v), a))

    snapped = (
        _nearest(tile_m, _FLYDSL_INT4_TILE_M),
        _nearest(tile_n, _FLYDSL_INT4_TILE_N1),
        _nearest(tile_k, _FLYDSL_INT4_TILE_K),
        128,  # stage2 N tile is only registered at 128
        _nearest(tile_k2, _FLYDSL_INT4_TILE_K),
    )
    if snapped != (tile_m, tile_n, tile_k, tile_n2, tile_k2):
        logger.warning_once(
            "FlyDSL int4 MoE tiles %s are not registered post-aiter#4646; "
            "using %s. Retune the config for the a16w-mix kernel.",
            (tile_m, tile_n, tile_k, tile_n2, tile_k2),
            snapped,
            scope="local",
        )
    return snapped


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
    tile_m, tile_n, tile_k, tile_n2, tile_k2 = _snap_flydsl_int4_tiles(
        tile_m, tile_n, tile_k, tile_n2, tile_k2
    )

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

    # aiter#4646 (v0.1.20) removed compile_moe_gemm{1,2} + the flyc.compile pointer ABI
    # and replaced them with the a16w-mix wrappers, which compile+launch internally
    # (functools-cached) and take torch tensors directly. w_dtype="int4" is the a16wi4
    # (W4A16) path. The stage-1 intermediate now lives in sorted
    # [sorted_size, inter_dim] layout, and topk weighting is applied in stage-2 via
    # sorted_weights (the port has no stage-1 doweight).
    sorted_weights_1d = sorted_weights.view(-1).contiguous()
    empty_scale = torch.empty((0,), device=device, dtype=torch.uint8)
    cumsum_tensor = num_valid_ids.to(torch.int32).contiguous()
    m_indices = sorted_token_ids.to(torch.int32).contiguous()

    # stage1: bf16 A x int4 W1 + SiLU -> bf16 sorted intermediate.
    inter_sorted = torch.empty(
        (sorted_size, inter_dim), device=device, dtype=torch.bfloat16
    )
    w1_scale_u8 = (
        w1_scale.view(torch.uint8).contiguous().view(-1)
        if w1_scale is not None
        else empty_scale
    )
    flydsl_a16w4_gemm1(
        a_bf16=hidden_states.to(torch.bfloat16).contiguous(),
        w1_u8=w1.view(torch.uint8).contiguous(),
        w1_scale_u8=w1_scale_u8,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        m_indices=m_indices,
        inter_sorted_bf16=inter_sorted,
        n_tokens=tokens,
        NE=num_experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=None,
        act="silu",
        w_dtype="int4",
        w_layout="standard",
    )

    # stage2: bf16 inter x int4 W2 -> weighted reduce into [tokens, model_dim].
    out_stage2 = torch.zeros((tokens, model_dim), device=device, dtype=out_torch_dtype)
    w2_scale_u8 = (
        w2_scale.view(torch.uint8).contiguous().view(-1)
        if w2_scale is not None
        else empty_scale
    )
    flydsl_a16w4_gemm2(
        inter_sorted_bf16=inter_sorted,
        w2_u8=w2.view(torch.uint8).contiguous(),
        w2_scale_u8=w2_scale_u8,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights_1d,
        flat_out=out_stage2.view(-1),
        M_logical=tokens,
        max_sorted=sorted_size,
        NE=num_experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n2,
        tile_k=tile_k2,
        waves_per_eu=None,
        w_dtype="int4",
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
