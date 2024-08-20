"""Fused MoE kernel."""

import functools
import json
import os
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import triton
import triton.language as tl

import vllm.model_executor.layers.phi_ops.moe.tensorrt_llm_moe.gather_scatter_kernel as gather_scatter_kernel

import vllm._phi_C

phi_ops_moe_align_block_size = torch.ops._phi_C.moe_align_block_size
phi_ops_grouped_gemm = torch.ops._phi_C.grouped_gemm

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils import print_warning_once

# import pycublas.trtllm_moe_grouped_gemm as phi_ops

logger = init_logger(__name__)

moe_gg_kernel_config = {
    1: (13, 21, 0.4587008017301559),
    2: (5, 11, 0.4829593604803085),
    3: (11, 4, 0.55322624117136),
    4: (5, 5, 0.6300467216968536),
    5: (5, 9, 0.6892339181900025),
    6: (5, 5, 0.7366860777139663),
    7: (17, 9, 0.7817830407619476),
    8: (5, 8, 0.8124313586950302),
    16: (5, 5, 1.0158489656448364),
    32: (4, 17, 1.0969907104969026),
    48: (5, 4, 1.1068108654022217),
    64: (17, 5, 1.1107225465774535),
    80: (4, 5, 1.1139481484889984),
    96: (16, 16, 1.1225907170772553),
    112: (16, 16, 1.1334041678905487),
    128: (17, 17, 1.137500158548355),
    144: (16, 17, 1.144709119796753),
    160: (16, 17, 1.1540889596939088),
    176: (16, 16, 1.1627110350131988),
    192: (17, 16, 1.1790643167495727),
    208: (22, 16, 1.2127846336364747),
    224: (23, 17, 1.2236697602272033),
    240: (22, 22, 1.2352307152748108),
    256: (23, 22, 1.2356915152072907),
    512: (23, 22, 1.6425676786899566),
    768: (27, 27, 1.7934028828144073),
    1024: (27, 23, 2.4730009508132933),
    1280: (22, 22, 3.02405633687973),
    1536: (27, 22, 3.2711680245399477),
    1792: (27, 26, 3.344619517326355),
    2048: (27, 26, 4.023920638561249),
    2304: (26, 22, 4.71138304233551),
    2560: (27, 27, 4.861614079475403),
    2816: (27, 27, 4.988712968826294),
    3072: (26, 27, 5.624104981422424),
    3328: (27, 26, 6.2363647985458375),
    3584: (26, 26, 6.384680962562561),
    3840: (26, 27, 6.581227521896363),
    4096: (26, 27, 7.1324774312973025),
}


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    expert_off = torch.empty(
        (num_experts + 1), dtype=torch.int32, device=topk_ids.device
    )
    expert_length = torch.empty(
        (num_experts + 1), dtype=torch.int32, device=topk_ids.device
    )

    phi_ops_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_off,
        expert_length,
    )
    return (
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_off.to(torch.int64),
        expert_length,
    )


def fused_moe(
    activation: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config: Optional[Dict[str, Any]] = None,
    use_fp8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    routing_func: Callable = torch.topk,
    use_grouped_topk=False,
    num_expert_group=0,
    topk_group=0,
    cfg_id_0=-1,
    cfg_id_1=-1,
) -> torch.Tensor:
    hidden_states_dtype = activation.dtype
    hidden_states = activation #.to(torch.float16)

    # Check constraints.
    M, K = hidden_states.shape
    E, _, N = w1.shape
    block_m = 16
    block_k = 128

    if cfg_id_0 < 1 or cfg_id_1 < 1:
        cfg_id_0, cfg_id_1, _ = moe_gg_kernel_config[min(moe_gg_kernel_config.keys(), key=lambda x: abs(x - M))]

    topk_weights, topk_ids = routing_func(gating_output, topk)

    sorted_token_ids, expert_ids, num_tokens_post_padded, expert_off, expert_length = (
        moe_align_block_size(topk_ids, block_m, E)
    )

    intermediate_cache3 = torch.empty(
        (M, topk, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gathered_cache = torch.empty(
        (sorted_token_ids.size(0), K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gathered_cache_1 = torch.empty(
        (sorted_token_ids.size(0), N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gathered_cache_2 = torch.empty(
        (sorted_token_ids.size(0), N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gathered_cache_3 = torch.empty(
        (sorted_token_ids.size(0), K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # hidden states -> sorted hidden states
    gather_scatter_kernel.invoke_moe_gather(
        hidden_states,
        gathered_cache,
        sorted_token_ids,
        num_tokens_post_padded,
        topk_ids,
        block_m,
        block_k,
        topk,
        4,
    )

    total_rows_before_expert = expert_off[1:]

    phi_ops_grouped_gemm(
        gathered_cache,
        w1,
        w1_scale,
        total_rows_before_expert,
        gathered_cache_1,
        5,
        cfg_id_0,
    )

    ops.silu_and_mul(gathered_cache_2, gathered_cache_1.view(-1, N))

    phi_ops_grouped_gemm(
        gathered_cache_2,
        w2.view(torch.int8),
        w2_scale,
        total_rows_before_expert,
        gathered_cache_3,
        5,
        cfg_id_1,
    )

    gather_scatter_kernel.invoke_moe_scatter(
        gathered_cache_3,
        intermediate_cache3.view(-1, K),
        sorted_token_ids,
        num_tokens_post_padded,
        topk_ids,
        block_m,
        block_k,
        topk,
        4,
        topk_weights=topk_weights,
    )

    intermediate_cache3 = intermediate_cache3[:M, :, :]

    if inplace:
        return torch.sum(
            intermediate_cache3.view(*intermediate_cache3.shape),
            dim=1,
            out=activation,
        )
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)
