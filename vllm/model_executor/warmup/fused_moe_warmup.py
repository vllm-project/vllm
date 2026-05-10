# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warm up fused MoE kernels that are not reliably exercised by model-level dummy
runs.
"""

from collections.abc import Iterable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import (
    should_moe_wna16_use_cuda,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method

logger = init_logger(__name__)

_WNA16_M_BUCKET_STARTS = (1, 21, 41)


def _wna16_uses_triton_path(
    *,
    m: int,
    call_top_k: int,
    num_experts: int,
    group_size: int,
    weight_bits: int,
) -> bool:
    return not should_moe_wna16_use_cuda(
        num_valid_tokens=m * call_top_k,
        group_size=group_size,
        num_experts=num_experts,
        bit=weight_bits,
    )


def _first_wna16_triton_m(
    *,
    max_num_batched_tokens: int,
    call_top_k: int,
    num_experts: int,
    group_size: int,
    weight_bits: int,
) -> int | None:
    if max_num_batched_tokens <= 0:
        return None

    if _wna16_uses_triton_path(
        m=1,
        call_top_k=call_top_k,
        num_experts=num_experts,
        group_size=group_size,
        weight_bits=weight_bits,
    ):
        return 1

    # should_moe_wna16_use_cuda() uses:
    #   num_valid_tokens / num_experts <= 6
    # so the first Triton M is floor(6 * E / top_k) + 1. Keep the final
    # predicate check below so this stays tied to the real dispatch helper.
    m = (6 * num_experts) // call_top_k + 1
    while m <= max_num_batched_tokens:
        if _wna16_uses_triton_path(
            m=m,
            call_top_k=call_top_k,
            num_experts=num_experts,
            group_size=group_size,
            weight_bits=weight_bits,
        ):
            return m
        m += 1
    return None


def _generate_wna16_triton_m_values(
    *,
    max_num_batched_tokens: int,
    num_experts: int,
    top_k: int,
    group_size: int,
    weight_bits: int,
) -> list[int]:
    if max_num_batched_tokens <= 0 or num_experts <= 0 or top_k <= 0:
        return []

    m_values: set[int] = set()
    # fused_experts_impl dispatches WNA16 twice:
    # - w13/gate-up with the model top-k
    # - w2/down with top_k=1
    # Warm up both dispatch thresholds because either GEMM can be the first
    # one to reach the Triton WNA16 path.
    for call_top_k in {top_k, 1}:
        first_triton_m = _first_wna16_triton_m(
            max_num_batched_tokens=max_num_batched_tokens,
            call_top_k=call_top_k,
            num_experts=num_experts,
            group_size=group_size,
            weight_bits=weight_bits,
        )
        if first_triton_m is None:
            continue

        for bucket_start in _WNA16_M_BUCKET_STARTS:
            m = max(first_triton_m, bucket_start)
            if m > max_num_batched_tokens:
                continue
            if _wna16_uses_triton_path(
                m=m,
                call_top_k=call_top_k,
                num_experts=num_experts,
                group_size=group_size,
                weight_bits=weight_bits,
            ):
                m_values.add(m)

    return sorted(m_values)


def _get_warmup_expert_ids(layer: FusedMoE, device: torch.device) -> torch.Tensor:
    expert_map = layer.expert_map
    if expert_map is None:
        return torch.arange(
            layer.global_num_experts,
            device=device,
            dtype=torch.int32,
        )

    expert_ids = torch.nonzero(expert_map >= 0, as_tuple=False).flatten()
    return expert_ids.to(device=device, dtype=torch.int32)


def _make_warmup_topk_ids(
    m: int,
    top_k: int,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    expert_indices = torch.arange(m * top_k, device=expert_ids.device)
    return expert_ids[expert_indices % expert_ids.numel()].view(m, top_k)


def _get_wna16_moe_warmup_key(layer: FusedMoE) -> tuple[Any, ...]:
    quant_method = layer.quant_method
    assert isinstance(quant_method, MoeWNA16Method)

    w13_zp = getattr(layer, "w13_qzeros", None)
    w2_zp = getattr(layer, "w2_qzeros", None)
    return (
        str(layer.w13_qweight.device),
        layer.moe_config.in_dtype,
        tuple(layer.w13_qweight.shape),
        tuple(layer.w2_qweight.shape),
        tuple(layer.w13_scales.shape),
        tuple(layer.w2_scales.shape),
        None if w13_zp is None else tuple(w13_zp.shape),
        None if w2_zp is None else tuple(w2_zp.shape),
        getattr(layer, "group_size", quant_method.quant_config.group_size),
        layer.top_k,
        layer.activation,
        layer.apply_router_weight_on_input,
    )


def _iter_wna16_moe_layers(model: torch.nn.Module) -> Iterable[FusedMoE]:
    for module in model.modules():
        if isinstance(module, FusedMoE) and isinstance(
            module.quant_method, MoeWNA16Method
        ):
            yield module


def _warmup_wna16_moe_layer(
    layer: FusedMoE,
    m_values: list[int],
) -> None:
    if not m_values:
        return

    layer.ensure_moe_quant_config_init()

    device = layer.w13_qweight.device
    dtype = layer.moe_config.in_dtype
    expert_ids = _get_warmup_expert_ids(layer, device)
    if expert_ids.numel() == 0:
        return

    top_k = layer.top_k
    hidden_dim = layer.moe_config.hidden_dim

    with torch.inference_mode():
        for m in m_values:
            x = torch.zeros((m, hidden_dim), device=device, dtype=dtype)
            topk_ids = _make_warmup_topk_ids(m, top_k, expert_ids)
            topk_weights = torch.full(
                (m, top_k),
                1.0 / top_k,
                device=device,
                dtype=dtype,
            )
            out = None
            try:
                out = layer.quant_method.apply(
                    layer,
                    x,
                    topk_weights,
                    topk_ids,
                    shared_experts_input=None,
                )
                if device.type != "cpu":
                    torch.accelerator.synchronize()
            finally:
                del out, x, topk_ids, topk_weights

    if device.type != "cpu":
        torch.accelerator.empty_cache()


def fused_moe_wna16_warmup(
    model: torch.nn.Module,
    max_num_batched_tokens: int,
) -> None:
    seen: set[tuple[Any, ...]] = set()
    num_warmed = 0

    for layer in _iter_wna16_moe_layers(model):
        warmup_key = _get_wna16_moe_warmup_key(layer)
        if warmup_key in seen:
            continue
        seen.add(warmup_key)

        quant_method = layer.quant_method
        assert isinstance(quant_method, MoeWNA16Method)
        group_size = getattr(layer, "group_size", quant_method.quant_config.group_size)
        num_experts = layer.w13_qweight.size(0)
        m_values = _generate_wna16_triton_m_values(
            max_num_batched_tokens=max_num_batched_tokens,
            num_experts=num_experts,
            top_k=layer.top_k,
            group_size=group_size,
            weight_bits=quant_method.quant_config.weight_bits,
        )
        if not m_values:
            continue

        logger.debug(
            "Warming up WNA16 MoE Triton kernels for layer %s with M values %s",
            getattr(layer, "layer_name", "<unknown>"),
            m_values,
        )
        _warmup_wna16_moe_layer(layer, m_values)
        num_warmed += 1

    if num_warmed:
        logger.info("Warmed up WNA16 MoE Triton kernels for %d config(s).", num_warmed)
