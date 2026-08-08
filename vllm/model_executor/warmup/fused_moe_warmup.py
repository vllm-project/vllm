# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile WNA16 MoE Triton kernels through the shared JIT contract."""

import torch

from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16Experts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _WNA16_TRITON_KERNEL,
    Wna16TritonWarmupConfig,
    get_moe_wna16_block_config,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.triton_utils import tl


def _compute_type(dtype: torch.dtype) -> tl.dtype:
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    if dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return tl.bfloat16
    raise ValueError(f"Unsupported WNA16 MoE dtype: {dtype}")


def _kernel_config(
    config: dict[str, int],
    *,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    group_size: int,
    real_top_k: int,
) -> dict[str, int]:
    kernel_config = config.copy()
    kernel_config.update(
        get_moe_wna16_block_config(
            config=kernel_config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_valid_tokens,
            size_k=size_k,
            size_n=size_n,
            num_experts=size_n,
            group_size=group_size,
            real_top_k=real_top_k,
            block_size_m=kernel_config["BLOCK_SIZE_M"],
        )
    )
    return kernel_config


def _warmup_config(
    *,
    dtype: torch.dtype,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    input_size: int,
    config: dict[str, int],
    group_size: int,
    mul_routed_weight: bool,
    top_k: int,
    topk_weights_dtype: torch.dtype | None,
    use_int4_w4a16: bool,
    use_int8_w8a16: bool,
) -> Wna16TritonWarmupConfig:
    return Wna16TritonWarmupConfig(
        a_dtype=dtype,
        b_dtype=weight.dtype,
        c_dtype=dtype,
        b_scale_dtype=scale.dtype,
        b_zp_dtype=None if zero_point is None else zero_point.dtype,
        topk_weights_dtype=topk_weights_dtype,
        N=weight.size(1),
        K=input_size,
        stride_am=input_size,
        stride_ak=1,
        stride_be=weight.stride(0),
        stride_bk=weight.stride(2),
        stride_bn=weight.stride(1),
        stride_cm=weight.size(1),
        stride_cn=1,
        stride_bse=scale.stride(0),
        stride_bsk=scale.stride(2),
        stride_bsn=scale.stride(1),
        stride_bze=zero_point.stride(0) if zero_point is not None else 0,
        stride_bzk=zero_point.stride(2) if zero_point is not None else 0,
        stride_bzn=zero_point.stride(1) if zero_point is not None else 0,
        block_k_diviable=input_size % config["BLOCK_SIZE_K"] == 0,
        group_size=group_size,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        SPLIT_K=config["SPLIT_K"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=_compute_type(dtype),
        has_zp=zero_point is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 3),
    )


def _layer_warmup_configs(
    layer: RoutedExperts,
    experts: TritonWNA16Experts,
    max_num_batched_tokens: int,
) -> list[Wna16TritonWarmupConfig]:
    if max_num_batched_tokens <= 0:
        return []

    dtype = layer.moe_config.in_dtype
    w1 = layer.w13_weight
    w2 = layer.w2_weight
    top_k = layer.top_k
    assert experts.block_shape is not None
    group_size = experts.block_shape[1]
    pack_factor = 2 if experts.quant_config.use_int4_w4a16 else 1
    w1_input_size = w1.size(2) * pack_factor
    w2_input_size = w2.size(2) * pack_factor
    config_name = experts.quant_config.config_name(dtype)
    w1_zp = experts.quant_config.w1_zp
    w2_zp = experts.quant_config.w2_zp

    configs: list[Wna16TritonWarmupConfig] = []
    for num_tokens in range(1, max_num_batched_tokens + 1):
        base_config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k,
            config_name,
            num_tokens,
            block_shape=experts.block_shape,
        )
        num_valid_tokens = num_tokens * top_k
        w1_config = _kernel_config(
            base_config,
            num_valid_tokens=num_valid_tokens,
            size_k=w1_input_size,
            size_n=w1.size(1),
            group_size=group_size,
            real_top_k=top_k,
        )
        w2_config = _kernel_config(
            base_config,
            num_valid_tokens=num_valid_tokens,
            size_k=w2_input_size,
            size_n=w2.size(1),
            group_size=group_size,
            real_top_k=1,
        )
        configs.append(
            _warmup_config(
                dtype=dtype,
                weight=w1,
                scale=experts.w1_scale,
                zero_point=w1_zp,
                input_size=w1_input_size,
                config=w1_config,
                group_size=group_size,
                mul_routed_weight=False,
                top_k=top_k,
                topk_weights_dtype=None,
                use_int4_w4a16=experts.quant_config.use_int4_w4a16,
                use_int8_w8a16=experts.quant_config.use_int8_w8a16,
            )
        )
        configs.append(
            _warmup_config(
                dtype=dtype,
                weight=w2,
                scale=experts.w2_scale,
                zero_point=w2_zp,
                input_size=w2_input_size,
                config=w2_config,
                group_size=group_size,
                mul_routed_weight=not layer.apply_router_weight_on_input,
                top_k=1,
                topk_weights_dtype=torch.float32,
                use_int4_w4a16=experts.quant_config.use_int4_w4a16,
                use_int8_w8a16=experts.quant_config.use_int8_w8a16,
            )
        )
    return configs


def fused_moe_wna16_warmup(
    model: torch.nn.Module,
    max_num_batched_tokens: int,
) -> None:
    configs: list[Wna16TritonWarmupConfig] = []
    for module in model.modules():
        if not isinstance(module, RoutedExperts):
            continue
        quant_method = module.quant_method
        if not isinstance(quant_method, MoeWNA16Method):
            continue
        moe_kernel = quant_method.moe_kernel
        if moe_kernel is None:
            continue
        experts = moe_kernel.fused_experts
        if not isinstance(experts, TritonWNA16Experts):
            continue
        configs.extend(_layer_warmup_configs(module, experts, max_num_batched_tokens))

    _WNA16_TRITON_KERNEL.warmup(configs)
