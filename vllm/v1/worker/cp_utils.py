# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""上下文并行工具函数模块。

本模块提供上下文并行（CP）相关的辅助函数，负责：
- 检查注意力层的 CP 兼容性
- 获取总的 CP 世界大小

主要函数：
- check_attention_cp_compatibility: 检查注意力 CP 兼容性
- get_total_cp_world_size: 获取总的 CP 世界大小
"""
from typing import TYPE_CHECKING, Any, cast

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    """检查注意力层的上下文并行兼容性。

    验证注意力层实现是否支持配置的 CP 功能：
    - MTP 与非平凡 interleave size 的配合
    - DCP 需要返回 softmax LSE
    - PCP 需要注意力实现支持

    Args:
        vllm_config: vLLM 配置

    Raises:
        AssertionError: 如果不支持配置的 CP 功能
    """
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "DCP 需要注意力实现返回 decode 的 softmax lse，但实现 "
                    f"{layer_impl.__class__.__name__} "
                    "不返回 decode 的 softmax lse。"
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "PCP 需要注意力实现支持，但实现 "
                    f"{layer_impl.__class__.__name__} "
                    "不支持 PCP。"
                )


def get_total_cp_world_size() -> int:
    """获取总的上下文并行世界大小。

    计算 DCP 和 PCP 的总世界大小。
    如果组未初始化（例如在测试中），则返回 1。

    Returns:
        总的 CP 世界大小
    """
    try:
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP 可能在测试中未初始化
        pcp_world_size = 1
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP 可能在测试中未初始化
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size
