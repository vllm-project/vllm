# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, cast

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object

logger = init_logger(__name__)


def _is_deepseek_v4_config(vllm_config: VllmConfig) -> bool:
    hf_config = vllm_config.model_config.hf_config
    compress_ratios = getattr(hf_config, "compress_ratios", None)
    return compress_ratios is not None and len(compress_ratios) > 0


def _is_deepseek_v4_megamoe_pcp_config(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.prefill_context_parallel_size > 1
        and parallel_config.decode_context_parallel_size == 1
        and parallel_config.enable_expert_parallel
        and vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        and _is_deepseek_v4_config(vllm_config)
    )


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    allow_deepseek_v4_pcp = _is_deepseek_v4_megamoe_pcp_config(vllm_config)
    if allow_deepseek_v4_pcp:
        if (
            vllm_config.cache_config.enable_prefix_caching
            or vllm_config.kv_transfer_config is not None
        ):
            raise ValueError(
                "Experimental DeepSeek V4 MegaMoE PCP does not support prefix "
                "caching or KV transfer yet. Disable prefix caching with "
                "--no-enable-prefix-caching."
            )
        logger.warning_once(
            "Allowing experimental DeepSeek V4 MegaMoE PCP attention path. "
            "Prefix caching and KV transfer are disabled for this path."
        )
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
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )

            if pcp_size > 1:
                if allow_deepseek_v4_pcp:
                    continue
                assert layer_impl.supports_pcp, (
                    "PCP requires attention impls' support, "
                    f"but the impl {layer_impl.__class__.__name__} "
                    "does not support PCP."
                )


def get_total_cp_world_size():
    world_size, _ = get_total_cp_world_size_and_rank()
    return world_size


def get_total_cp_world_size_and_rank() -> tuple[int, int]:
    try:
        pcp_group = get_pcp_group()
        pcp_world_size = pcp_group.world_size
        pcp_rank = pcp_group.rank_in_group
    except AssertionError:
        # PCP might not be initialized in testing
        pcp_world_size = 1
        pcp_rank = 0
    try:
        dcp_group = get_dcp_group()
        dcp_world_size = dcp_group.world_size
        dcp_rank = dcp_group.rank_in_group
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
        dcp_rank = 0

    total_cp_world_size = dcp_world_size * pcp_world_size
    total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
    return total_cp_world_size, total_cp_rank
