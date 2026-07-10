# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, cast

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object

# Backends that support Decode Context Parallelism (return softmax LSE on
# decode). Kept as a constant so the error message always reflects the truth.
_DCP_COMPATIBLE_BACKENDS = (
    "FLASH_ATTN",
    "FLASHINFER",
    "FLASH_ATTN_MLA",
    "FLASHMLA",
    "TRITON_MLA",
    "CUTLASS_MLA",
)


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
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
                    f"{layer_impl.__class__.__name__} does not support MTP "
                    "with cp_kv_cache_interleave_size > 1. "
                    "Disable speculative decoding by removing "
                    "--speculative-config (or setting "
                    "--speculative-config '{\"method\": \"none\"}'), or set "
                    "--cp-kv-cache-interleave-size 1."
                )
            if dcp_size > 1:
                _dcp_backends = ", ".join(_DCP_COMPATIBLE_BACKENDS)
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not support "
                    "this. To fix, switch to a DCP-compatible backend "
                    f"({_dcp_backends}) by setting "
                    "--attention-backend <BACKEND> or the environment variable "
                    "VLLM_ATTENTION_BACKEND=<BACKEND>, "
                    "or disable DCP with --decode-context-parallel-size 1."
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "Prefill Context Parallelism (PCP) requires attention "
                    "implementations that support PCP, but "
                    f"{layer_impl.__class__.__name__} does not. "
                    "Try a PCP-compatible backend by setting "
                    "--attention-backend <BACKEND> or the environment variable "
                    "VLLM_ATTENTION_BACKEND=<BACKEND>, "
                    "or disable PCP with --prefill-context-parallel-size 1."
                )


def get_total_cp_world_size():
    try:
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP might not be initialized in testing
        pcp_world_size = 1
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size
