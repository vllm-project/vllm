# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, cast

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object

# Backends known to support DCP (decode context parallelism).
# Non-MLA backends:
_DCP_COMPATIBLE_BACKENDS = ["FLASH_ATTN", "FLASHINFER"]
# MLA backends:
_DCP_COMPATIBLE_MLA_BACKENDS = [
    "FLASHMLA",
    "TRITON_MLA",
    "CUTLASS_MLA",
    "FLASH_ATTN_MLA",
]


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
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported by the current attention backend "
                    f"({layer_impl.__class__.__name__}). "
                    f"Try switching to a compatible backend via "
                    f"the --attention-backend CLI flag."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    f"Decode context parallelism (DCP) requires the "
                    f"attention backend to return softmax LSE for "
                    f"decode, but the current backend "
                    f"({layer_impl.__class__.__name__}) does not "
                    f"support this. To fix, switch to a compatible "
                    f"backend via the --attention-backend CLI flag. "
                    f"Compatible backends: "
                    f"{', '.join(_DCP_COMPATIBLE_BACKENDS)} "
                    f"(or for MLA models: "
                    f"{', '.join(_DCP_COMPATIBLE_MLA_BACKENDS)})."
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    f"Prefill context parallelism (PCP) is not "
                    f"supported by the current attention backend "
                    f"({layer_impl.__class__.__name__}). "
                    f"Try switching to a compatible backend via "
                    f"the --attention-backend CLI flag."
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
