# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-level checks for direct weight reload, kept free of model-executor
imports so config hashing and CLI validation stay light."""

from typing import TYPE_CHECKING, Any

import vllm.envs as envs

from .quantization import ONLINE_QUANT_SHORTHAND_NAMES

if TYPE_CHECKING:
    from .vllm import VllmConfig


def uses_online_quantization(vllm_config: "VllmConfig | None") -> bool:
    """Whether `vllm_config` may quantize weights online (a shorthand or an
    explicit `quantization_config`)."""
    model_config = getattr(vllm_config, "model_config", None)
    return model_config is not None and (
        model_config.quantization in ONLINE_QUANT_SHORTHAND_NAMES
        or model_config.quantization_config is not None
    )


def direct_reload_enabled(vllm_config: "VllmConfig | None") -> bool:
    """Whether `vllm_config` selects direct reload: runtime weight updates on a
    backend other than sharded_rdt, and `VLLM_RELOAD_MODE=direct`."""
    transfer = getattr(vllm_config, "weight_transfer_config", None)
    return (
        transfer is not None
        and transfer.backend != "sharded_rdt"
        and envs.VLLM_RELOAD_MODE == "direct"
    )


def direct_reload_config_refusals(vllm_config: "VllmConfig | None") -> list[str]:
    """Reasons a configuration cannot reload directly."""
    if vllm_config is None:
        return []
    reasons = []
    offload = vllm_config.offload_config
    if (
        offload.offload_backend != "auto"
        or offload.prefetch.offload_group_size > 0
        or offload.uva.cpu_offload_gb > 0
    ):
        # CPU/UVA offload gives parameters new storage on every post-load
        # processing; prefetch offload shares one buffer between layers
        reasons.append("weight offloading")
    load_configs: tuple[Any, ...] = (
        getattr(vllm_config, "load_config", None),
        getattr(vllm_config.speculative_config, "draft_load_config", None),
    )
    if any(
        getattr(c, "load_format", None) in ("ipc_cache", "peer_cache")
        for c in load_configs
    ):
        # the weight cache's runtime-layout weights have no checkpoint layout
        reasons.append("pre-processed weights")
    model_config = vllm_config.model_config
    if model_config is not None and model_config.quantization == "torchao":
        # its load_weights runs a layerwise reload, and its post-load
        # processing replaces the weight parameter
        reasons.append("torchao quantization")
    spec = vllm_config.speculative_config
    if spec is not None and spec.draft_lm_head_quantization is not None:
        # after `begin` the draft's quantized lm_head is a view of the target
        # lm_head's storage: post-load processing would write it there
        reasons.append("draft_lm_head_quantization")
    return reasons


def direct_reload_applies(vllm_config: "VllmConfig | None") -> bool:
    """Whether direct reload is enabled and the configuration does not refuse
    it, which quant-method selection must know before loading."""
    return direct_reload_enabled(vllm_config) and not direct_reload_config_refusals(
        vllm_config
    )
