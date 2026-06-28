# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch.nn as nn

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.models.deepseek_v4.nvidia.dspark import load_dspark_model

logger = init_logger(__name__)


def _get_target_layer_ids(spec_config: SpeculativeConfig) -> tuple[int, ...]:
    if not (spec_config and spec_config.draft_model_config):
        raise RuntimeError("DSpark requires a draft model config.")
    layer_ids = getattr(
        spec_config.draft_model_config.hf_config,
        "dspark_target_layer_ids",
        None,
    )
    if not isinstance(layer_ids, Sequence) or isinstance(layer_ids, str):
        raise RuntimeError("DSpark model config must define dspark_target_layer_ids.")
    result = tuple(int(i) for i in layer_ids)
    if not result:
        raise RuntimeError("DSpark target layer list must not be empty.")
    return result


def set_dspark_aux_hidden_state_layers(
    model: nn.Module,
    spec_config: SpeculativeConfig,
) -> None:
    """Configure target-model auxiliary hidden-state capture for DSpark.

    DeepSeek's reference DSpark code concatenates mean(HC-stream) hidden states
    after the configured target layers. This is not EAGLE3's pre-layer residual
    capture, so keep it as a separate mode.
    """
    layer_ids = _get_target_layer_ids(spec_config)
    parent_ref = (
        model.get_language_model() if hasattr(model, "get_language_model") else model
    )
    if not hasattr(parent_ref, "model"):
        raise RuntimeError("DSpark target model must expose a .model module.")
    inner = parent_ref.model
    if not hasattr(inner, "set_dspark_aux_hidden_state_layers"):
        raise RuntimeError(
            "Target model does not support DSpark auxiliary hidden-state capture."
        )
    inner.set_dspark_aux_hidden_state_layers(layer_ids)
    logger.info("Using DSpark auxiliary target layers: %s", layer_ids)


def load_deepseek_v4_dspark_model(target_model: nn.Module, vllm_config):
    """Load the DeepSeek V4 DSpark draft module from the target checkpoint.

    DSpark shares the target embedding table and lm_head, but its ``mtp.*``
    tensors are a draft-only block model rather than serial MTP layers.
    """
    dspark_model = load_dspark_model(vllm_config)
    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    target_lm_head = getattr(target_language_model, "lm_head", None)
    if target_embed is None:
        raise RuntimeError("DSpark target model does not expose embed_tokens.")
    if target_lm_head is None:
        raise RuntimeError("DSpark target model does not expose lm_head.")
    dspark_model.attach_target_modules(target_embed, target_lm_head)
    return dspark_model
